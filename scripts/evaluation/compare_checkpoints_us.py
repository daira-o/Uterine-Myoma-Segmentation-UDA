"""
Compare two US segmentation checkpoints visually.

This script runs inference with two Attention U-Net segmenters extracted from
DANN checkpoints. GRL and domain discriminator weights are ignored.

Example:
    python scripts/evaluation/compare_checkpoints_us.py ^
        --us_dir data_ready_US ^
        --num_samples 20 ^
        --threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from config import CONFIG
except Exception:
    CONFIG = {
        "in_channels": 1,
        "num_classes": 1,
        "base_filters": 64,
        "threshold": 0.5,
    }

from models.attention_unet import AttentionUNet
from scripts.inference.postprocessing import (
    postprocess_prediction,
    save_postprocessing_debug_overlay,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = "lesion"

    def clipped(self, size: int = 256) -> "BBox | None":
        xmin = min(max(float(self.xmin), 0.0), float(size))
        xmax = min(max(float(self.xmax), 0.0), float(size))
        ymin = min(max(float(self.ymin), 0.0), float(size))
        ymax = min(max(float(self.ymax), 0.0), float(size))
        if xmax <= xmin or ymax <= ymin:
            return None
        return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=self.label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara visualmente inferencias US entre dos checkpoints."
    )
    parser.add_argument(
        "--checkpoint_a",
        type=Path,
        default=None,
        help="Checkpoint A. Si se omite junto con B, se usan los ultimos dos de --checkpoint_dir.",
    )
    parser.add_argument(
        "--checkpoint_b",
        type=Path,
        default=None,
        help="Checkpoint B. Si se omite junto con A, se usan los ultimos dos de --checkpoint_dir.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=ROOT / "logs" / "checkpoints_dann",
        help="Directorio donde buscar automaticamente checkpoints .pth.",
    )
    parser.add_argument(
        "--auto_select",
        choices=("best", "latest"),
        default="best",
        help="Criterio automatico si no se pasan checkpoints: best usa Dice guardado; latest usa fecha/epoca.",
    )
    parser.add_argument(
        "--us_dir",
        type=Path,
        default=ROOT / "data_ready_US",
        help="Directorio US procesado. Puede ser la raiz con splits o un split/images.",
    )
    parser.add_argument(
        "--bbox_manifest",
        type=Path,
        default=None,
        help="JSON opcional con bboxes. Si se omite, busca split/bboxes/<stem>.json.",
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "checkpoint_comparison")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=float(CONFIG.get("threshold", 0.5)))
    parser.add_argument("--min_area_px", type=int, default=50)
    parser.add_argument("--bbox_margin_px", type=int, default=8)
    parser.add_argument("--closing_kernel_px", type=int, default=3)
    parser.add_argument(
        "--postprocess_debug_dir",
        type=Path,
        default=ROOT / "outputs" / "postprocessing_debug",
        help="Directorio para overlays debug antes/despues del postprocessing.",
    )
    parser.add_argument("--no_postprocess_debug", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint_a_label", default=None)
    parser.add_argument("--checkpoint_b_label", default=None)
    parser.add_argument("--no_bbox", action="store_true", help="No dibujar bbox aunque exista.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidio --device cuda pero CUDA no esta disponible.")
    return torch.device(name)


def checkpoint_epoch(path: Path) -> int | None:
    matches = re.findall(r"(?:epoch|ep|ckpt)[_\- ]?(\d+)|(?:^|[_\- ])(\d+)(?:\.pth$)", path.name, re.IGNORECASE)
    numbers = [int(group) for match in matches for group in match if group]
    return max(numbers) if numbers else None


def checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    epoch = checkpoint_epoch(path)
    if epoch is not None:
        return (1, float(epoch), path.name.lower())
    return (0, path.stat().st_mtime, path.name.lower())


def checkpoint_dice(path: Path) -> float | None:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        log.warning("No se pudo leer metricas de %s: %s", path.name, exc)
        return None
    if not isinstance(checkpoint, dict):
        return None
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("Dice") or metrics.get("dice") or metrics.get("val_dice")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_checkpoint_candidates(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No existe checkpoint_dir: {checkpoint_dir}")
    candidates = [path for path in checkpoint_dir.glob("*.pth") if path.is_file()]
    if len(candidates) < 2:
        raise FileNotFoundError(
            f"Se necesitan al menos dos checkpoints .pth en {checkpoint_dir}. "
            f"Encontrados: {len(candidates)}"
        )
    return candidates


def find_latest_checkpoints(checkpoint_dir: Path) -> tuple[Path, Path]:
    candidates = sorted(
        list_checkpoint_candidates(checkpoint_dir),
        key=checkpoint_sort_key,
        reverse=True,
    )
    newer, older = candidates[0], candidates[1]
    return older, newer


def find_best_checkpoints(checkpoint_dir: Path) -> tuple[Path, Path]:
    scored: list[tuple[float, tuple[int, float, str], Path]] = []
    unscored: list[Path] = []
    for path in list_checkpoint_candidates(checkpoint_dir):
        dice = checkpoint_dice(path)
        if dice is None:
            unscored.append(path)
            continue
        scored.append((dice, checkpoint_sort_key(path), path))

    if len(scored) < 2:
        log.warning(
            "No hay dos checkpoints con metrics['Dice']; usando ultimos por fecha/epoca."
        )
        return find_latest_checkpoints(checkpoint_dir)

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_two = [scored[0][2], scored[1][2]]
    # A queda como el segundo mejor/mas viejo, B como el mejor para leer la figura como mejora.
    checkpoint_b, checkpoint_a = best_two[0], best_two[1]
    return checkpoint_a, checkpoint_b


def resolve_checkpoint_pair(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.checkpoint_a is None and args.checkpoint_b is None:
        if args.auto_select == "best":
            checkpoint_a, checkpoint_b = find_best_checkpoints(args.checkpoint_dir)
        else:
            checkpoint_a, checkpoint_b = find_latest_checkpoints(args.checkpoint_dir)
        log.info(
            "Checkpoints auto (%s): A=%s | B=%s",
            args.auto_select,
            checkpoint_a,
            checkpoint_b,
        )
        return checkpoint_a, checkpoint_b
    if args.checkpoint_a is None or args.checkpoint_b is None:
        raise ValueError(
            "Pasa --checkpoint_a y --checkpoint_b juntos, o no pases ninguno "
            "para seleccion automatica."
        )
    return args.checkpoint_a, args.checkpoint_b


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)
    if arr.shape != (256, 256):
        raise ValueError(f"Se esperaba imagen US 256x256, shape recibido: {arr.shape}")
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def discover_us_images(us_dir: Path) -> list[Path]:
    if not us_dir.exists():
        raise FileNotFoundError(f"No existe us_dir: {us_dir}")
    if us_dir.is_file() and us_dir.suffix.lower() == ".npy":
        return [us_dir]

    patterns = [
        "*.npy",
        "images/*.npy",
        "*/images/*.npy",
        "**/images/*.npy",
    ]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in us_dir.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda p: str(p).lower())


def clean_state_key(key: str) -> str:
    prefixes = ("model.", "module.", "segmenter.", "attention_unet.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def is_domain_key(key: str) -> bool:
    clean = clean_state_key(key)
    return clean.startswith(("domain_discriminator.", "grl."))


def extract_segmenter_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    Extract AttentionUNet weights from DANN checkpoints or plain state_dicts.

    Explicitly ignores GRL and domain discriminator. Supports common prefixes:
    model., module., segmenter., attention_unet.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint

    if "segmenter_state_dict" in checkpoint:
        raw_state = checkpoint["segmenter_state_dict"]
    elif "model_state_dict" in checkpoint:
        raw_state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        raw_state = checkpoint["state_dict"]
    else:
        raw_state = checkpoint

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        if is_domain_key(key):
            continue
        clean_key = clean_state_key(key)
        if clean_key.startswith(("domain_discriminator.", "grl.")):
            continue
        cleaned[clean_key] = value

    if not cleaned:
        raise ValueError("No se pudieron extraer pesos del segmenter desde el checkpoint.")
    return cleaned


def checkpoint_config(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return checkpoint["config"]
    return {}


def load_segmenter(checkpoint_path: Path, device: torch.device) -> AttentionUNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_config = checkpoint_config(checkpoint)
    state_dict = extract_segmenter_state_dict(checkpoint)

    model = AttentionUNet(
        in_channels=int(ckpt_config.get("in_channels", CONFIG.get("in_channels", 1))),
        num_classes=int(ckpt_config.get("num_classes", CONFIG.get("num_classes", 1))),
        base_filters=int(ckpt_config.get("base_filters", CONFIG.get("base_filters", 64))),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("%s: pesos faltantes: %s", checkpoint_path.name, list(missing))
    if unexpected:
        log.warning("%s: pesos inesperados ignorados: %s", checkpoint_path.name, list(unexpected))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_probability(model: AttentionUNet, image_np: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)


def bbox_json_path(image_path: Path) -> Path:
    if image_path.parent.name == "images":
        return image_path.parent.parent / "bboxes" / f"{image_path.stem}.json"
    return image_path.parent / "bboxes" / f"{image_path.stem}.json"


def rows_to_bboxes(rows: Any) -> list[BBox]:
    boxes: list[BBox] = []
    if not isinstance(rows, list):
        return boxes
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            box = BBox(
                xmin=float(row["xmin"]),
                ymin=float(row["ymin"]),
                xmax=float(row["xmax"]),
                ymax=float(row["ymax"]),
                label=str(row.get("label", "lesion")),
            ).clipped()
        except (KeyError, TypeError, ValueError):
            continue
        if box is not None:
            boxes.append(box)
    return boxes[:1]


def load_bbox_manifest(path: Path | None) -> dict[str, list[BBox]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"No existe bbox_manifest: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    mapping: dict[str, list[BBox]] = {}
    if isinstance(payload, dict):
        items = payload.items()
        for key, value in items:
            rows = value.get("bbox_256", value.get("bboxes", value)) if isinstance(value, dict) else value
            boxes = rows_to_bboxes(rows)
            if boxes:
                mapping[str(key)] = boxes
                mapping[Path(str(key)).name] = boxes
                mapping[Path(str(key)).stem] = boxes
    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = row.get("filename") or row.get("processed_npy") or row.get("image") or row.get("path")
            rows = row.get("bbox_256") or row.get("bboxes")
            if key is None:
                continue
            boxes = rows_to_bboxes(rows)
            if boxes:
                mapping[str(key)] = boxes
                mapping[Path(str(key)).name] = boxes
                mapping[Path(str(key)).stem] = boxes
    return mapping


def load_bboxes(image_path: Path, manifest: dict[str, list[BBox]]) -> list[BBox]:
    for key in (str(image_path), image_path.name, image_path.stem):
        if key in manifest:
            return manifest[key]

    json_path = bbox_json_path(image_path)
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return rows_to_bboxes(payload.get("bbox_256", []))[:1]


def boxes_to_mask(boxes: list[BBox], size: int = 256) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for box in boxes:
        xmin = int(np.floor(box.xmin))
        ymin = int(np.floor(box.ymin))
        xmax = int(np.ceil(box.xmax))
        ymax = int(np.ceil(box.ymax))
        xmin = min(max(xmin, 0), size)
        xmax = min(max(xmax, 0), size)
        ymin = min(max(ymin, 0), size)
        ymax = min(max(ymax, 0), size)
        if xmax > xmin and ymax > ymin:
            mask[ymin:ymax, xmin:xmax] = True
    return mask


def prediction_metrics(mask: np.ndarray, boxes: list[BBox]) -> dict[str, Any]:
    pred = mask.astype(bool)
    pred_area = int(pred.sum())
    if not boxes:
        return {
            "pred_area": pred_area,
            "bbox_in": None,
            "bbox_iou": None,
            "centroid_inside": None,
        }

    bbox_mask = boxes_to_mask(boxes)
    intersection = int((pred & bbox_mask).sum())
    union = int((pred | bbox_mask).sum())
    bbox_in = float(intersection / pred_area) if pred_area > 0 else 0.0
    bbox_iou = float(intersection / union) if union > 0 else 0.0

    points = np.argwhere(pred)
    if len(points) == 0:
        centroid_inside: bool | None = False
    else:
        cy, cx = points.mean(axis=0)
        y = int(np.clip(round(cy), 0, pred.shape[0] - 1))
        x = int(np.clip(round(cx), 0, pred.shape[1] - 1))
        centroid_inside = bool(bbox_mask[y, x])

    return {
        "pred_area": pred_area,
        "bbox_in": bbox_in,
        "bbox_iou": bbox_iou,
        "centroid_inside": centroid_inside,
    }


def metric_text(metrics: dict[str, Any]) -> str:
    if metrics["bbox_in"] is None:
        return f"area={metrics['pred_area']} px\nbbox=N/A"
    centroid = "inside" if metrics["centroid_inside"] else "outside"
    return (
        f"area={metrics['pred_area']} px\n"
        f"in_bbox={metrics['bbox_in']:.3f} | IoU={metrics['bbox_iou']:.3f}\n"
        f"centroid={centroid}"
    )


def draw_bboxes(ax: plt.Axes, boxes: list[BBox], color: str = "#00e5ff") -> None:
    for box in boxes:
        rect = Rectangle(
            (box.xmin, box.ymin),
            box.xmax - box.xmin,
            box.ymax - box.ymin,
            fill=False,
            edgecolor=color,
            linewidth=1.7,
        )
        ax.add_patch(rect)


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def render_comparison(
    image_path: Path,
    image_np: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    boxes: list[BBox],
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    label_a: str,
    label_b: str,
    threshold: float,
    output_path: Path,
    draw_bbox: bool,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), facecolor="white", constrained_layout=True)
    fig.suptitle(f"{image_path.name} | threshold={threshold:.3f}", fontsize=11)

    panels = [
        ("US preprocesada", image_np, "gray", None),
        (f"{label_a}\n{metric_text(metrics_a)}", mask_a.astype(float), "gray", None),
        (f"Overlay {label_a}", image_np, "gray", ("autumn", mask_a)),
        (f"{label_b}\n{metric_text(metrics_b)}", mask_b.astype(float), "gray", None),
        (f"Overlay {label_b}", image_np, "gray", ("cool", mask_b)),
    ]

    for ax, (title, base, cmap, overlay) in zip(axes, panels):
        ax.imshow(base, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        if overlay is not None:
            overlay_cmap, overlay_mask = overlay
            masked = np.ma.masked_where(overlay_mask == 0, overlay_mask)
            ax.imshow(masked, cmap=overlay_cmap, alpha=0.42, interpolation="nearest")
        if draw_bbox and boxes:
            draw_bboxes(ax, boxes)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples debe ser mayor que 0.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold debe estar entre 0 y 1.")

    device = resolve_device(args.device)
    checkpoint_a, checkpoint_b = resolve_checkpoint_pair(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_a = args.checkpoint_a_label or checkpoint_a.stem
    label_b = args.checkpoint_b_label or checkpoint_b.stem

    log.info("Device: %s", device)
    log.info("Cargando checkpoint A: %s", checkpoint_a)
    model_a = load_segmenter(checkpoint_a, device)
    log.info("Cargando checkpoint B: %s", checkpoint_b)
    model_b = load_segmenter(checkpoint_b, device)

    images = discover_us_images(args.us_dir)
    if not images:
        raise FileNotFoundError(f"No se encontraron .npy en {args.us_dir}")

    rng = random.Random(args.seed)
    selected = list(images)
    rng.shuffle(selected)
    selected = selected[: min(args.num_samples, len(selected))]
    log.info("Imagenes disponibles: %d | seleccionadas: %d", len(images), len(selected))

    manifest = load_bbox_manifest(args.bbox_manifest)
    if args.bbox_manifest is not None:
        log.info("BBox manifest cargado: %s (%d claves)", args.bbox_manifest, len(manifest))

    rows: list[dict[str, Any]] = []
    for idx, image_path in enumerate(selected, start=1):
        log.info("[%d/%d] %s", idx, len(selected), image_path.name)
        image_np = normalize_image(np.load(image_path))
        boxes = load_bboxes(image_path, manifest)

        prob_a = predict_probability(model_a, image_np, device)
        prob_b = predict_probability(model_b, image_np, device)
        post_a = postprocess_prediction(
            prob=prob_a,
            boxes=boxes,
            threshold=args.threshold,
            min_area_px=args.min_area_px,
            bbox_margin_px=args.bbox_margin_px,
            closing_kernel_px=args.closing_kernel_px,
        )
        post_b = postprocess_prediction(
            prob=prob_b,
            boxes=boxes,
            threshold=args.threshold,
            min_area_px=args.min_area_px,
            bbox_margin_px=args.bbox_margin_px,
            closing_kernel_px=args.closing_kernel_px,
        )
        mask_a = post_a.mask_final
        mask_b = post_b.mask_final

        if not args.no_postprocess_debug:
            debug_base = args.postprocess_debug_dir.resolve()
            save_postprocessing_debug_overlay(
                image_np=image_np,
                result=post_a,
                boxes=boxes,
                output_dir=debug_base / safe_stem(checkpoint_a),
                stem=f"{idx:03d}_{image_path.stem}_{label_a}",
            )
            save_postprocessing_debug_overlay(
                image_np=image_np,
                result=post_b,
                boxes=boxes,
                output_dir=debug_base / safe_stem(checkpoint_b),
                stem=f"{idx:03d}_{image_path.stem}_{label_b}",
            )

        log.info(
            (
                "%s post A: objects=%d->%d area=%d->%d selected=%s overlap=%d "
                "centroid_inside=%s bbox_in=%s removed_small=%d discarded=%s overlaps=%s"
            ),
            image_path.name,
            post_a.stats["objects_before"],
            post_a.stats["objects_final"],
            post_a.stats["area_before"],
            post_a.stats["area_final"],
            post_a.stats["selected_component_label"],
            post_a.stats["selected_overlap_bbox_px"],
            post_a.stats["centroid_inside_bbox"],
            (
                "N/A"
                if post_a.stats["bbox_in_ratio"] is None
                else f"{post_a.stats['bbox_in_ratio']:.3f}"
            ),
            post_a.stats["removed_by_small_objects_count"],
            post_a.stats["discarded_component_labels"],
            [
                (
                    item["label"],
                    item["overlap_expanded_bbox_px"],
                    item["overlap_bbox_px"],
                )
                for item in post_a.stats["component_overlaps_before"]
            ],
        )
        log.info(
            (
                "%s post B: objects=%d->%d area=%d->%d selected=%s overlap=%d "
                "centroid_inside=%s bbox_in=%s removed_small=%d discarded=%s overlaps=%s"
            ),
            image_path.name,
            post_b.stats["objects_before"],
            post_b.stats["objects_final"],
            post_b.stats["area_before"],
            post_b.stats["area_final"],
            post_b.stats["selected_component_label"],
            post_b.stats["selected_overlap_bbox_px"],
            post_b.stats["centroid_inside_bbox"],
            (
                "N/A"
                if post_b.stats["bbox_in_ratio"] is None
                else f"{post_b.stats['bbox_in_ratio']:.3f}"
            ),
            post_b.stats["removed_by_small_objects_count"],
            post_b.stats["discarded_component_labels"],
            [
                (
                    item["label"],
                    item["overlap_expanded_bbox_px"],
                    item["overlap_bbox_px"],
                )
                for item in post_b.stats["component_overlaps_before"]
            ],
        )

        metrics_a = prediction_metrics(mask_a, boxes)
        metrics_b = prediction_metrics(mask_b, boxes)

        out_png = output_dir / f"{idx:03d}_{safe_stem(image_path)}__{safe_stem(checkpoint_a)}_vs_{safe_stem(checkpoint_b)}.png"
        render_comparison(
            image_path=image_path,
            image_np=image_np,
            prob_a=prob_a,
            prob_b=prob_b,
            mask_a=mask_a,
            mask_b=mask_b,
            boxes=boxes,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            label_a=label_a,
            label_b=label_b,
            threshold=args.threshold,
            output_path=out_png,
            draw_bbox=not args.no_bbox,
        )

        rows.append(
            {
                "filename": str(image_path),
                "pred_area_A": metrics_a["pred_area"],
                "pred_area_B": metrics_b["pred_area"],
                "bbox_in_A": metrics_a["bbox_in"],
                "bbox_in_B": metrics_b["bbox_in"],
                "bbox_iou_A": metrics_a["bbox_iou"],
                "bbox_iou_B": metrics_b["bbox_iou"],
                "centroid_inside_A": metrics_a["centroid_inside"],
                "centroid_inside_B": metrics_b["centroid_inside"],
                "objects_before_A": post_a.stats["objects_before"],
                "objects_before_B": post_b.stats["objects_before"],
                "objects_final_A": post_a.stats["objects_final"],
                "objects_final_B": post_b.stats["objects_final"],
                "selected_component_A": post_a.stats["selected_component_label"],
                "selected_component_B": post_b.stats["selected_component_label"],
                "selected_overlap_bbox_A": post_a.stats["selected_overlap_bbox_px"],
                "selected_overlap_bbox_B": post_b.stats["selected_overlap_bbox_px"],
                "figure": str(out_png),
            }
        )

    csv_path = output_dir / "comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "filename",
            "pred_area_A",
            "pred_area_B",
            "bbox_in_A",
            "bbox_in_B",
            "bbox_iou_A",
            "bbox_iou_B",
            "centroid_inside_A",
            "centroid_inside_B",
            "objects_before_A",
            "objects_before_B",
            "objects_final_A",
            "objects_final_B",
            "selected_component_A",
            "selected_component_B",
            "selected_overlap_bbox_A",
            "selected_overlap_bbox_B",
            "figure",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})

    log.info("Figuras guardadas en: %s", output_dir)
    log.info("CSV resumen: %s", csv_path)


if __name__ == "__main__":
    main()
