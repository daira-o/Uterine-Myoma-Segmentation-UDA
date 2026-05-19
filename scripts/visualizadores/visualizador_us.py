"""
Visualizador rapido de ultrasonido procesado (.npy) con bbox Pascal VOC.

Muestra la imagen original y, al lado, el tile generado por us_pipeline.py
con la bbox transformada a coordenadas 256x256.

Uso:
    python scripts/visualizadores/visualizador_us.py
    python scripts/visualizadores/visualizador_us.py --base data_ready_US_v2 --split test
    python scripts/visualizadores/visualizador_us.py --save-preview outputs/us_preview.png --no-show
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ORIGINAL_ROOT = PROJECT_ROOT / "data" / "Ultrasound"

from scripts.us_pipeline import BBox, FOV_MAP, UltrasoundPipeline


def default_processed_base() -> Path:
    """Prefiere data_ready_US_v2 si existe; si no, usa data_ready_US."""
    v2 = PROJECT_ROOT / "data_ready_US_v2"
    if v2.exists():
        return v2
    return PROJECT_ROOT / "data_ready_US"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualizador rapido de US procesado.")
    parser.add_argument(
        "--base",
        type=str,
        default=str(default_processed_base()),
        help="Carpeta base procesada, ej: data_ready_US o data_ready_US_v2.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Split a visualizar.",
    )
    parser.add_argument(
        "--original-root",
        type=str,
        default=str(DEFAULT_ORIGINAL_ROOT),
        help="Carpeta con imagenes originales data/Ultrasound.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Indice fijo dentro de la lista ordenada. Si no se indica, usa aleatorio.",
    )
    parser.add_argument(
        "--save-preview",
        type=str,
        default=None,
        help="Guarda una previsualizacion PNG.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abre ventana interactiva. Util con --save-preview.",
    )
    parser.add_argument(
        "--show-original-bbox",
        action="store_true",
        help="Tambien dibuja la bbox original sobre la imagen original.",
    )
    return parser.parse_args()


def list_npy_paths(base: Path, split: str) -> list[Path]:
    if split == "all":
        patterns = [
            str(base / "*" / "images" / "*.npy"),
            str(base / "images" / "*.npy"),
        ]
    else:
        patterns = [str(base / split / "images" / "*.npy")]
    paths = sorted({Path(p) for pattern in patterns for p in glob(pattern)})
    return paths


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.clip(arr, 0.0, 1.0)


def parse_processed_name(path: Path) -> tuple[str | None, str | None]:
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def find_original(processed_path: Path, original_root: Path) -> Path | None:
    fov, source_stem = parse_processed_name(processed_path)
    if fov is None or source_stem is None:
        return None

    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = original_root / fov / f"{source_stem}{ext}"
        if candidate.exists():
            return candidate

    hits = sorted(original_root.rglob(f"{source_stem}.*"))
    for hit in hits:
        if hit.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return hit
    return None


def load_original(path: Path | None) -> np.ndarray | None:
    if path is None or not PIL_OK:
        return None
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def read_original_bboxes(xml_path: Path | None) -> list[BBox]:
    if xml_path is None or not xml_path.exists():
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    boxes: list[BBox] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="lesion")
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            boxes.append(
                BBox(
                    xmin=float(bnd.findtext("xmin", "0")),
                    ymin=float(bnd.findtext("ymin", "0")),
                    xmax=float(bnd.findtext("xmax", "0")),
                    ymax=float(bnd.findtext("ymax", "0")),
                    label=label,
                )
            )
        except ValueError:
            continue
    return boxes


def bbox_json_path(processed_path: Path) -> Path:
    if processed_path.parent.name == "images":
        return processed_path.parent.parent / "bboxes" / f"{processed_path.stem}.json"
    return processed_path.parent / "bboxes" / f"{processed_path.stem}.json"


def load_bboxes_from_json(processed_path: Path) -> list[BBox]:
    json_path = bbox_json_path(processed_path)
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    boxes: list[BBox] = []
    for row in payload.get("bbox_256", []):
        boxes.append(
            BBox(
                xmin=float(row["xmin"]),
                ymin=float(row["ymin"]),
                xmax=float(row["xmax"]),
                ymax=float(row["ymax"]),
                label=str(row.get("label", "lesion")),
            )
        )
    return boxes


def transform_bboxes_on_the_fly(processed_path: Path, original_path: Path | None) -> list[BBox]:
    fov, _source_stem = parse_processed_name(processed_path)
    if original_path is None or fov not in FOV_MAP:
        return []
    pipeline = UltrasoundPipeline()
    sample = pipeline._process_single(str(original_path), FOV_MAP[fov])
    if sample is None:
        return []
    return sample.transformed_bboxes


def load_transformed_bboxes(
    processed_path: Path,
    original_path: Path | None,
) -> tuple[list[BBox], str]:
    boxes = load_bboxes_from_json(processed_path)
    if boxes:
        return boxes, "json"
    boxes = transform_bboxes_on_the_fly(processed_path, original_path)
    if boxes:
        return boxes, "pipeline"
    return [], "sin bbox"


def draw_boxes(ax, boxes: list[BBox], color: str, linewidth: float) -> None:
    for box in boxes:
        rect = Rectangle(
            (box.xmin, box.ymin),
            box.xmax - box.xmin,
            box.ymax - box.ymin,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
        ax.add_patch(rect)


def describe_tile(tile: np.ndarray) -> str:
    black_fraction = float((tile <= 1e-4).mean())
    return (
        f"shape={tile.shape} | rango=[{tile.min():.3f}, {tile.max():.3f}] | "
        f"fondo~{black_fraction:.1%}"
    )


def render_sample(
    axes: np.ndarray,
    processed_path: Path,
    original_root: Path,
    show_original_bbox: bool,
) -> None:
    tile = load_npy(processed_path)
    original_path = find_original(processed_path, original_root)
    original = load_original(original_path)
    original_xml = original_path.with_suffix(".xml") if original_path else None
    original_boxes = read_original_bboxes(original_xml)
    transformed_boxes, bbox_source = load_transformed_bboxes(processed_path, original_path)

    for ax in axes:
        ax.clear()
        ax.axis("off")

    if original is not None:
        axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
        if show_original_bbox:
            draw_boxes(axes[0], original_boxes, color="#00d1ff", linewidth=1.8)
        axes[0].set_title(
            f"Original\n{original_path.parent.name}/{original_path.name}",
            fontsize=9,
        )
    else:
        axes[0].text(
            0.5,
            0.5,
            "Original no encontrada",
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[0].set_title("Original", fontsize=9)

    axes[1].imshow(tile, cmap="gray", vmin=0, vmax=1)
    draw_boxes(axes[1], transformed_boxes, color="#ffd400", linewidth=2.2)
    axes[1].set_title(
        f"Procesada + bbox\n{processed_path.name}\n"
        f"{describe_tile(tile)} | boxes={len(transformed_boxes)} ({bbox_source})",
        fontsize=9,
    )


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    original_root = Path(args.original_root)

    npy_paths = list_npy_paths(base, args.split)
    if not npy_paths:
        print(f"[ERROR] No se encontraron .npy en base={base} split={args.split}")
        print("Ejecuta primero scripts/us_pipeline.py o revisa --base/--split.")
        return

    if args.index is not None:
        current_idx = max(0, min(args.index, len(npy_paths) - 1))
    else:
        current_idx = random.randrange(len(npy_paths))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plt.subplots_adjust(bottom=0.18, wspace=0.05)

    def show_index(idx: int) -> None:
        nonlocal current_idx
        current_idx = idx % len(npy_paths)
        render_sample(
            axes,
            npy_paths[current_idx],
            original_root,
            show_original_bbox=args.show_original_bbox,
        )
        fig.suptitle(
            f"US procesado | split={args.split} | {current_idx + 1}/{len(npy_paths)}",
            fontsize=12,
            fontweight="bold",
        )
        fig.canvas.draw_idle()

    show_index(current_idx)

    if args.save_preview:
        out = Path(args.save_preview)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Preview guardada en: {out}")

    if args.no_show:
        plt.close(fig)
        return

    ax_prev = plt.axes([0.28, 0.04, 0.16, 0.07])
    ax_next = plt.axes([0.56, 0.04, 0.16, 0.07])
    btn_prev = Button(ax_prev, "Anterior")
    btn_next = Button(ax_next, "Siguiente")
    btn_prev.on_clicked(lambda _event: show_index(current_idx - 1))
    btn_next.on_clicked(lambda _event: show_index(current_idx + 1))

    plt.show()


if __name__ == "__main__":
    main()
