"""
scripts/training/train_target_pu.py
Entrenamiento experimental PU-inspired MRI -> US para la Attention U-Net.

Parte de train_target.py, pero cambia la interpretacion weak US:
fuera de bbox es background confiable; dentro de bbox es region candidata /
unlabeled, no una mascara positiva completa. Mantiene DANN apagado por defecto.

Prioriza segmentacion supervisada MRI, supervision weak PU-inspired US:
    total = lambda_mri * seg_loss_mri
          + lambda_us * weak_loss_us
          + lambda_domain * domain_loss
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import logging
import os
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import CONFIG
from models.attention_unet import bce_dice_loss, compute_all_metrics
from models.dann_unet import DANNUNet
from models.grl import compute_lambda_schedule
from scripts.training.train_source import SagitalDataset, load_split_paths, log_split_diagnostics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


MRI_DOMAIN = 0
US_DOMAIN = 1
WEAK_MODES = ("bbox", "pseudomask")
WEAK_LOSS_TYPES = ("soft_bbox", "legacy_bbox")

# Overrides locales del experimento PU. No tocar train_target.py: esta variante
# guarda sus artefactos por separado y mantiene DANN desactivado por defecto.
CONFIG["lambda_domain_base"] = float(os.getenv("LAMBDA_DOMAIN_BASE", "0.0"))
CONFIG["weak_debug_dir"] = os.getenv(
    "WEAK_DEBUG_DIR",
    str(Path(CONFIG["outputs_path"]) / "debug_weak_supervision_pu"),
)


class UltrasoundDataset(Dataset):
    """
    Dataset target weak.

    Devuelve imagen US, target weak y bbox-mask. En modo bbox, target weak y
    bbox-mask son iguales. En modo pseudomask, target weak viene de una mascara
    pseudo y bbox-mask se mantiene para metricas bbox_in/IoU.
    """

    def __init__(
        self,
        img_paths: list[str],
        image_size: int = 256,
        weak_mode: str = "bbox",
    ) -> None:
        if not img_paths:
            raise ValueError("UltrasoundDataset recibio una lista vacia.")
        if weak_mode not in WEAK_MODES:
            raise ValueError(f"weak_mode no soportado: {weak_mode}")
        self.img_paths = img_paths
        self.image_size = image_size
        self.weak_mode = weak_mode

    def __len__(self) -> int:
        return len(self.img_paths)

    @staticmethod
    def bbox_json_path(img_path: str) -> Path:
        path = Path(img_path)
        if path.parent.name == "images":
            return path.parent.parent / "bboxes" / f"{path.stem}.json"
        return path.parent / "bboxes" / f"{path.stem}.json"

    @staticmethod
    def load_bbox_rows(json_path: Path) -> list[dict]:
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("bbox_256", [])

    @staticmethod
    def pseudomask_path(img_path: str) -> Path:
        path = Path(img_path)
        split_dir = path.parent.parent if path.parent.name == "images" else path.parent
        for dirname in ("pseudomasks", "pseudo_masks", "masks"):
            candidate = split_dir / dirname / f"{path.stem}.npy"
            if candidate.exists():
                return candidate
        return split_dir / "pseudomasks" / f"{path.stem}.npy"

    def _bbox_mask(self, img_path: str) -> np.ndarray:
        json_path = self.bbox_json_path(img_path)
        if not json_path.exists():
            raise FileNotFoundError(f"No existe bbox JSON para US: {json_path}")

        rows = self.load_bbox_rows(json_path)
        if not rows:
            raise ValueError(f"BBox vacia en {json_path}")

        mask = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        for row in rows[:1]:
            raw_xmin = float(row["xmin"])
            raw_ymin = float(row["ymin"])
            raw_xmax = float(row["xmax"])
            raw_ymax = float(row["ymax"])
            if raw_xmax <= raw_xmin or raw_ymax <= raw_ymin:
                raise ValueError(f"BBox invalida en {json_path}: {row}")
            xmin = int(np.floor(raw_xmin))
            ymin = int(np.floor(raw_ymin))
            xmax = int(np.ceil(raw_xmax))
            ymax = int(np.ceil(raw_ymax))
            xmin = min(max(xmin, 0), self.image_size)
            xmax = min(max(xmax, 0), self.image_size)
            ymin = min(max(ymin, 0), self.image_size)
            ymax = min(max(ymax, 0), self.image_size)
            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    f"BBox fuera de rango tras clipping en {json_path}: "
                    f"xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}"
                )
            mask[:, ymin:ymax, xmin:xmax] = 1.0
        return mask

    def _pseudomask(self, img_path: str) -> np.ndarray:
        mask_path = self.pseudomask_path(img_path)
        if not mask_path.exists():
            raise FileNotFoundError(
                "No existe pseudomask para US: "
                f"{mask_path}. Esperado en pseudomasks/, pseudo_masks/ o masks/."
            )
        mask = np.load(mask_path).astype(np.float32)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        elif mask.ndim == 3 and mask.shape[0] != 1:
            mask = np.moveaxis(mask, -1, 0)
        return np.clip(mask, 0.0, 1.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        img = np.load(img_path).astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 3 and img.shape[0] != 1:
            img = np.moveaxis(img, -1, 0)
        bbox_mask = self._bbox_mask(img_path)
        weak_mask = self._pseudomask(img_path) if self.weak_mode == "pseudomask" else bbox_mask
        return torch.from_numpy(img), torch.from_numpy(weak_mask), torch.from_numpy(bbox_mask)


def load_us_image_paths(
    base_path: str,
    split: str = "train",
    require_bboxes: bool = True,
    weak_mode: str = "bbox",
) -> list[str]:
    """Carga imagenes US desde <base_path>/<split>/images/*.npy, con bbox asociada."""
    img_dir = os.path.join(base_path, split, "images")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
    if not imgs:
        raise FileNotFoundError(
            f"No se encontraron .npy en '{img_dir}'. "
            "Ejecuta scripts/data_preparation/build_us_splits_from_clean.py tras la curacion "
            "o revisa US_READY_PATH."
        )

    if require_bboxes:
        usable = []
        skipped = 0
        for img_path in imgs:
            json_path = UltrasoundDataset.bbox_json_path(img_path)
            if not json_path.exists():
                skipped += 1
                continue
            rows = UltrasoundDataset.load_bbox_rows(json_path)
            has_pseudomask = (
                weak_mode != "pseudomask"
                or UltrasoundDataset.pseudomask_path(img_path).exists()
            )
            if rows and has_pseudomask:
                usable.append(img_path)
            else:
                skipped += 1
        imgs = usable
        if not imgs:
            raise FileNotFoundError(
                f"No se encontraron imagenes US validas para weak_mode={weak_mode} en '{img_dir}'."
            )
        if skipped:
            log.warning("Split US %-5s -> %d imagenes omitidas sin bbox valida.", split, skipped)

    log.info("Split US %-5s -> %d imagenes cargadas.", split, len(imgs))
    return imgs


def get_us_base_path() -> str:
    """
    Prioriza el dataset curado porque la limpieza visual debe ocurrir antes
    del split final. Se puede sobreescribir con US_READY_PATH sin tocar config.py.
    """
    return (
        os.getenv("US_READY_PATH")
        or CONFIG.get("us_ready_path")
        or os.path.join(str(ROOT), "data_ready_US")
    )


def set_decoder_trainable(model: DANNUNet, trainable: bool) -> None:
    """Congela/descongela solo decoder + salida; encoder y discriminator siguen activos."""
    decoder_modules = [
        model.segmenter.gate4,
        model.segmenter.gate3,
        model.segmenter.gate2,
        model.segmenter.gate1,
        model.segmenter.att4,
        model.segmenter.att3,
        model.segmenter.att2,
        model.segmenter.att1,
        model.segmenter.up4,
        model.segmenter.up3,
        model.segmenter.up2,
        model.segmenter.up1,
        model.segmenter.dec4,
        model.segmenter.dec3,
        model.segmenter.dec2,
        model.segmenter.dec1,
        model.segmenter.output_conv,
    ]
    for module in decoder_modules:
        for param in module.parameters():
            param.requires_grad = trainable


def build_optimizer(model: DANNUNet) -> torch.optim.Optimizer:
    """Param groups con LR diferenciado para encoder, decoder y discriminator."""
    base_lr = CONFIG["lr"]
    encoder_lr = CONFIG.get("dann_encoder_lr", base_lr * 0.1)
    decoder_lr = CONFIG.get("dann_decoder_lr", base_lr * 0.05)
    discriminator_lr = CONFIG.get("dann_discriminator_lr", base_lr)

    encoder_params = list(model.segmenter.enc1.parameters())
    encoder_params += list(model.segmenter.enc2.parameters())
    encoder_params += list(model.segmenter.enc3.parameters())
    encoder_params += list(model.segmenter.enc4.parameters())
    encoder_params += list(model.segmenter.bottleneck.parameters())

    decoder_params = [
        param
        for name, param in model.segmenter.named_parameters()
        if not name.startswith(("enc1.", "enc2.", "enc3.", "enc4.", "bottleneck."))
    ]

    return torch.optim.Adam(
        [
            {"params": encoder_params, "lr": encoder_lr, "name": "encoder"},
            {"params": decoder_params, "lr": decoder_lr, "name": "decoder"},
            {
                "params": model.domain_discriminator.parameters(),
                "lr": discriminator_lr,
                "name": "discriminator",
            },
        ]
    )


def domain_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits.detach(), dim=1)
    return float((preds == labels).float().mean().item())


def weak_bbox_loss(
    seg_logits: torch.Tensor,
    bbox_mask: torch.Tensor,
    inside_fraction: float = 0.10,
    inside_weight: float = 1.0,
    outside_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Supervision debil por bbox.

    - Penaliza probabilidad fuera del rectangulo.
    - Dentro del rectangulo usa una loss tipo MIL sobre los pixeles mas activos.
      Asi se incentiva localizar lesion sin convertir toda la bbox en mascara.
    """
    probs = torch.sigmoid(seg_logits.float())
    bbox_mask = bbox_mask.float()
    outside_mask = 1.0 - bbox_mask

    outside_area = outside_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    outside_loss = (probs * outside_mask).sum(dim=(1, 2, 3)) / outside_area

    inside_losses = []
    for sample_probs, sample_bbox in zip(probs, bbox_mask):
        inside_probs = sample_probs[sample_bbox.bool()]
        if inside_probs.numel() == 0:
            inside_losses.append(sample_probs.new_tensor(0.0))
            continue

        k = max(1, int(round(float(inside_probs.numel()) * inside_fraction)))
        topk_mean = inside_probs.topk(k).values.mean().clamp(min=eps, max=1.0 - eps)
        inside_losses.append(-torch.log(topk_mean))

    inside_loss = torch.stack(inside_losses)
    return (inside_weight * inside_loss + outside_weight * outside_loss).mean()


def expand_bbox_mask(bbox_mask: torch.Tensor, expand_ratio: float) -> torch.Tensor:
    """Expande bbox estricta solo para outside_loss, manteniendo coordenadas 256x256."""
    bbox_mask = (bbox_mask.float() > 0.5).float()
    expand_ratio = float(expand_ratio)
    if expand_ratio <= 0:
        return bbox_mask

    expanded_samples = []
    for sample in bbox_mask:
        coords = torch.nonzero(sample[0] > 0.5, as_tuple=False)
        if coords.numel() == 0:
            expanded_samples.append(sample)
            continue
        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values
        bbox_h = int(y_max.item() - y_min.item() + 1)
        bbox_w = int(x_max.item() - x_min.item() + 1)
        margin_px = max(1, int(np.ceil(max(bbox_h, bbox_w) * expand_ratio)))
        kernel_size = int(2 * margin_px + 1)
        expanded = F.max_pool2d(
            sample.unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=margin_px,
        ).squeeze(0)
        expanded_samples.append(expanded)
    return torch.stack(expanded_samples, dim=0)


def soft_bbox_loss(
    pred_logits: torch.Tensor,
    bbox_mask: torch.Tensor,
    bbox_margin_px: int = 8,
    bbox_expand_ratio: float = 0.10,
    outside_weight: float = 1.0,
    inside_weight: float = 0.5,
    area_weight: float = 0.05,
    min_inside_activation: float = 0.20,
    min_area_ratio: float = 0.10,
    max_area_ratio: float = 1.5,
    under_area_weight: float = 2.0,
    over_area_weight: float = 1.0,
    inside_topk_fraction: float = 0.10,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Supervision debil PU-inspired por bbox unica.

    - Fuera de bbox expandida: background confiable.
    - Dentro de bbox estricta: region unlabeled/candidata, no mascara positiva
      completa. Solo se pide presencia top-k para evitar asumir que toda la bbox
      es mioma.
    - Area usa la bbox estricta como prior suave de tamano, no como etiqueta
      pixel-wise positiva.
    """
    probs = torch.sigmoid(pred_logits.float())
    bbox_mask = (bbox_mask.float() > 0.5).float()
    inside_topk_fraction = min(max(float(inside_topk_fraction), eps), 1.0)
    expanded_bbox = expand_bbox_mask(bbox_mask, bbox_expand_ratio)
    outside_mask = 1.0 - expanded_bbox

    outside_targets = torch.zeros_like(pred_logits, dtype=torch.float32)
    # AMP forbids BCE on sigmoid probabilities. The outside term is a standard
    # negative-class BCE, so the logits variant is numerically safer and keeps
    # the weak bbox objective equivalent for this component.
    outside_loss_map = F.binary_cross_entropy_with_logits(
        pred_logits.float(),
        outside_targets,
        reduction="none",
    )
    # Once the empty-mask collapse is fixed, the main failure mode becomes
    # spatial drift: visible probability mass outside the annotated anatomy.
    # This term is the localization brake, so its default is intentionally back
    # at 1.0 while inside/area terms keep the mask from disappearing.
    # The expanded bbox must stay tight: debug PNGs showed activations on bright
    # echogenic structures outside the annotation, so expansion only tolerates
    # small annotation errors. If 0.10 is still permissive, try 0.05.
    outside_area = outside_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    outside_loss = (outside_loss_map * outside_mask).sum(dim=(1, 2, 3)) / outside_area

    inside_losses = []
    inside_means = []
    inside_maxes = []
    inside_topk_means = []
    for sample_probs, sample_bbox in zip(probs, bbox_mask):
        inside_probs = sample_probs[sample_bbox.bool()]
        if inside_probs.numel() == 0:
            inside_losses.append(sample_probs.new_tensor(0.0))
            inside_means.append(sample_probs.new_tensor(0.0))
            inside_maxes.append(sample_probs.new_tensor(0.0))
            inside_topk_means.append(sample_probs.new_tensor(0.0))
            continue
        k = max(1, int(round(float(inside_probs.numel()) * inside_topk_fraction)))
        topk_mean = inside_probs.topk(k).values.mean()
        mean_inside = inside_probs.mean()
        inside_means.append(mean_inside)
        inside_maxes.append(inside_probs.max())
        inside_topk_means.append(topk_mean)
        # PU-inspired MIL: the bbox is not treated as a dense positive mask.
        # We only require a small high-probability subset inside the candidate
        # region, while outside the expanded bbox remains reliable background.
        inside_losses.append(F.relu(min_inside_activation - topk_mean))
    inside_presence_loss = torch.stack(inside_losses)
    topk_inside_prob = torch.stack(inside_topk_means)
    mean_inside_prob = torch.stack(inside_means)
    max_inside_prob = torch.stack(inside_maxes)

    pred_area = probs.sum(dim=(1, 2, 3))
    bbox_area = bbox_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    predicted_area_ratio = pred_area / bbox_area
    # Use sigmoid probability mass, not a thresholded mask, so the loss remains
    # differentiable. Penalize both empty masks and masks much larger than the
    # bbox; the previous one-sided max-only term made area_loss=0 for collapse.
    # A max_area_ratio around 0.5-0.6 is stricter than the old 1.5 because bbox
    # supervision is coarse: allowing more area than the full box encouraged
    # masks that were visible but poorly localized.
    area_loss = (
        under_area_weight * F.relu(min_area_ratio - predicted_area_ratio)
        + over_area_weight * F.relu(predicted_area_ratio - max_area_ratio)
    )

    total = (
        outside_weight * outside_loss
        + inside_weight * inside_presence_loss
        + area_weight * area_loss
    )
    components = {
        "outside_loss": outside_loss.mean(),
        "inside_presence_loss": inside_presence_loss.mean(),
        "area_loss": area_loss.mean(),
        "weak_us_total": total.mean(),
        "predicted_area_ratio": predicted_area_ratio.mean(),
        "min_area_ratio": probs.new_tensor(float(min_area_ratio)),
        "max_area_ratio": probs.new_tensor(float(max_area_ratio)),
        "mean_inside_prob": mean_inside_prob.mean(),
        "max_inside_prob": max_inside_prob.mean(),
        "topk_inside_prob": topk_inside_prob.mean(),
        "expanded_bbox_mask": expanded_bbox.detach(),
        "outside_penalty_map": (outside_loss_map * outside_mask).detach(),
    }
    return total.mean(), components


def compute_us_weak_loss(
    seg_logits: torch.Tensor,
    weak_mask: torch.Tensor,
    weak_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Selecciona supervision US: bbox weak o pseudomask Dice+BCE."""
    if weak_mode == "bbox":
        weak_loss_type = str(CONFIG.get("weak_loss_type", "soft_bbox"))
        if weak_loss_type == "soft_bbox":
            return soft_bbox_loss(
                seg_logits,
                weak_mask,
                bbox_margin_px=int(CONFIG.get("weak_bbox_margin_px", 8)),
                bbox_expand_ratio=float(CONFIG.get("weak_bbox_expand_ratio", 0.10)),
                outside_weight=float(CONFIG.get("weak_outside_weight", 1.0)),
                inside_weight=float(CONFIG.get("weak_inside_weight", 0.5)),
                area_weight=float(CONFIG.get("weak_area_weight", 0.05)),
                min_inside_activation=float(CONFIG.get("weak_min_inside_activation", 0.20)),
                min_area_ratio=float(CONFIG.get("weak_min_area_ratio", 0.10)),
                max_area_ratio=float(CONFIG.get("weak_max_area_ratio", 1.5)),
                under_area_weight=float(CONFIG.get("weak_under_area_weight", 2.0)),
                over_area_weight=float(CONFIG.get("weak_over_area_weight", 1.0)),
                inside_topk_fraction=float(CONFIG.get("weak_inside_fraction", 0.10)),
            )
        if weak_loss_type == "legacy_bbox":
            loss = weak_bbox_loss(
                seg_logits,
                weak_mask,
                inside_fraction=CONFIG.get("weak_inside_fraction", 0.10),
                inside_weight=CONFIG.get("weak_inside_weight", 1.0),
                outside_weight=CONFIG.get("weak_outside_weight", 1.0),
            )
            zero = loss.detach().new_tensor(0.0)
            return loss, {
                "outside_loss": zero,
                "inside_presence_loss": zero,
                "area_loss": zero,
                "weak_us_total": loss.detach(),
                "predicted_area_ratio": zero,
                "min_area_ratio": zero,
                "max_area_ratio": zero,
                "mean_inside_prob": zero,
                "max_inside_prob": zero,
                "topk_inside_prob": zero,
            }
        raise ValueError(f"weak_loss_type no soportado: {weak_loss_type}")
    if weak_mode == "pseudomask":
        loss = bce_dice_loss(seg_logits, weak_mask.float())
        zero = loss.detach().new_tensor(0.0)
        return loss, {
            "outside_loss": zero,
            "inside_presence_loss": zero,
            "area_loss": zero,
            "weak_us_total": loss.detach(),
            "predicted_area_ratio": zero,
            "min_area_ratio": zero,
            "max_area_ratio": zero,
            "mean_inside_prob": zero,
            "max_inside_prob": zero,
            "topk_inside_prob": zero,
        }
    raise ValueError(f"weak_mode no soportado: {weak_mode}")


def compute_us_bbox_metrics(
    seg_logits: torch.Tensor,
    bbox_mask: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Metricas weak US contra el rectangulo, no contra una mascara real."""
    pred = (torch.sigmoid(seg_logits.detach()) >= threshold).float()
    bbox = (bbox_mask.detach() > 0.5).float()

    pred_area = pred.sum(dim=(1, 2, 3))
    bbox_area = bbox.sum(dim=(1, 2, 3))
    intersection = (pred * bbox).sum(dim=(1, 2, 3))
    union = ((pred + bbox) > 0).float().sum(dim=(1, 2, 3))

    inside_ratio = torch.where(
        pred_area > 0,
        intersection / pred_area.clamp_min(eps),
        torch.zeros_like(pred_area),
    )
    bbox_iou = torch.where(
        union > 0,
        intersection / union.clamp_min(eps),
        torch.zeros_like(union),
    )

    batch_size, _, height, width = pred.shape
    y_coords = torch.arange(height, device=pred.device, dtype=pred.dtype).view(1, 1, height, 1)
    x_coords = torch.arange(width, device=pred.device, dtype=pred.dtype).view(1, 1, 1, width)
    centroid_y = (pred * y_coords).sum(dim=(1, 2, 3)) / pred_area.clamp_min(eps)
    centroid_x = (pred * x_coords).sum(dim=(1, 2, 3)) / pred_area.clamp_min(eps)
    centroid_y = centroid_y.round().clamp(0, height - 1).long()
    centroid_x = centroid_x.round().clamp(0, width - 1).long()
    sample_idx = torch.arange(batch_size, device=pred.device)
    centroid_inside = torch.where(
        pred_area > 0,
        bbox[sample_idx, 0, centroid_y, centroid_x],
        torch.zeros_like(pred_area),
    )

    return {
        "bbox_inside_ratio": float(inside_ratio.mean().item()),
        "bbox_iou": float(bbox_iou.mean().item()),
        "bbox_centroid_inside": float(centroid_inside.mean().item()),
        "predicted_area_ratio": float((pred_area / bbox_area.clamp_min(eps)).mean().item()),
    }


def parse_us_metric_thresholds() -> list[float]:
    raw = str(CONFIG.get("us_metric_thresholds", "0.30,0.35,0.40,0.50"))
    thresholds: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            log.warning("Threshold US invalido ignorado: %s", item)
            continue
        if 0.0 < value < 1.0:
            thresholds.append(value)
    return thresholds or [float(CONFIG["threshold"])]


def summarize_threshold_area_ratios(
    seg_logits: torch.Tensor,
    bbox_mask: torch.Tensor,
    thresholds: list[float],
) -> str:
    parts = []
    for threshold in thresholds:
        metrics = compute_us_bbox_metrics(seg_logits, bbox_mask, threshold=threshold)
        parts.append(f"t{threshold:.2f}:area={metrics['predicted_area_ratio']:.3f},in={metrics['bbox_inside_ratio']:.3f}")
    return " | ".join(parts)


def validate_weak_bbox_spatial_consistency(
    weak_masks: torch.Tensor,
    bbox_masks: torch.Tensor,
    epoch: int,
    step: int,
) -> None:
    """Fail fast if bbox-mode weak masks and metric bbox masks diverge spatially."""
    if weak_masks.shape != bbox_masks.shape:
        raise ValueError(
            f"Shape mismatch weak/bbox en ep={epoch} step={step}: "
            f"{tuple(weak_masks.shape)} vs {tuple(bbox_masks.shape)}"
        )
    bbox_binary = (bbox_masks == 0) | (bbox_masks == 1)
    if not bool(bbox_binary.all().item()):
        raise ValueError(f"BBox mask no binaria en ep={epoch} step={step}.")
    if bool((bbox_masks.sum(dim=(1, 2, 3)) <= 0).any().item()):
        raise ValueError(f"BBox mask vacia en ep={epoch} step={step}.")
    if not bool(torch.equal((weak_masks > 0.5), (bbox_masks > 0.5))):
        diff = torch.abs(weak_masks.float() - bbox_masks.float()).sum().item()
        raise ValueError(
            f"En weak_mode=bbox, weak_mask y bbox_mask deben ser la misma region base "
            f"(ep={epoch}, step={step}, diff={diff:.1f})."
        )


def validate_mri(
    model: DANNUNet,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[dict, int, float]:
    """Validacion MRI: loss de segmentacion + metricas heredadas de Fase 1."""
    model.eval()
    batch_losses, batch_dices, batch_hd95s, batch_oprecs = [], [], [], []
    inf_count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            seg_logits, _ = model(images, alpha=0.0, return_segmentation=True)
            assert seg_logits is not None
            batch_losses.append(float(bce_dice_loss(seg_logits, masks).item()))

            metrics = compute_all_metrics(
                seg_logits,
                masks,
                threshold=CONFIG["threshold"],
                iou_threshold=CONFIG["iou_threshold"],
            )
            batch_dices.append(metrics["Dice"])
            batch_oprecs.append(metrics["Object_Precision"])
            if np.isfinite(metrics["HD95"]):
                batch_hd95s.append(metrics["HD95"])
            else:
                inf_count += 1

    return (
        {
            "Dice": float(np.mean(batch_dices)) if batch_dices else 0.0,
            "HD95": float(np.mean(batch_hd95s)) if batch_hd95s else np.inf,
            "Object_Precision": float(np.mean(batch_oprecs)) if batch_oprecs else 0.0,
        },
        inf_count,
        float(np.mean(batch_losses)) if batch_losses else 0.0,
    )


def validate_us_weak(
    model: DANNUNet,
    val_loader: DataLoader | None,
    device: torch.device,
    weak_mode: str,
) -> dict[str, float]:
    """Validacion weak US: loss weak y localizacion respecto a bbox rectangular."""
    if val_loader is None:
        return {
            "weak_loss_us": 0.0,
            "outside_loss": 0.0,
            "inside_presence_loss": 0.0,
            "area_loss": 0.0,
            "bbox_inside_ratio": 0.0,
            "bbox_iou": 0.0,
            "bbox_centroid_inside": 0.0,
            "predicted_area_ratio": 0.0,
            "mean_inside_prob": 0.0,
            "max_inside_prob": 0.0,
            "topk_inside_prob": 0.0,
        }

    model.eval()
    losses = []
    metrics = {
        "outside_loss": [],
        "inside_presence_loss": [],
        "area_loss": [],
        "bbox_inside_ratio": [],
        "bbox_iou": [],
        "bbox_centroid_inside": [],
        "predicted_area_ratio": [],
        "mean_inside_prob": [],
        "max_inside_prob": [],
        "topk_inside_prob": [],
    }
    with torch.no_grad():
        for images, weak_masks, bbox_masks in val_loader:
            images = images.to(device)
            weak_masks = weak_masks.to(device)
            bbox_masks = bbox_masks.to(device)
            seg_logits, _ = model(images, alpha=0.0, return_segmentation=True)
            assert seg_logits is not None
            weak_loss, weak_parts = compute_us_weak_loss(seg_logits, weak_masks, weak_mode)
            losses.append(float(weak_loss.item()))
            for key in (
                "outside_loss",
                "inside_presence_loss",
                "area_loss",
                "mean_inside_prob",
                "max_inside_prob",
                "topk_inside_prob",
            ):
                metrics[key].append(float(weak_parts[key].item()))
            batch_metrics = compute_us_bbox_metrics(
                seg_logits,
                bbox_masks,
                threshold=CONFIG["threshold"],
            )
            for key, value in batch_metrics.items():
                metrics[key].append(value)

    return {
        "weak_loss_us": float(np.mean(losses)) if losses else 0.0,
        "outside_loss": float(np.mean(metrics["outside_loss"])) if metrics["outside_loss"] else 0.0,
        "inside_presence_loss": float(np.mean(metrics["inside_presence_loss"]))
        if metrics["inside_presence_loss"]
        else 0.0,
        "area_loss": float(np.mean(metrics["area_loss"])) if metrics["area_loss"] else 0.0,
        "bbox_inside_ratio": float(np.mean(metrics["bbox_inside_ratio"]))
        if metrics["bbox_inside_ratio"]
        else 0.0,
        "bbox_iou": float(np.mean(metrics["bbox_iou"])) if metrics["bbox_iou"] else 0.0,
        "bbox_centroid_inside": float(np.mean(metrics["bbox_centroid_inside"]))
        if metrics["bbox_centroid_inside"]
        else 0.0,
        "predicted_area_ratio": float(np.mean(metrics["predicted_area_ratio"]))
        if metrics["predicted_area_ratio"]
        else 0.0,
        "mean_inside_prob": float(np.mean(metrics["mean_inside_prob"]))
        if metrics["mean_inside_prob"]
        else 0.0,
        "max_inside_prob": float(np.mean(metrics["max_inside_prob"]))
        if metrics["max_inside_prob"]
        else 0.0,
        "topk_inside_prob": float(np.mean(metrics["topk_inside_prob"]))
        if metrics["topk_inside_prob"]
        else 0.0,
    }


_CSV_HEADER = [
    "run_id",
    "epoch",
    "seg_mri",
    "domain_loss_mri",
    "domain_loss_us",
    "domain_loss",
    "weak_us",
    "outside_loss",
    "inside_presence_loss",
    "area_loss",
    "total_loss",
    "dom_acc",
    "bbox_in",
    "bbox_iou",
    "bbox_centroid_inside",
    "predicted_area_ratio",
    "mean_inside_prob",
    "max_inside_prob",
    "topk_inside_prob",
    "val_loss",
    "val_dice",
    "val_hd95",
    "val_obj_precision",
    "val_weak_us",
    "val_outside_loss",
    "val_inside_presence_loss",
    "val_area_loss",
    "val_bbox_in",
    "val_bbox_iou",
    "val_bbox_centroid_inside",
    "val_predicted_area_ratio",
    "val_mean_inside_prob",
    "val_max_inside_prob",
    "val_topk_inside_prob",
    "alpha",
    "lambda_mri",
    "lambda_us",
    "lambda_domain",
    "use_dann",
    "weak_mode",
    "lr_encoder",
    "lr_decoder",
    "lr_discriminator",
    "target_score",
    "steps_per_epoch",
    "is_best",
    "timestamp",
]


def init_target_metrics_csv(run_id: str) -> str:
    os.makedirs(CONFIG["logs_path"], exist_ok=True)
    csv_path = os.path.join(CONFIG["logs_path"], "target_training_metrics_pu.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(_CSV_HEADER)
        log.info("Metricas target -> %s (archivo nuevo)", csv_path)
    else:
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            existing_header = next(csv.reader(fh), [])
        if existing_header != _CSV_HEADER:
            stem = Path(csv_path).stem
            csv_path = os.path.join(
                CONFIG["logs_path"],
                f"{stem}_{run_id}.csv",
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(_CSV_HEADER)
            log.warning("Header CSV previo incompatible; metricas nuevas -> %s", csv_path)
        else:
            log.info("Metricas target -> %s (append)", csv_path)
    return csv_path


def append_target_metrics(
    csv_path: str,
    run_id: str,
    epoch: int,
    train_stats: dict[str, float],
    val_metrics: dict,
    val_us_metrics: dict[str, float],
    val_loss: float,
    alpha: float,
    lambda_mri: float,
    lambda_us: float,
    lambda_domain: float,
    use_dann: bool,
    weak_mode: str,
    optimizer: torch.optim.Optimizer,
    target_score: float,
    steps_per_epoch: int,
    is_best: bool,
) -> None:
    hd95_val = val_metrics["HD95"]
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            [
                run_id,
                epoch,
                f"{train_stats['seg_loss_mri']:.6f}",
                f"{train_stats['domain_loss_mri']:.6f}",
                f"{train_stats['domain_loss_us']:.6f}",
                f"{train_stats['domain_loss']:.6f}",
                f"{train_stats['weak_loss_us']:.6f}",
                f"{train_stats['outside_loss']:.6f}",
                f"{train_stats['inside_presence_loss']:.6f}",
                f"{train_stats['area_loss']:.6f}",
                f"{train_stats['total_loss']:.6f}",
                f"{train_stats['domain_acc']:.6f}",
                f"{train_stats['bbox_inside_ratio']:.6f}",
                f"{train_stats['bbox_iou']:.6f}",
                f"{train_stats['bbox_centroid_inside']:.6f}",
                f"{train_stats['predicted_area_ratio']:.6f}",
                f"{train_stats['mean_inside_prob']:.6f}",
                f"{train_stats['max_inside_prob']:.6f}",
                f"{train_stats['topk_inside_prob']:.6f}",
                f"{val_loss:.6f}",
                f"{val_metrics['Dice']:.6f}",
                f"{hd95_val:.4f}" if np.isfinite(hd95_val) else "inf",
                f"{val_metrics['Object_Precision']:.6f}",
                f"{val_us_metrics['weak_loss_us']:.6f}",
                f"{val_us_metrics['outside_loss']:.6f}",
                f"{val_us_metrics['inside_presence_loss']:.6f}",
                f"{val_us_metrics['area_loss']:.6f}",
                f"{val_us_metrics['bbox_inside_ratio']:.6f}",
                f"{val_us_metrics['bbox_iou']:.6f}",
                f"{val_us_metrics['bbox_centroid_inside']:.6f}",
                f"{val_us_metrics['predicted_area_ratio']:.6f}",
                f"{val_us_metrics['mean_inside_prob']:.6f}",
                f"{val_us_metrics['max_inside_prob']:.6f}",
                f"{val_us_metrics['topk_inside_prob']:.6f}",
                f"{alpha:.6f}",
                f"{lambda_mri:.6f}",
                f"{lambda_us:.6f}",
                f"{lambda_domain:.6f}",
                int(use_dann),
                weak_mode,
                f"{optimizer.param_groups[0]['lr']:.2e}",
                f"{optimizer.param_groups[1]['lr']:.2e}",
                f"{optimizer.param_groups[2]['lr']:.2e}",
                f"{target_score:.6f}",
                steps_per_epoch,
                int(is_best),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ]
        )


def load_phase1_checkpoint(model: DANNUNet, device: torch.device) -> None:
    checkpoint_path = CONFIG.get("source_model_path") or CONFIG["model_path"]
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        log.warning(
            "Checkpoint Fase 1 no encontrado en '%s'. Se entrenara desde inicializacion.",
            checkpoint_path,
        )
        return

    missing, unexpected = model.load_pretrained_segmenter(
        checkpoint_path,
        map_location=device,
        strict=False,
    )
    log.info("Checkpoint Fase 1 cargado en DANNUNet.segmenter -> %s", checkpoint_path)
    if missing:
        log.warning("Pesos faltantes al cargar segmenter: %s", missing)
    if unexpected:
        log.warning("Pesos inesperados ignorados: %s", unexpected)


def save_checkpoint(
    path: str,
    model: DANNUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    best_target_score: float,
    val_metrics: dict,
    val_us_metrics: dict[str, float],
    alpha: float,
    lambda_mri: float,
    lambda_us: float,
    lambda_domain: float,
    use_dann: bool,
    weak_mode: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "segmenter_state_dict": model.segmenter.state_dict(),
            "domain_discriminator_state_dict": model.domain_discriminator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "best_target_score": best_target_score,
            "metrics": val_metrics,
            "weak_us_metrics": val_us_metrics,
            "alpha": alpha,
            "lambda_mri": lambda_mri,
            "lambda_us": lambda_us,
            "lambda_domain": lambda_domain,
            "lambda_weak": lambda_us,
            "use_dann": use_dann,
            "weak_mode": weak_mode,
            "config": CONFIG,
        },
        path,
    )


def save_weak_loss_debug(
    us_images: torch.Tensor,
    seg_logits_us: torch.Tensor,
    bbox_masks: torch.Tensor,
    weak_parts: dict[str, torch.Tensor],
    epoch: int,
    step: int,
    output_dir: str,
    threshold: float,
) -> None:
    if not output_dir:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    expanded = weak_parts.get("expanded_bbox_mask")
    outside_penalty = weak_parts.get("outside_penalty_map")
    max_samples = min(int(CONFIG.get("weak_debug_max_samples", 4)), us_images.size(0))
    thresholds = parse_us_metric_thresholds()

    for sample_idx in range(max_samples):
        image = us_images[sample_idx, 0].detach().float().cpu().numpy()
        prob = torch.sigmoid(seg_logits_us[sample_idx, 0].detach().float()).cpu().numpy()
        binary = (prob >= threshold).astype(np.float32)
        bbox = bbox_masks[sample_idx, 0].detach().float().cpu().numpy()
        expanded_np = (
            expanded[sample_idx, 0].detach().float().cpu().numpy()
            if expanded is not None
            else bbox
        )
        outside_np = (
            outside_penalty[sample_idx, 0].detach().float().cpu().numpy()
            if outside_penalty is not None
            else np.zeros_like(prob)
        )

        inside_probs = prob[bbox.astype(bool)]
        topk_mask = np.zeros_like(prob, dtype=np.float32)
        if inside_probs.size > 0:
            inside_fraction = min(max(float(CONFIG.get("weak_inside_fraction", 0.25)), 1e-6), 1.0)
            k = max(1, int(round(float(inside_probs.size) * inside_fraction)))
            bbox_indices = np.argwhere(bbox.astype(bool))
            selected = np.argpartition(inside_probs, -k)[-k:]
            topk_coords = bbox_indices[selected]
            topk_mask[topk_coords[:, 0], topk_coords[:, 1]] = 1.0

        metrics = compute_us_bbox_metrics(
            seg_logits_us[sample_idx : sample_idx + 1].detach(),
            bbox_masks[sample_idx : sample_idx + 1].detach(),
            threshold=threshold,
        )
        threshold_summary = summarize_threshold_area_ratios(
            seg_logits_us[sample_idx : sample_idx + 1].detach(),
            bbox_masks[sample_idx : sample_idx + 1].detach(),
            thresholds,
        )
        strict_area = float(bbox.sum())
        expanded_area = float(expanded_np.sum())
        prob_area_ratio = float(prob.sum() / max(strict_area, 1.0))
        inside_mean = float(inside_probs.mean()) if inside_probs.size else 0.0
        inside_max = float(inside_probs.max()) if inside_probs.size else 0.0
        inside_topk = float(prob[topk_mask.astype(bool)].mean()) if topk_mask.any() else 0.0

        # Strict bbox is the candidate/unlabeled region for PU inside top-k,
        # area prior and bbox metrics. Expanded bbox is only used by outside_loss
        # as a tight no-penalty margin around the coarse annotation.
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor="white", constrained_layout=True)
        axes = axes.ravel()
        panels = [
            ("US + strict bbox", image, "gray"),
            ("US + expanded bbox", image, "gray"),
            ("Strict bbox mask", bbox, "gray"),
            ("Sigmoid probability", prob, "magma"),
            ("Binary prediction", image, "gray"),
            ("Inside / outside regions", expanded_np - bbox, "viridis"),
            ("Inside top-k region", topk_mask, "magma"),
            ("Full overlay", image, "gray"),
        ]
        for ax, (title, base, cmap) in zip(axes, panels):
            ax.imshow(base, cmap=cmap, vmin=0 if cmap != "gray" else None, vmax=1 if cmap != "gray" else None)
            if title in {"US + strict bbox", "Full overlay"}:
                ax.contour(bbox, levels=[0.5], colors=["#00d16f"], linewidths=1.8)
            if title in {"US + expanded bbox", "Full overlay"}:
                ax.contour(expanded_np, levels=[0.5], colors=["#ffd400"], linewidths=1.4)
            if title in {"Binary prediction", "Full overlay"} and binary.any():
                ax.contour(binary, levels=[0.5], colors=["#ff3030"], linewidths=1.5)
            if title == "Inside / outside regions":
                ax.imshow(bbox, cmap="Greens", alpha=0.45, vmin=0, vmax=1)
                ax.imshow(np.clip(expanded_np - bbox, 0, 1), cmap="Wistia", alpha=0.45, vmin=0, vmax=1)
                ax.imshow(outside_np, cmap="Reds", alpha=0.35)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fig.suptitle(
            (
                f"ep={epoch} step={step} sample={sample_idx} thr={threshold:.2f} | "
                f"prob_area={prob_area_ratio:.3f} "
                f"bin_area={metrics['predicted_area_ratio']:.3f} bbox_in={metrics['bbox_inside_ratio']:.3f}\n"
                f"inside mean/max/topk={inside_mean:.3f}/{inside_max:.3f}/{inside_topk:.3f} | "
                f"strict_area={strict_area:.0f}px expanded_area={expanded_area:.0f}px | {threshold_summary}"
            ),
            fontsize=11,
        )
        fig.savefig(
            out / f"weak_ep{epoch:03d}_step{step:04d}_sample{sample_idx:02d}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.close(fig)


def resolve_steps_per_epoch(us_loader: DataLoader) -> int:
    configured = int(CONFIG.get("target_steps_per_epoch", 0) or 0)
    if configured > 0:
        return configured
    return max(1, len(us_loader))


def compute_dann_warmup_schedule(
    epoch: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    lambda_domain_base: float,
    use_dann: bool,
) -> tuple[float, float]:
    """Return GRL alpha and effective domain-loss weight after target warm-up."""
    if not use_dann or epoch <= int(CONFIG.get("dann_warmup_epochs", 10)):
        return 0.0, 0.0

    active_step = max(0, global_step - warmup_steps)
    active_total = max(1, total_steps - warmup_steps)
    alpha = compute_lambda_schedule(active_step, active_total)
    # Both GRL strength and loss weight ramp after warm-up. This keeps MRI
    # anatomical features learned in phase 1 dominant while the discriminator
    # becomes a weak regularizer instead of the main training signal.
    return alpha, lambda_domain_base * alpha


def compute_target_selection_score(
    val_us_metrics: dict[str, float],
    train_stats: dict[str, float],
    val_dice: float,
    has_us_validation: bool,
) -> float:
    """US-oriented checkpoint score with a small MRI guard against anatomical drift."""
    if has_us_validation:
        bbox_inside = float(val_us_metrics.get("bbox_inside_ratio", 0.0))
        bbox_iou = float(val_us_metrics.get("bbox_iou", 0.0))
        bbox_centroid = float(val_us_metrics.get("bbox_centroid_inside", 0.0))
        topk_inside = float(val_us_metrics.get("topk_inside_prob", 0.0))
        max_inside = float(val_us_metrics.get("max_inside_prob", 0.0))
        pred_area = float(val_us_metrics.get("predicted_area_ratio", 0.0))
        weak_loss = float(val_us_metrics.get("weak_loss_us", 0.0))
        min_area = float(CONFIG.get("weak_min_area_ratio", 0.10))
        max_area = float(CONFIG.get("weak_max_area_ratio", 0.60))
        area_penalty = max(0.0, min_area - pred_area) + max(0.0, pred_area - max_area)

        # PU checkpointing must not reward the degenerate "low loss but no US
        # activation" solution. Prefer visible probability mass inside the
        # strict bbox and only use weak_loss as a secondary tie-breaker.
        target_score = float(
            2.0 * bbox_inside
            + 1.5 * topk_inside
            + 0.5 * max_inside
            + 0.25 * bbox_iou
            + 0.25 * bbox_centroid
            - 0.25 * weak_loss
            - area_penalty
        )
    else:
        # Fallback when no US validation split exists: prefer lower target loss
        # rather than selecting checkpoints purely from MRI validation Dice.
        target_score = float(-train_stats["weak_loss_us"] - train_stats["total_loss"])

    min_val_dice = float(CONFIG.get("target_min_val_dice", 0.0))
    if min_val_dice > 0.0 and val_dice < min_val_dice:
        target_score -= min_val_dice - val_dice
    target_score += float(CONFIG.get("target_mri_guard_weight", 0.05)) * val_dice
    return target_score


def run_epoch(
    model: DANNUNet,
    mri_loader: DataLoader,
    us_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_steps: int,
    start_step: int,
    lambda_mri: float,
    lambda_us: float,
    lambda_domain_base: float,
    use_dann: bool,
    weak_mode: str,
    use_amp: bool,
    steps_per_epoch: int,
    warmup_steps: int,
    log_every: int = 10,
) -> tuple[dict[str, float], float]:
    model.train()
    mri_iter = cycle(mri_loader)
    us_iter = cycle(us_loader)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weak_metric_every = max(1, int(CONFIG.get("weak_metric_every", log_every)))

    totals = {
        "seg_loss_mri": 0.0,
        "domain_loss_mri": 0.0,
        "domain_loss_us": 0.0,
        "domain_loss": 0.0,
        "weak_loss_us": 0.0,
        "outside_loss": 0.0,
        "inside_presence_loss": 0.0,
        "area_loss": 0.0,
        "total_loss": 0.0,
        "domain_acc": 0.0,
        "bbox_inside_ratio": 0.0,
        "bbox_iou": 0.0,
        "bbox_centroid_inside": 0.0,
        "predicted_area_ratio": 0.0,
        "weak_predicted_area_ratio": 0.0,
        "mean_inside_prob": 0.0,
        "max_inside_prob": 0.0,
        "topk_inside_prob": 0.0,
    }
    last_alpha = 0.0
    last_lambda_domain = 0.0
    metric_count = 0
    last_us_metrics = {
        "bbox_inside_ratio": 0.0,
        "bbox_iou": 0.0,
        "bbox_centroid_inside": 0.0,
        "predicted_area_ratio": 0.0,
    }

    # The epoch is target-governed: by default steps_per_epoch == len(us_loader).
    # cycle(us_loader) is kept only as a safe iterator wrapper for optional
    # overrides; with the default length we consume one shuffled US pass and do
    # not replay cached US batches 11x inside the same epoch.
    for step in range(steps_per_epoch):
        mri_images, mri_masks = next(mri_iter)
        global_step = start_step + step
        alpha, lambda_domain = compute_dann_warmup_schedule(
            epoch,
            global_step,
            total_steps,
            warmup_steps,
            lambda_domain_base,
            use_dann,
        )
        last_alpha = alpha
        last_lambda_domain = lambda_domain

        mri_images = mri_images.to(device, non_blocking=True)
        mri_masks = mri_masks.to(device, non_blocking=True)
        us_images, us_weak_masks, us_bbox_masks = next(us_iter)
        us_images = us_images.to(device, non_blocking=True)
        us_weak_masks = us_weak_masks.to(device, non_blocking=True)
        us_bbox_masks = us_bbox_masks.to(device, non_blocking=True)
        if weak_mode == "bbox":
            validate_weak_bbox_spatial_consistency(us_weak_masks, us_bbox_masks, epoch, step)

        mri_labels = torch.full(
            (mri_images.size(0),), MRI_DOMAIN, dtype=torch.long, device=device
        )
        us_labels = torch.full(
            (us_images.size(0),), US_DOMAIN, dtype=torch.long, device=device
        )

        optimizer.zero_grad(set_to_none=True)

        all_images = torch.cat([mri_images, us_images], dim=0)
        all_domain_labels = torch.cat([mri_labels, us_labels], dim=0)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if lambda_domain > 0.0:
                seg_logits_all, domain_logits_all = model(
                    all_images,
                    alpha=alpha,
                    return_segmentation=True,
                )
                assert seg_logits_all is not None
            else:
                # During DANN warm-up the discriminator must not shape the
                # encoder. Running only the segmenter preserves the MRI
                # pretraining signal, keeps US weak supervision active, and
                # avoids computing domain logits that have zero loss weight.
                seg_logits_all = model.segmenter(all_images)
                domain_logits_all = None

            mri_batch_size = mri_images.size(0)
            seg_logits = seg_logits_all[:mri_batch_size]
            seg_logits_us = seg_logits_all[mri_batch_size:]

            seg_loss = bce_dice_loss(seg_logits, mri_masks)
            weak_loss_us, weak_parts = compute_us_weak_loss(seg_logits_us, us_weak_masks, weak_mode)
            if domain_logits_all is not None:
                domain_logits_mri = domain_logits_all[:mri_batch_size]
                domain_logits_us = domain_logits_all[mri_batch_size:]
                domain_loss_mri = F.cross_entropy(domain_logits_mri, mri_labels)
                domain_loss_us = F.cross_entropy(domain_logits_us, us_labels)
                domain_loss = F.cross_entropy(domain_logits_all, all_domain_labels)
                weighted_domain = lambda_domain * domain_loss
            else:
                domain_loss_mri = seg_loss.new_tensor(0.0)
                domain_loss_us = seg_loss.new_tensor(0.0)
                domain_loss = seg_loss.new_tensor(0.0)
                weighted_domain = seg_loss.new_tensor(0.0)
            total_loss = lambda_mri * seg_loss + lambda_us * weak_loss_us + weighted_domain

            debug_every = int(CONFIG.get("weak_debug_every", 0))
            debug_dir = str(CONFIG.get("weak_debug_dir", ""))
            debug_max_batches = int(CONFIG.get("weak_debug_max_batches", 2))
            if (
                debug_every > 0
                and debug_dir
                and step < debug_max_batches
                and epoch % debug_every == 0
                and weak_mode == "bbox"
            ):
                save_weak_loss_debug(
                    us_images=us_images,
                    seg_logits_us=seg_logits_us,
                    bbox_masks=us_bbox_masks,
                    weak_parts=weak_parts,
                    epoch=epoch,
                    step=step,
                    output_dir=debug_dir,
                    threshold=CONFIG["threshold"],
                )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if domain_logits_all is not None:
            acc = 0.5 * (
                domain_accuracy(domain_logits_mri, mri_labels)
                + domain_accuracy(domain_logits_us, us_labels)
            )
        else:
            acc = 0.5
        should_compute_metrics = (step % weak_metric_every == 0) or (step == steps_per_epoch - 1)
        if should_compute_metrics:
            last_us_metrics = compute_us_bbox_metrics(
                seg_logits_us,
                us_bbox_masks,
                threshold=CONFIG["threshold"],
            )
            totals["bbox_inside_ratio"] += last_us_metrics["bbox_inside_ratio"]
            totals["bbox_iou"] += last_us_metrics["bbox_iou"]
            totals["bbox_centroid_inside"] += last_us_metrics["bbox_centroid_inside"]
            totals["predicted_area_ratio"] += last_us_metrics["predicted_area_ratio"]
            metric_count += 1

        totals["seg_loss_mri"] += float(seg_loss.item())
        totals["domain_loss_mri"] += float(domain_loss_mri.item())
        totals["domain_loss_us"] += float(domain_loss_us.item())
        totals["domain_loss"] += float(domain_loss.item())
        totals["weak_loss_us"] += float(weak_loss_us.item())
        totals["outside_loss"] += float(weak_parts["outside_loss"].item())
        totals["inside_presence_loss"] += float(weak_parts["inside_presence_loss"].item())
        totals["area_loss"] += float(weak_parts["area_loss"].item())
        totals["weak_predicted_area_ratio"] += float(weak_parts["predicted_area_ratio"].item())
        totals["mean_inside_prob"] += float(weak_parts["mean_inside_prob"].item())
        totals["max_inside_prob"] += float(weak_parts["max_inside_prob"].item())
        totals["topk_inside_prob"] += float(weak_parts["topk_inside_prob"].item())
        totals["total_loss"] += float(total_loss.item())
        totals["domain_acc"] += acc

        if step % log_every == 0:
            log.info(
                (
                    "Ep %02d | Step %4d/%d | alpha %.3f | seg_mri %.4f | "
                    "weak_us %.4f (out %.4f | in %.4f | area %.4f) | "
                    "domain_loss %.4f | total %.4f | dom_acc %.3f | bbox_in %.3f | "
                    "pred_area_ratio %.3f | min_area %.3f | max_area %.3f | "
                    "inside mean/max/topk %.3f/%.3f/%.3f | bin_area_ratio %.3f | %s"
                ),
                epoch,
                step,
                steps_per_epoch,
                alpha,
                seg_loss.item(),
                weak_loss_us.item(),
                weak_parts["outside_loss"].item(),
                weak_parts["inside_presence_loss"].item(),
                weak_parts["area_loss"].item(),
                domain_loss.item(),
                total_loss.item(),
                acc,
                last_us_metrics["bbox_inside_ratio"],
                weak_parts["predicted_area_ratio"].item(),
                weak_parts["min_area_ratio"].item(),
                weak_parts["max_area_ratio"].item(),
                weak_parts["mean_inside_prob"].item(),
                weak_parts["max_inside_prob"].item(),
                weak_parts["topk_inside_prob"].item(),
                last_us_metrics["predicted_area_ratio"],
                summarize_threshold_area_ratios(
                    seg_logits_us.detach(),
                    us_bbox_masks.detach(),
                    parse_us_metric_thresholds(),
                ),
            )

    n_steps = max(1, steps_per_epoch)
    stats = {key: value / n_steps for key, value in totals.items()}
    stats["lambda_domain_effective"] = last_lambda_domain
    metric_denominator = max(1, metric_count)
    for key in ("bbox_inside_ratio", "bbox_iou", "bbox_centroid_inside", "predicted_area_ratio"):
        stats[key] = totals[key] / metric_denominator
    return stats, last_alpha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena target MRI->US con weak supervision PU-inspired por bbox."
    )
    parser.add_argument(
        "--lambda_mri",
        type=float,
        default=float(CONFIG.get("lambda_mri", 1.0)),
        help="Peso de Dice+BCE supervisada MRI.",
    )
    parser.add_argument(
        "--lambda_us",
        type=float,
        default=float(CONFIG.get("lambda_us", CONFIG.get("lambda_weak", 0.5))),
        help="Peso de supervision US weak/pseudomask.",
    )
    parser.add_argument(
        "--lambda_domain",
        type=float,
        default=float(CONFIG.get("lambda_domain_base", CONFIG.get("lambda_domain", 0.001))),
        help="Peso base bajo para regularizacion adversarial DANN tras warm-up.",
    )
    parser.add_argument(
        "--dann_warmup_epochs",
        type=int,
        default=int(CONFIG.get("dann_warmup_epochs", 10)),
        help="Epocas iniciales con lambda_domain=0 para proteger el pretraining MRI.",
    )
    parser.add_argument(
        "--use_dann",
        action=argparse.BooleanOptionalAction,
        default=bool(CONFIG.get("use_dann", True)),
        help="Activa/desactiva el termino DANN sin cambiar la arquitectura.",
    )
    parser.add_argument(
        "--weak_mode",
        choices=WEAK_MODES,
        default=str(CONFIG.get("weak_mode", "bbox")),
        help="Tipo de supervision US: bbox o pseudomask.",
    )
    parser.add_argument(
        "--weak_loss_type",
        choices=WEAK_LOSS_TYPES,
        default=str(CONFIG.get("weak_loss_type", "soft_bbox")),
        help="Loss para weak_mode=bbox: soft_bbox nueva o legacy_bbox previa.",
    )
    parser.add_argument("--bbox_margin_px", type=int, default=int(CONFIG.get("weak_bbox_margin_px", 8)))
    parser.add_argument(
        "--bbox_expand_ratio",
        type=float,
        default=float(CONFIG.get("weak_bbox_expand_ratio", 0.10)),
        help="Expansion relativa de bbox usada solo por outside_loss. Probar 0.05 si 0.10 sigue permisivo.",
    )
    parser.add_argument("--outside_weight", type=float, default=float(CONFIG.get("weak_outside_weight", 1.0)))
    parser.add_argument("--inside_weight", type=float, default=float(CONFIG.get("weak_inside_weight", 0.5)))
    parser.add_argument("--area_weight", type=float, default=float(CONFIG.get("weak_area_weight", 0.05)))
    parser.add_argument(
        "--min_inside_activation",
        type=float,
        default=float(CONFIG.get("weak_min_inside_activation", 0.20)),
    )
    parser.add_argument(
        "--min_area_ratio",
        type=float,
        default=float(CONFIG.get("weak_min_area_ratio", 0.10)),
        help="Area minima predicha como fraccion del area bbox usando probabilidad sigmoid.",
    )
    parser.add_argument(
        "--under_area_weight",
        type=float,
        default=float(CONFIG.get("weak_under_area_weight", 2.0)),
        help="Multiplicador del castigo por area predicha insuficiente.",
    )
    parser.add_argument(
        "--over_area_weight",
        type=float,
        default=float(CONFIG.get("weak_over_area_weight", 1.0)),
        help="Multiplicador del castigo por area predicha excesiva.",
    )
    parser.add_argument(
        "--inside_fraction",
        type=float,
        default=float(CONFIG.get("weak_inside_fraction", 0.25)),
        help="Fraccion top-k dentro de bbox usada para presencia positiva.",
    )
    parser.add_argument(
        "--max_area_ratio",
        type=float,
        default=float(CONFIG.get("weak_max_area_ratio", 1.5)),
    )
    parser.add_argument(
        "--weak_debug_dir",
        type=str,
        default=str(CONFIG.get("weak_debug_dir", "")),
        help="Directorio opcional para guardar debug visual de la weak loss.",
    )
    parser.add_argument(
        "--weak_debug_every",
        type=int,
        default=int(CONFIG.get("weak_debug_every", 0)),
        help="Guardar debug visual cada N epocas. 0 desactiva.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    lambda_mri = args.lambda_mri
    lambda_us = args.lambda_us
    lambda_domain_base = args.lambda_domain
    # With lambda_domain_base=0.0 this experiment is weak-supervision-only. Keep
    # the DANN module available in the model/checkpoints, but avoid logging or
    # scheduling an adversarial branch that has no effective weight.
    use_dann = bool(args.use_dann) and lambda_domain_base > 0.0
    weak_mode = args.weak_mode
    CONFIG["dann_warmup_epochs"] = args.dann_warmup_epochs
    CONFIG["lambda_domain_base"] = lambda_domain_base
    CONFIG["weak_loss_type"] = args.weak_loss_type
    CONFIG["weak_bbox_margin_px"] = args.bbox_margin_px
    CONFIG["weak_bbox_expand_ratio"] = args.bbox_expand_ratio
    CONFIG["weak_outside_weight"] = args.outside_weight
    CONFIG["weak_inside_weight"] = args.inside_weight
    CONFIG["weak_area_weight"] = args.area_weight
    CONFIG["weak_min_inside_activation"] = args.min_inside_activation
    CONFIG["weak_min_area_ratio"] = args.min_area_ratio
    CONFIG["weak_under_area_weight"] = args.under_area_weight
    CONFIG["weak_over_area_weight"] = args.over_area_weight
    CONFIG["weak_inside_fraction"] = args.inside_fraction
    CONFIG["weak_max_area_ratio"] = args.max_area_ratio
    CONFIG["weak_debug_dir"] = args.weak_debug_dir
    CONFIG["weak_debug_every"] = args.weak_debug_every
    freeze_decoder_epochs = CONFIG.get("freeze_decoder_epochs", 0)
    log_every = CONFIG.get("log_every", 10)
    use_amp = bool(CONFIG.get("use_amp", device.type == "cuda")) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(CONFIG.get("cudnn_benchmark", True))
        try:
            torch.set_float32_matmul_precision(CONFIG.get("matmul_precision", "high"))
        except Exception:
            pass

    log.info("=" * 60)
    log.info("  MiomaVision - Entrenamiento PU-inspired MRI -> US")
    log.info("  Run ID:      %s", run_id)
    log.info("  Dispositivo: %s", device)
    log.info("  Epocas:      %d", CONFIG["epochs"])
    log.info("  Lambda MRI:  %.4f", lambda_mri)
    log.info("  Lambda US:   %.4f", lambda_us)
    log.info("  Lambda dom base: %.6f", lambda_domain_base if use_dann else 0.0)
    log.info("  DANN warm-up: %d epoca(s)", CONFIG["dann_warmup_epochs"])
    log.info("  DANN:        %s", "ON (regularizador suave)" if use_dann else "OFF")
    log.info("  Weak mode:   %s", weak_mode)
    log.info("  Weak loss:   %s", CONFIG["weak_loss_type"])
    log.info(
        "  Soft bbox:   expand_ratio=%.3f | outside=%.3f | inside=%.3f | area=%.3f | min_inside=%.3f | min_area=%.3f | max_area=%.3f | under=%.3f | over=%.3f | topk=%.3f",
        CONFIG["weak_bbox_expand_ratio"],
        CONFIG["weak_outside_weight"],
        CONFIG["weak_inside_weight"],
        CONFIG["weak_area_weight"],
        CONFIG["weak_min_inside_activation"],
        CONFIG["weak_min_area_ratio"],
        CONFIG["weak_max_area_ratio"],
        CONFIG["weak_under_area_weight"],
        CONFIG["weak_over_area_weight"],
        CONFIG["weak_inside_fraction"],
    )
    log.info("  AMP:         %s", "ON" if use_amp else "OFF")
    log.info("  Log every:   %d step(s)", log_every)
    log.info("  US thresholds inspeccion: %s", parse_us_metric_thresholds())
    log.info("=" * 60)

    mri_base_path = CONFIG["base_path"]
    us_base_path = get_us_base_path()

    tr_imgs, tr_masks = load_split_paths(mri_base_path, "train")
    val_imgs, val_masks = load_split_paths(mri_base_path, "val")
    us_train_imgs = load_us_image_paths(us_base_path, "train", weak_mode=weak_mode)
    try:
        us_val_imgs = load_us_image_paths(us_base_path, "val", weak_mode=weak_mode)
    except FileNotFoundError as exc:
        log.warning("Validacion weak US desactivada: %s", exc)
        us_val_imgs = []

    log.info("- Diagnostico MRI -")
    log_split_diagnostics(tr_imgs, "train")
    log_split_diagnostics(val_imgs, "val")
    log.info(
        "US train: %d imagenes | US val: %d imagenes | base: %s",
        len(us_train_imgs),
        len(us_val_imgs),
        us_base_path,
    )
    log.info(
        "Dataloaders simultaneos: US gobierna la epoca; MRI usa cycle() para aportar "
        "un batch supervisado por step sin reciclar US repetidamente."
    )

    num_workers = CONFIG.get("num_workers", 0)
    pin_mem = device.type == "cuda"

    mri_train_loader = DataLoader(
        SagitalDataset(tr_imgs, tr_masks),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        # drop_last keeps MRI/US batch sizes equal for torch.cat. Because MRI is
        # cycled and US governs the epoch, only a small shuffled MRI subset is
        # consumed per target epoch instead of oversampling the small US split.
        drop_last=True,
    )
    mri_val_loader = DataLoader(
        SagitalDataset(val_imgs, val_masks),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    us_train_loader = DataLoader(
        UltrasoundDataset(
            us_train_imgs,
            image_size=CONFIG.get("image_size", 256),
            weak_mode=weak_mode,
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        # With steps_per_epoch=len(us_loader), shuffle=True gives one fresh US
        # order per epoch and drop_last avoids a final smaller target batch.
        # cycle(us_loader) is therefore not replaying cached US batches unless
        # target_steps_per_epoch is explicitly set above len(us_loader).
        drop_last=True,
    )
    us_val_loader = (
        DataLoader(
            UltrasoundDataset(
                us_val_imgs,
                image_size=CONFIG.get("image_size", 256),
                weak_mode=weak_mode,
            ),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )
        if us_val_imgs
        else None
    )
    steps_per_epoch = resolve_steps_per_epoch(us_train_loader)
    if len(us_train_loader) == 0:
        raise ValueError(
            "US train loader quedo vacio. Reduce BATCH_SIZE o desactiva drop_last "
            "si el split target tiene menos imagenes que un batch."
        )
    log.info(
        "Steps/epoca target: %d (len US loader=%d | len MRI loader=%d). "
        "Con drop_last=True, la epoca default consume una pasada US y evita repetir "
        "los mismos batches target dentro de la epoca.",
        steps_per_epoch,
        len(us_train_loader),
        len(mri_train_loader),
    )

    model = DANNUNet(
        in_channels=CONFIG["in_channels"],
        num_classes=CONFIG["num_classes"],
        base_filters=CONFIG.get("base_filters", 64),
        discriminator_hidden_dim=CONFIG.get("domain_hidden_dim", 512),
        discriminator_dropout=CONFIG.get("domain_dropout", 0.5),
        feature_mode=CONFIG.get("domain_feature_mode", "bottleneck"),
    ).to(device)

    load_phase1_checkpoint(model, device)

    if freeze_decoder_epochs > 0:
        set_decoder_trainable(model, False)
        log.info("Decoder congelado por %d epoca(s).", freeze_decoder_epochs)

    optimizer = build_optimizer(model)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CONFIG.get("lr_factor", 0.5),
        patience=CONFIG.get("lr_patience", 5),
        min_lr=CONFIG.get("lr_min", 1e-6),
    )
    csv_path = init_target_metrics_csv(run_id)

    ckpt_dir = Path(CONFIG["logs_path"]) / "checkpoints_target_pu"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(ckpt_dir / "best_model_target_pu.pth")
    last_path = str(ckpt_dir / "last_model_target_pu.pth")

    best_dice = 0.0
    best_weak_score = -np.inf
    best_target_score = -np.inf
    epochs_without_improvement = 0
    early_patience = int(CONFIG.get("target_early_stopping_patience", 10))
    min_delta = float(CONFIG.get("target_early_stopping_min_delta", 1e-4))
    total_steps = CONFIG["epochs"] * max(1, steps_per_epoch)
    warmup_steps = min(total_steps, int(CONFIG.get("dann_warmup_epochs", 10)) * max(1, steps_per_epoch))
    global_step = 0

    initial_val_metrics = {"Dice": 0.0, "HD95": np.inf, "Object_Precision": 0.0}
    initial_us_metrics = {
        "weak_loss_us": 0.0,
        "outside_loss": 0.0,
        "inside_presence_loss": 0.0,
        "area_loss": 0.0,
        "bbox_inside_ratio": 0.0,
        "bbox_iou": 0.0,
        "bbox_centroid_inside": 0.0,
        "predicted_area_ratio": 0.0,
        "mean_inside_prob": 0.0,
        "max_inside_prob": 0.0,
        "topk_inside_prob": 0.0,
    }
    save_checkpoint(
        last_path,
        model,
        optimizer,
        epoch=0,
        best_dice=best_dice,
        best_target_score=best_target_score,
        val_metrics=initial_val_metrics,
        val_us_metrics=initial_us_metrics,
        alpha=0.0,
        lambda_mri=lambda_mri,
        lambda_us=lambda_us,
        lambda_domain=0.0,
        use_dann=use_dann,
        weak_mode=weak_mode,
    )
    log.info("Checkpoint inicial PU guardado: %s", last_path)

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        if freeze_decoder_epochs > 0 and epoch == freeze_decoder_epochs + 1:
            set_decoder_trainable(model, True)
            log.info("Decoder descongelado desde la epoca %d.", epoch)

        train_stats, alpha = run_epoch(
            model=model,
            mri_loader=mri_train_loader,
            us_loader=us_train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_steps=total_steps,
            start_step=global_step,
            lambda_mri=lambda_mri,
            lambda_us=lambda_us,
            lambda_domain_base=lambda_domain_base,
            use_dann=use_dann,
            weak_mode=weak_mode,
            use_amp=use_amp,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=warmup_steps,
            log_every=log_every,
        )
        global_step += steps_per_epoch

        val_metrics, inf_batches, val_loss = validate_mri(model, mri_val_loader, device)
        val_us_metrics = validate_us_weak(model, us_val_loader, device, weak_mode)
        if inf_batches > 0:
            log.warning("HD95=inf en %d batch(es) de validacion.", inf_batches)

        dice_val = val_metrics["Dice"]
        weak_score = 0.5 * (
            val_us_metrics["bbox_inside_ratio"] + val_us_metrics["bbox_centroid_inside"]
        )
        target_score = compute_target_selection_score(
            val_us_metrics,
            train_stats,
            dice_val,
            has_us_validation=us_val_loader is not None,
        )
        is_best = target_score > best_target_score + min_delta
        scheduler.step(target_score)

        if is_best:
            best_target_score = target_score
            best_dice = max(best_dice, dice_val)
            best_weak_score = weak_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_dice,
            best_target_score,
            val_metrics,
            val_us_metrics,
            alpha,
            lambda_mri,
            lambda_us,
            train_stats["lambda_domain_effective"],
            use_dann,
            weak_mode,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_dice,
                best_target_score,
                val_metrics,
                val_us_metrics,
                alpha,
                lambda_mri,
                lambda_us,
                train_stats["lambda_domain_effective"],
                use_dann,
                weak_mode,
            )

        append_target_metrics(
            csv_path,
            run_id,
            epoch,
            train_stats,
            val_metrics,
            val_us_metrics,
            val_loss,
            alpha,
            lambda_mri,
            lambda_us,
            train_stats["lambda_domain_effective"],
            use_dann,
            weak_mode,
            optimizer,
            target_score,
            steps_per_epoch,
            is_best,
        )

        hd95_val = val_metrics["HD95"]
        hd95_str = f"{hd95_val:.2f}px" if np.isfinite(hd95_val) else "inf"
        log.info(
            (
                "Epoca %02d/%02d | total %.4f | seg_mri %.4f | weak_us %.4f "
                "(out %.4f | in %.4f | area %.4f) | domain_loss %.4f | "
                "val_loss %.4f | Dice %.4f | HD95 %s | dom_acc %.3f | "
                "bbox_in %.3f | val_bbox_in %.3f | pred_area_ratio %.3f | "
                "min_area %.3f | max_area %.3f | bin_area_ratio %.3f | "
                "val_area_ratio %.3f | inside mean/max/topk %.3f/%.3f/%.3f | "
                "val_inside mean/max/topk %.3f/%.3f/%.3f | target_score %.4f | "
                "lambda_dom %.6f | %.1fs%s"
            ),
            epoch,
            CONFIG["epochs"],
            train_stats["total_loss"],
            train_stats["seg_loss_mri"],
            train_stats["weak_loss_us"],
            train_stats["outside_loss"],
            train_stats["inside_presence_loss"],
            train_stats["area_loss"],
            train_stats["domain_loss"],
            val_loss,
            dice_val,
            hd95_str,
            train_stats["domain_acc"],
            train_stats["bbox_inside_ratio"],
            val_us_metrics["bbox_inside_ratio"],
            train_stats["weak_predicted_area_ratio"],
            CONFIG["weak_min_area_ratio"],
            CONFIG["weak_max_area_ratio"],
            train_stats["predicted_area_ratio"],
            val_us_metrics["predicted_area_ratio"],
            train_stats["mean_inside_prob"],
            train_stats["max_inside_prob"],
            train_stats["topk_inside_prob"],
            val_us_metrics["mean_inside_prob"],
            val_us_metrics["max_inside_prob"],
            val_us_metrics["topk_inside_prob"],
            target_score,
            train_stats["lambda_domain_effective"],
            time.time() - t0,
            " * BEST" if is_best else "",
        )

        if epoch <= CONFIG["dann_warmup_epochs"] + 3 and train_stats["domain_acc"] > 0.90:
            log.warning(
                "Discriminador separa dominios muy temprano (acc=%.3f). "
                "Si esto persiste, DANN puede volver a dominar aunque lambda_domain sea suave.",
                train_stats["domain_acc"],
            )

        if early_patience > 0 and epochs_without_improvement >= early_patience:
            log.info(
                "Early stopping target: %d epoca(s) sin mejora en score US/adaptacion.",
                epochs_without_improvement,
            )
            break

    log.info("=" * 60)
    log.info("Entrenamiento DANN finalizado")
    log.info("Mejor Dice: %.4f", best_dice)
    log.info("Mejor weak score asociado: %.4f", best_weak_score)
    log.info("Mejor target score: %.4f", best_target_score)
    log.info("Best checkpoint: %s", best_path)
    log.info("Last checkpoint: %s", last_path)
    log.info("Metricas: %s", csv_path)
    log.info("=" * 60)


if __name__ == "__main__":
    train()
