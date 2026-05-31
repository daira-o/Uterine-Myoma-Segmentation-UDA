"""
scripts/training/train_target.py
Entrenamiento target MRI -> US para la Attention U-Net.

Prioriza segmentacion supervisada MRI y supervision debil US. DANN queda
disponible como regularizador opcional, apagado por defecto:
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
import re

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
            xmin = int(np.floor(float(row["xmin"])))
            ymin = int(np.floor(float(row["ymin"])))
            xmax = int(np.ceil(float(row["xmax"])))
            ymax = int(np.ceil(float(row["ymax"])))
            xmin = min(max(xmin, 0), self.image_size)
            xmax = min(max(xmax, 0), self.image_size)
            ymin = min(max(ymin, 0), self.image_size)
            ymax = min(max(ymax, 0), self.image_size)
            if xmax > xmin and ymax > ymin:
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
    return CONFIG.get("us_ready_path") or os.path.join(str(ROOT), "data_ready_US")


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


def expand_bbox_mask(bbox_mask: torch.Tensor, margin_px: int) -> torch.Tensor:
    """Expande una bbox-mask binaria sin usarla como clipping duro."""
    bbox_mask = (bbox_mask.float() > 0.5).float()
    if margin_px <= 0:
        return bbox_mask
    kernel_size = int(2 * margin_px + 1)
    return F.max_pool2d(
        bbox_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=margin_px,
    )


def soft_bbox_loss(
    pred_logits: torch.Tensor,
    bbox_mask: torch.Tensor,
    bbox_margin_px: int = 8,
    outside_weight: float = 1.0,
    inside_weight: float = 0.5,
    area_weight: float = 0.05,
    min_inside_activation: float = 0.20,
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 1.5,
    under_area_weight: float = 2.0,
    over_area_weight: float = 1.0,
    inside_topk_fraction: float = 0.10,
    min_inside_mean: float = 0.08,
    inside_mean_weight: float = 0.5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Supervision debil suave por bbox unica.

    - bbox original: incentiva presencia positiva, sin exigir llenar la caja.
    - bbox expandida: zona permitida.
    - fuera de bbox expandida: penalizacion principal contra activaciones.
    """
    probs = torch.sigmoid(pred_logits.float())
    bbox_mask = (bbox_mask.float() > 0.5).float()
    expanded_bbox = expand_bbox_mask(bbox_mask, bbox_margin_px)
    outside_mask = 1.0 - expanded_bbox

    outside_targets = torch.zeros_like(pred_logits.float())
    outside_loss_map = F.binary_cross_entropy_with_logits(
        pred_logits.float(),
        outside_targets,
        reduction="none",
    )
    outside_area = outside_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    outside_loss = (outside_loss_map * outside_mask).sum(dim=(1, 2, 3)) / outside_area

    inside_losses = []
    inside_mean_losses = []
    mean_inside_probs = []
    topk_inside_probs = []
    for sample_probs, sample_bbox in zip(probs, bbox_mask):
        inside_probs = sample_probs[sample_bbox.bool()]
        if inside_probs.numel() == 0:
            inside_losses.append(sample_probs.new_tensor(0.0))
            inside_mean_losses.append(sample_probs.new_tensor(0.0))
            mean_inside_probs.append(sample_probs.new_tensor(0.0))
            topk_inside_probs.append(sample_probs.new_tensor(0.0))
            continue
        k = max(1, int(round(float(inside_probs.numel()) * inside_topk_fraction)))
        topk_mean = inside_probs.topk(k).values.mean()
        mean_inside = inside_probs.mean()
        topk_inside_probs.append(topk_mean)
        mean_inside_probs.append(mean_inside)
        inside_losses.append(F.relu(min_inside_activation - topk_mean).pow(2))
        inside_mean_losses.append(F.relu(min_inside_mean - mean_inside))
    topk_presence_loss = torch.stack(inside_losses)
    inside_mean_loss = torch.stack(inside_mean_losses)
    inside_presence_loss = topk_presence_loss + inside_mean_weight * inside_mean_loss
    mean_prob_inside_bbox = torch.stack(mean_inside_probs)
    topk_inside_prob = torch.stack(topk_inside_probs)

    # Una mascara vacia tiene area 0. Si solo penalizamos exceso de area,
    # la solucion vacia puede volverse optima cuando outside_loss domina.
    # Esta banda penaliza mascaras demasiado chicas y demasiado grandes.
    pred_area = probs.sum(dim=(1, 2, 3))
    bbox_area = bbox_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    predicted_area_ratio = pred_area / bbox_area
    under_area = F.relu(min_area_ratio - predicted_area_ratio)
    over_area = F.relu(predicted_area_ratio - max_area_ratio)
    area_loss = under_area_weight * under_area + over_area_weight * over_area

    total = (
        outside_weight * outside_loss
        + inside_weight * inside_presence_loss
        + area_weight * area_loss
    )
    components = {
        "outside_loss": outside_loss.mean(),
        "inside_presence_loss": inside_presence_loss.mean(),
        "inside_mean_loss": inside_mean_loss.mean(),
        "mean_prob_inside_bbox": mean_prob_inside_bbox.mean(),
        "topk_inside_prob": topk_inside_prob.mean(),
        "area_loss": area_loss.mean(),
        "under_area_loss": under_area.mean(),
        "over_area_loss": over_area.mean(),
        "weak_us_total": total.mean(),
        "predicted_area_ratio": predicted_area_ratio.mean(),
        "inside_activation": topk_inside_prob.mean(),
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
                outside_weight=float(CONFIG.get("weak_outside_weight", 1.0)),
                inside_weight=float(CONFIG.get("weak_inside_weight", 0.5)),
                area_weight=float(CONFIG.get("weak_area_weight", 0.05)),
                min_inside_activation=float(CONFIG.get("weak_min_inside_activation", 0.20)),
                min_area_ratio=float(CONFIG.get("weak_min_area_ratio", 0.15)),
                max_area_ratio=float(CONFIG.get("weak_max_area_ratio", 1.5)),
                under_area_weight=float(CONFIG.get("weak_under_area_weight", 2.0)),
                over_area_weight=float(CONFIG.get("weak_over_area_weight", 1.0)),
                inside_topk_fraction=float(CONFIG.get("weak_inside_fraction", 0.10)),
                min_inside_mean=float(CONFIG.get("weak_min_inside_mean", 0.08)),
                inside_mean_weight=float(CONFIG.get("weak_inside_mean_weight", 0.5)),
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
                "inside_mean_loss": zero,
                "mean_prob_inside_bbox": zero,
                "topk_inside_prob": zero,
                "area_loss": zero,
                "under_area_loss": zero,
                "over_area_loss": zero,
                "weak_us_total": loss.detach(),
                "predicted_area_ratio": zero,
            }
        raise ValueError(f"weak_loss_type no soportado: {weak_loss_type}")
    if weak_mode == "pseudomask":
        loss = bce_dice_loss(seg_logits, weak_mask.float())
        zero = loss.detach().new_tensor(0.0)
        return loss, {
            "outside_loss": zero,
            "inside_presence_loss": zero,
            "inside_mean_loss": zero,
            "mean_prob_inside_bbox": zero,
            "topk_inside_prob": zero,
            "area_loss": zero,
            "under_area_loss": zero,
            "over_area_loss": zero,
            "weak_us_total": loss.detach(),
            "predicted_area_ratio": zero,
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
            "inside_mean_loss": 0.0,
            "mean_prob_inside_bbox": 0.0,
            "topk_inside_prob": 0.0,
            "area_loss": 0.0,
            "under_area_loss": 0.0,
            "over_area_loss": 0.0,
            "bbox_inside_ratio": 0.0,
            "bbox_iou": 0.0,
            "bbox_centroid_inside": 0.0,
            "predicted_area_ratio": 0.0,
            "min_area_ratio": float(CONFIG.get("weak_min_area_ratio", 0.15)),
            "max_area_ratio": float(CONFIG.get("weak_max_area_ratio", 1.0)),
        }

    model.eval()
    losses = []
    metrics = {
        "outside_loss": [],
        "inside_presence_loss": [],
        "inside_mean_loss": [],
        "mean_prob_inside_bbox": [],
        "topk_inside_prob": [],
        "area_loss": [],
        "under_area_loss": [],
        "over_area_loss": [],
        "bbox_inside_ratio": [],
        "bbox_iou": [],
        "bbox_centroid_inside": [],
        "predicted_area_ratio": [],
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
                "inside_mean_loss",
                "mean_prob_inside_bbox",
                "topk_inside_prob",
                "area_loss",
                "under_area_loss",
                "over_area_loss",
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
        "inside_mean_loss": float(np.mean(metrics["inside_mean_loss"]))
        if metrics["inside_mean_loss"]
        else 0.0,
        "mean_prob_inside_bbox": float(np.mean(metrics["mean_prob_inside_bbox"]))
        if metrics["mean_prob_inside_bbox"]
        else 0.0,
        "topk_inside_prob": float(np.mean(metrics["topk_inside_prob"]))
        if metrics["topk_inside_prob"]
        else 0.0,
        "area_loss": float(np.mean(metrics["area_loss"])) if metrics["area_loss"] else 0.0,
        "under_area_loss": float(np.mean(metrics["under_area_loss"]))
        if metrics["under_area_loss"]
        else 0.0,
        "over_area_loss": float(np.mean(metrics["over_area_loss"]))
        if metrics["over_area_loss"]
        else 0.0,
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
        "min_area_ratio": float(CONFIG.get("weak_min_area_ratio", 0.15)),
        "max_area_ratio": float(CONFIG.get("weak_max_area_ratio", 1.0)),
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
    "inside_mean_loss",
    "mean_prob_inside_bbox",
    "topk_inside_prob",
    "area_loss",
    "under_area_loss",
    "over_area_loss",
    "total_loss",
    "dom_acc",
    "bbox_in",
    "bbox_iou",
    "bbox_centroid_inside",
    "predicted_area_ratio",
    "min_area_ratio",
    "max_area_ratio",
    "val_loss",
    "val_dice",
    "val_hd95",
    "val_obj_precision",
    "val_weak_us",
    "val_outside_loss",
    "val_inside_presence_loss",
    "val_inside_mean_loss",
    "val_mean_prob_inside_bbox",
    "val_topk_inside_prob",
    "val_area_loss",
    "val_under_area_loss",
    "val_over_area_loss",
    "val_bbox_in",
    "val_bbox_iou",
    "val_bbox_centroid_inside",
    "val_predicted_area_ratio",
    "val_min_area_ratio",
    "val_max_area_ratio",
    "val_area_score",
    "target_score",
    "alpha",
    "lambda_mri",
    "lambda_us",
    "lambda_domain",
    "use_dann",
    "weak_mode",
    "lr_encoder",
    "lr_decoder",
    "lr_discriminator",
    "is_best",
    "timestamp",
]


TARGET_SCORE_WEIGHTS = {
    "val_bbox_inside_ratio": 0.45,
    "val_bbox_centroid_inside": 0.25,
    "val_area_score": 0.20,
    "val_dice": 0.10,
    "area_min": 0.10,
    "area_max": 0.40,
}


def init_target_metrics_csv(run_id: str) -> str:
    os.makedirs(CONFIG["logs_path"], exist_ok=True)
    csv_path = os.path.join(CONFIG["logs_path"], "target_training_metrics_soft_dann.csv")
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


def init_run_metrics_csv(run_dir: Path) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(_CSV_HEADER)
    log.info("Metricas de corrida -> %s", csv_path)
    return str(csv_path)


def append_csv_row(csv_path: str, row: list[object]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(row)


def compute_val_area_score(area_ratio: float, area_min: float = 0.10, area_max: float = 0.40) -> float:
    if area_min <= area_ratio <= area_max:
        return 1.0
    if area_ratio < area_min:
        return max(0.0, area_ratio / max(area_min, 1e-6))
    return max(0.0, 1.0 - (area_ratio - area_max) / max(area_max, 1e-6))


def compute_target_score(val_us_metrics: dict[str, float], dice_val: float) -> tuple[float, float]:
    area_score = compute_val_area_score(
        val_us_metrics["predicted_area_ratio"],
        area_min=TARGET_SCORE_WEIGHTS["area_min"],
        area_max=TARGET_SCORE_WEIGHTS["area_max"],
    )
    score = (
        TARGET_SCORE_WEIGHTS["val_bbox_inside_ratio"] * val_us_metrics["bbox_inside_ratio"]
        + TARGET_SCORE_WEIGHTS["val_bbox_centroid_inside"] * val_us_metrics["bbox_centroid_inside"]
        + TARGET_SCORE_WEIGHTS["val_area_score"] * area_score
        + TARGET_SCORE_WEIGHTS["val_dice"] * dice_val
    )
    return float(score), float(area_score)


def append_target_metrics(
    csv_paths: list[str],
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
    val_area_score: float,
    is_best: bool,
) -> None:
    hd95_val = val_metrics["HD95"]
    row = [
        run_id,
        epoch,
        f"{train_stats['seg_loss_mri']:.6f}",
        f"{train_stats['domain_loss_mri']:.6f}",
        f"{train_stats['domain_loss_us']:.6f}",
        f"{train_stats['domain_loss']:.6f}",
        f"{train_stats['weak_loss_us']:.6f}",
        f"{train_stats['outside_loss']:.6f}",
        f"{train_stats['inside_presence_loss']:.6f}",
        f"{train_stats['inside_mean_loss']:.6f}",
        f"{train_stats['mean_prob_inside_bbox']:.6f}",
        f"{train_stats['topk_inside_prob']:.6f}",
        f"{train_stats['area_loss']:.6f}",
        f"{train_stats['under_area_loss']:.6f}",
        f"{train_stats['over_area_loss']:.6f}",
        f"{train_stats['total_loss']:.6f}",
        f"{train_stats['domain_acc']:.6f}",
        f"{train_stats['bbox_inside_ratio']:.6f}",
        f"{train_stats['bbox_iou']:.6f}",
        f"{train_stats['bbox_centroid_inside']:.6f}",
        f"{train_stats['predicted_area_ratio']:.6f}",
        f"{CONFIG.get('weak_min_area_ratio', 0.15):.6f}",
        f"{CONFIG.get('weak_max_area_ratio', 1.0):.6f}",
        f"{val_loss:.6f}",
        f"{val_metrics['Dice']:.6f}",
        f"{hd95_val:.4f}" if np.isfinite(hd95_val) else "inf",
        f"{val_metrics['Object_Precision']:.6f}",
        f"{val_us_metrics['weak_loss_us']:.6f}",
        f"{val_us_metrics['outside_loss']:.6f}",
        f"{val_us_metrics['inside_presence_loss']:.6f}",
        f"{val_us_metrics['inside_mean_loss']:.6f}",
        f"{val_us_metrics['mean_prob_inside_bbox']:.6f}",
        f"{val_us_metrics['topk_inside_prob']:.6f}",
        f"{val_us_metrics['area_loss']:.6f}",
        f"{val_us_metrics['under_area_loss']:.6f}",
        f"{val_us_metrics['over_area_loss']:.6f}",
        f"{val_us_metrics['bbox_inside_ratio']:.6f}",
        f"{val_us_metrics['bbox_iou']:.6f}",
        f"{val_us_metrics['bbox_centroid_inside']:.6f}",
        f"{val_us_metrics['predicted_area_ratio']:.6f}",
        f"{val_us_metrics['min_area_ratio']:.6f}",
        f"{val_us_metrics['max_area_ratio']:.6f}",
        f"{val_area_score:.6f}",
        f"{target_score:.6f}",
        f"{alpha:.6f}",
        f"{lambda_mri:.6f}",
        f"{lambda_us:.6f}",
        f"{lambda_domain:.6f}",
        int(use_dann),
        weak_mode,
        f"{optimizer.param_groups[0]['lr']:.2e}",
        f"{optimizer.param_groups[1]['lr']:.2e}",
        f"{optimizer.param_groups[2]['lr']:.2e}",
        int(is_best),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]
    for csv_path in csv_paths:
        append_csv_row(csv_path, row)


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
    max_samples: int = 1,
    metric_thresholds: list[float] | None = None,
) -> None:
    if not output_dir:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    max_samples = max(1, min(int(max_samples), int(us_images.size(0))))
    metric_thresholds = metric_thresholds or [0.30, 0.35, 0.40, 0.50]

    expanded = weak_parts.get("expanded_bbox_mask")
    if expanded is None:
        expanded = expand_bbox_mask(
            bbox_masks.detach(),
            int(CONFIG.get("weak_bbox_margin_px", 8)),
        )

    probs = torch.sigmoid(seg_logits_us.detach().float()).cpu().numpy()
    images_np = us_images.detach().float().cpu().numpy()
    bboxes_np = (bbox_masks.detach().float().cpu().numpy() > 0.5)
    expanded_np_all = (expanded.detach().float().cpu().numpy() > 0.5)

    def contour_if_present(ax, mask: np.ndarray, color: str, linewidth: float) -> None:
        if np.any(mask) and np.any(~mask):
            ax.contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=linewidth)

    def threshold_metrics(prob: np.ndarray, bbox: np.ndarray, thr: float, eps: float = 1e-6) -> dict[str, float]:
        pred = prob >= thr
        pred_area = float(pred.sum())
        bbox_area = float(bbox.sum())
        intersection = float((pred & bbox).sum())
        union = float((pred | bbox).sum())
        return {
            "bin_area_ratio": pred_area / max(bbox_area, eps),
            "bbox_in": intersection / max(pred_area, eps) if pred_area > 0 else 0.0,
            "bbox_iou": intersection / max(union, eps) if union > 0 else 0.0,
        }

    for sample_idx in range(max_samples):
        image = images_np[sample_idx, 0]
        prob = probs[sample_idx, 0]
        binary = prob >= threshold
        bbox = bboxes_np[sample_idx, 0]
        expanded_np = expanded_np_all[sample_idx, 0]
        expanded_only = expanded_np & ~bbox
        outside = ~expanded_np

        bbox_area = float(bbox.sum())
        expanded_area = float(expanded_np.sum())
        prob_area_ratio = float(prob.sum() / max(bbox_area, 1.0))
        bin_area_ratio = float(binary.sum() / max(bbox_area, 1.0))
        bbox_in = float((binary & bbox).sum() / max(float(binary.sum()), 1.0))

        inside_probs = prob[bbox]
        if inside_probs.size:
            topk_fraction = float(CONFIG.get("weak_inside_fraction", 0.10))
            k = max(1, int(round(float(inside_probs.size) * topk_fraction)))
            cutoff = np.partition(inside_probs, -k)[-k]
            topk_mask = bbox & (prob >= cutoff)
            inside_mean = float(inside_probs.mean())
            inside_max = float(inside_probs.max())
            inside_topk = float(inside_probs[inside_probs >= cutoff].mean())
        else:
            topk_mask = np.zeros_like(bbox, dtype=bool)
            inside_mean = inside_max = inside_topk = 0.0

        region_rgb = np.zeros((*prob.shape, 3), dtype=np.float32)
        region_rgb[outside] = (0.10, 0.20, 0.55)
        region_rgb[expanded_only] = (1.00, 0.82, 0.05)
        region_rgb[bbox] = (0.00, 0.80, 0.35)

        thresh_lines = []
        for thr in metric_thresholds:
            metrics = threshold_metrics(prob, bbox, float(thr))
            thresh_lines.append(
                f"thr {thr:.2f}: area {metrics['bin_area_ratio']:.2f} | "
                f"in {metrics['bbox_in']:.2f} | iou {metrics['bbox_iou']:.2f}"
            )

        summary = (
            f"epoch {epoch} | step {step} | sample {sample_idx} | threshold {threshold:.2f}\n"
            f"prob_area_ratio {prob_area_ratio:.3f} | bin_area_ratio {bin_area_ratio:.3f} | "
            f"bbox_in {bbox_in:.3f}\n"
            f"inside mean/max/topk {inside_mean:.3f}/{inside_max:.3f}/{inside_topk:.3f} | "
            f"strict_area {bbox_area:.0f} | expanded_area {expanded_area:.0f}\n"
            + "\n".join(thresh_lines)
        )

        fig, axes = plt.subplots(2, 4, figsize=(18, 10), facecolor="white", constrained_layout=True)
        axes = axes.ravel()

        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        contour_if_present(axes[0], bbox, "#00c853", 1.8)
        axes[0].set_title("US + bbox estricta", fontsize=10)

        axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
        contour_if_present(axes[1], expanded_np, "#ffd600", 1.8)
        contour_if_present(axes[1], bbox, "#00c853", 1.3)
        axes[1].set_title("US + bbox expandida", fontsize=10)

        axes[2].imshow(bbox.astype(np.float32), cmap="Greens", vmin=0, vmax=1)
        axes[2].set_title("Mascara bbox estricta", fontsize=10)

        im = axes[3].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[3].set_title("Mapa sigmoid probability", fontsize=10)
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        axes[4].imshow(binary.astype(np.float32), cmap="gray", vmin=0, vmax=1)
        contour_if_present(axes[4], binary, "#ff1744", 1.5)
        axes[4].set_title("Prediccion binaria", fontsize=10)

        axes[5].imshow(region_rgb)
        axes[5].set_title("Regiones inside/outside", fontsize=10)

        axes[6].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[6].imshow(topk_mask.astype(np.float32), cmap="Reds", alpha=0.55, vmin=0, vmax=1)
        contour_if_present(axes[6], bbox, "#00c853", 1.3)
        axes[6].set_title("Top-k dentro de bbox", fontsize=10)

        axes[7].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[7].imshow(prob, cmap="magma", alpha=0.35, vmin=0, vmax=1)
        axes[7].imshow(np.ma.masked_where(~binary, binary), cmap="Reds", alpha=0.35, vmin=0, vmax=1)
        contour_if_present(axes[7], bbox, "#00c853", 1.6)
        contour_if_present(axes[7], expanded_np, "#ffd600", 1.3)
        contour_if_present(axes[7], binary, "#ff1744", 1.2)
        axes[7].set_title("Overlay final completo", fontsize=10)

        for ax in axes:
            ax.axis("off")

        fig.suptitle(summary, fontsize=10, y=1.03)
        fig.savefig(
            out / f"weak_debug_ep{epoch:03d}_step{step:04d}_sample{sample_idx:02d}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.close(fig)


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
    lambda_domain: float,
    use_dann: bool,
    weak_mode: str,
    use_amp: bool,
    steps_per_epoch: int,
    log_every: int = 10,
) -> tuple[dict[str, float], float]:
    model.train()
    # En target adaptation el dominio limitante es US: las imagenes target son
    # pocas y tienen supervision weak. La epoca debe quedar definida por US
    # para evitar reciclarlo muchas veces contra todos los batches MRI.
    mri_iter = cycle(mri_loader)
    us_iter = cycle(us_loader)
    steps_per_epoch = max(1, int(steps_per_epoch))
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
        "inside_mean_loss": 0.0,
        "mean_prob_inside_bbox": 0.0,
        "topk_inside_prob": 0.0,
        "area_loss": 0.0,
        "under_area_loss": 0.0,
        "over_area_loss": 0.0,
        "total_loss": 0.0,
        "domain_acc": 0.0,
        "bbox_inside_ratio": 0.0,
        "bbox_iou": 0.0,
        "bbox_centroid_inside": 0.0,
        "predicted_area_ratio": 0.0,
    }
    last_alpha = 0.0
    metric_count = 0
    last_us_metrics = {
        "bbox_inside_ratio": 0.0,
        "bbox_iou": 0.0,
        "bbox_centroid_inside": 0.0,
        "predicted_area_ratio": 0.0,
    }

    for step in range(steps_per_epoch):
        global_step = start_step + step
        alpha = compute_lambda_schedule(global_step, total_steps) if use_dann else 0.0
        last_alpha = alpha

        mri_images, mri_masks = next(mri_iter)
        us_images, us_weak_masks, us_bbox_masks = next(us_iter)
        mri_images = mri_images.to(device, non_blocking=True)
        mri_masks = mri_masks.to(device, non_blocking=True)
        us_images = us_images.to(device, non_blocking=True)
        us_weak_masks = us_weak_masks.to(device, non_blocking=True)
        us_bbox_masks = us_bbox_masks.to(device, non_blocking=True)

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
            seg_logits_all, domain_logits_all = model(
                all_images,
                alpha=alpha,
                return_segmentation=True,
            )

            assert seg_logits_all is not None
            mri_batch_size = mri_images.size(0)
            seg_logits = seg_logits_all[:mri_batch_size]
            seg_logits_us = seg_logits_all[mri_batch_size:]
            domain_logits_mri = domain_logits_all[:mri_batch_size]
            domain_logits_us = domain_logits_all[mri_batch_size:]

            seg_loss = bce_dice_loss(seg_logits, mri_masks)
            domain_loss_mri = F.cross_entropy(domain_logits_mri, mri_labels)
            domain_loss_us = F.cross_entropy(domain_logits_us, us_labels)
            domain_loss = F.cross_entropy(domain_logits_all, all_domain_labels)
            weak_loss_us, weak_parts = compute_us_weak_loss(seg_logits_us, us_weak_masks, weak_mode)
            weighted_domain = lambda_domain * domain_loss if use_dann else domain_loss * 0.0
            total_loss = lambda_mri * seg_loss + lambda_us * weak_loss_us + weighted_domain

            debug_every = int(CONFIG.get("weak_debug_every", 0))
            debug_max_batches = int(CONFIG.get("weak_debug_max_batches", 1))
            debug_max_samples = int(CONFIG.get("weak_debug_max_samples", 1))
            debug_dir = str(CONFIG.get("weak_debug_dir", ""))
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
                    max_samples=debug_max_samples,
                    metric_thresholds=CONFIG.get("us_metric_thresholds", [0.30, 0.35, 0.40, 0.50]),
                )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = 0.5 * (
            domain_accuracy(domain_logits_mri, mri_labels)
            + domain_accuracy(domain_logits_us, us_labels)
        )
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
        totals["inside_mean_loss"] += float(weak_parts["inside_mean_loss"].item())
        totals["mean_prob_inside_bbox"] += float(weak_parts["mean_prob_inside_bbox"].item())
        totals["topk_inside_prob"] += float(weak_parts["topk_inside_prob"].item())
        totals["area_loss"] += float(weak_parts["area_loss"].item())
        totals["under_area_loss"] += float(weak_parts["under_area_loss"].item())
        totals["over_area_loss"] += float(weak_parts["over_area_loss"].item())
        totals["total_loss"] += float(total_loss.item())
        totals["domain_acc"] += acc

        if step % log_every == 0:
            log.info(
                (
                    "Ep %02d | Step %4d/%d | alpha %.3f | seg_mri %.4f | "
                    "weak_us %.4f (out %.4f | in %.4f | mean_loss %.4f | "
                    "mean_prob %.4f | topk %.4f | area %.4f | under %.4f | over %.4f) | "
                    "domain_loss %.4f | total %.4f | dom_acc %.3f | bbox_in %.3f | area_ratio %.3f"
                    " [min %.2f | max %.2f]"
                ),
                epoch,
                step,
                steps_per_epoch,
                alpha,
                seg_loss.item(),
                weak_loss_us.item(),
                weak_parts["outside_loss"].item(),
                weak_parts["inside_presence_loss"].item(),
                weak_parts["inside_mean_loss"].item(),
                weak_parts["mean_prob_inside_bbox"].item(),
                weak_parts["topk_inside_prob"].item(),
                weak_parts["area_loss"].item(),
                weak_parts["under_area_loss"].item(),
                weak_parts["over_area_loss"].item(),
                domain_loss.item(),
                total_loss.item(),
                acc,
                last_us_metrics["bbox_inside_ratio"],
                last_us_metrics["predicted_area_ratio"],
                float(CONFIG.get("weak_min_area_ratio", 0.15)),
                float(CONFIG.get("weak_max_area_ratio", 1.0)),
            )

    n_steps = steps_per_epoch
    stats = {key: value / n_steps for key, value in totals.items()}
    metric_denominator = max(1, metric_count)
    for key in ("bbox_inside_ratio", "bbox_iou", "bbox_centroid_inside", "predicted_area_ratio"):
        stats[key] = totals[key] / metric_denominator
    return stats, last_alpha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena target MRI->US con MRI fuerte, US weak y DANN suave."
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
        default=float(CONFIG.get("lambda_domain", 0.0)),
        help="Peso para regularizacion adversarial DANN. Default: 0.0.",
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
    parser.add_argument("--outside_weight", type=float, default=float(CONFIG.get("weak_outside_weight", 1.0)))
    parser.add_argument("--inside_weight", type=float, default=float(CONFIG.get("weak_inside_weight", 0.5)))
    parser.add_argument("--area_weight", type=float, default=float(CONFIG.get("weak_area_weight", 0.05)))
    parser.add_argument(
        "--min_inside_activation",
        type=float,
        default=float(CONFIG.get("weak_min_inside_activation", 0.20)),
    )
    parser.add_argument(
        "--min_inside_mean",
        type=float,
        default=float(CONFIG.get("weak_min_inside_mean", 0.08)),
    )
    parser.add_argument(
        "--inside_mean_weight",
        type=float,
        default=float(CONFIG.get("weak_inside_mean_weight", 0.5)),
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
        help=(
            "Compatibilidad legacy. En corridas nuevas el debug se guarda en "
            "logs/checkpoints/<run_slug>/debug."
        ),
    )
    parser.add_argument(
        "--weak_debug_every",
        type=int,
        default=int(CONFIG.get("weak_debug_every", 0)),
        help="Guardar debug visual cada N epocas. 0 desactiva.",
    )
    parser.add_argument(
        "--weak_debug_max_batches",
        type=int,
        default=int(CONFIG.get("weak_debug_max_batches", 1)),
        help="Maximo de batches US por epoca para guardar debug visual.",
    )
    parser.add_argument(
        "--weak_debug_max_samples",
        type=int,
        default=int(CONFIG.get("weak_debug_max_samples", 1)),
        help="Maximo de samples US por batch para guardar debug visual.",
    )
    parser.add_argument(
        "--us_metric_thresholds",
        type=str,
        default=",".join(str(v) for v in CONFIG.get("us_metric_thresholds", [0.30, 0.35, 0.40, 0.50])),
        help="Thresholds separados por coma para metricas US de debug.",
    )
    parser.add_argument(
        "--target_steps_per_epoch",
        type=int,
        default=int(CONFIG.get("target_steps_per_epoch", 0)),
        help="Steps por epoca target. <=0 usa len(us_loader) para ver US una vez por epoca.",
    )
    return parser.parse_args()


def parse_threshold_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("us_metric_thresholds no puede quedar vacio.")
    return values


def safe_run_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def build_run_slug(run_id: str, lambda_us: float, lambda_domain: float, use_dann: bool, weak_mode: str) -> str:
    suffix = f"{weak_mode}_us{lambda_us:g}"
    if use_dann:
        suffix += f"_dom{lambda_domain:g}_dann1"
    return safe_run_slug(f"{run_id}_{suffix}")


def write_run_config(run_dir: Path, run_id: str, run_slug: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "run_slug": run_slug,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_score_weights": TARGET_SCORE_WEIGHTS,
        "config": CONFIG,
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def train(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    lambda_mri = args.lambda_mri
    lambda_us = args.lambda_us
    lambda_domain = args.lambda_domain
    use_dann = args.use_dann
    weak_mode = args.weak_mode
    run_slug = build_run_slug(run_id, lambda_us, lambda_domain, use_dann, weak_mode)
    CONFIG["weak_loss_type"] = args.weak_loss_type
    CONFIG["weak_bbox_margin_px"] = args.bbox_margin_px
    CONFIG["weak_outside_weight"] = args.outside_weight
    CONFIG["weak_inside_weight"] = args.inside_weight
    CONFIG["weak_area_weight"] = args.area_weight
    CONFIG["weak_min_inside_activation"] = args.min_inside_activation
    CONFIG["weak_min_inside_mean"] = args.min_inside_mean
    CONFIG["weak_inside_mean_weight"] = args.inside_mean_weight
    CONFIG["weak_max_area_ratio"] = args.max_area_ratio
    CONFIG["weak_debug_every"] = args.weak_debug_every
    CONFIG["weak_debug_max_batches"] = args.weak_debug_max_batches
    CONFIG["weak_debug_max_samples"] = args.weak_debug_max_samples
    CONFIG["us_metric_thresholds"] = parse_threshold_list(args.us_metric_thresholds)
    CONFIG["lambda_domain"] = lambda_domain
    CONFIG["use_dann"] = use_dann
    CONFIG["target_steps_per_epoch"] = args.target_steps_per_epoch
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
    log.info("  MiomaVision - Entrenamiento target MRI -> US")
    log.info("  Run ID:      %s", run_id)
    log.info("  Dispositivo: %s", device)
    log.info("  Epocas:      %d", CONFIG["epochs"])
    log.info("  Lambda MRI:  %.4f", lambda_mri)
    log.info("  Lambda US:   %.4f", lambda_us)
    log.info("  Lambda dom:  %.4f", lambda_domain if use_dann else 0.0)
    log.info("  DANN:        %s", "ON (regularizador suave)" if use_dann else "OFF")
    log.info("  Weak mode:   %s", weak_mode)
    log.info("  Weak loss:   %s", CONFIG["weak_loss_type"])
    log.info(
        "  Debug weak:  every=%d | max_batches=%d | max_samples=%d | thresholds=%s",
        CONFIG["weak_debug_every"],
        CONFIG["weak_debug_max_batches"],
        CONFIG["weak_debug_max_samples"],
        ",".join(f"{thr:.2f}" for thr in CONFIG["us_metric_thresholds"]),
    )
    log.info(
        "  Soft bbox:   margin=%d | outside=%.3f | inside=%.3f | area=%.3f | "
        "min_inside=%.3f | min_inside_mean=%.3f | mean_w=%.3f | "
        "area_bounds=[%.3f, %.3f] | under_w=%.3f | over_w=%.3f",
        CONFIG["weak_bbox_margin_px"],
        CONFIG["weak_outside_weight"],
        CONFIG["weak_inside_weight"],
        CONFIG["weak_area_weight"],
        CONFIG["weak_min_inside_activation"],
        CONFIG["weak_min_inside_mean"],
        CONFIG["weak_inside_mean_weight"],
        CONFIG["weak_min_area_ratio"],
        CONFIG["weak_max_area_ratio"],
        CONFIG["weak_under_area_weight"],
        CONFIG["weak_over_area_weight"],
    )
    log.info("  AMP:         %s", "ON" if use_amp else "OFF")
    log.info("  Log every:   %d step(s)", log_every)
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
        "Dataloaders simultaneos: la epoca target se gobierna por US y se usa cycle(MRI). "
        "Asi US se ve aproximadamente una vez por epoca y MRI aporta supervision fuerte."
    )

    num_workers = CONFIG.get("num_workers", 0)
    pin_mem = device.type == "cuda"

    mri_train_loader = DataLoader(
        SagitalDataset(tr_imgs, tr_masks),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
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
    ckpt_dir = Path(CONFIG["logs_path"]) / "checkpoints" / run_slug
    CONFIG["weak_debug_dir"] = str(ckpt_dir / "debug")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(ckpt_dir, run_id, run_slug)
    write_run_config(Path(CONFIG["weak_debug_dir"]), run_id, run_slug)
    csv_path = init_target_metrics_csv(run_id)
    run_csv_path = init_run_metrics_csv(ckpt_dir)
    best_path = str(ckpt_dir / "best_model.pth")
    last_path = str(ckpt_dir / "last_model.pth")
    log.info("Artefactos de corrida: checkpoints=%s | debug=%s", ckpt_dir, CONFIG["weak_debug_dir"])

    best_dice = 0.0
    best_target_score = -np.inf
    configured_steps = int(CONFIG.get("target_steps_per_epoch", 0))
    steps_per_epoch = configured_steps if configured_steps > 0 else len(us_train_loader)
    steps_per_epoch = max(1, steps_per_epoch)
    total_steps = CONFIG["epochs"] * steps_per_epoch
    global_step = 0
    log.info(
        "Steps target por epoca: %d (MRI batches=%d | US batches=%d | target_steps_per_epoch=%d)",
        steps_per_epoch,
        len(mri_train_loader),
        len(us_train_loader),
        configured_steps,
    )

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
            lambda_domain=lambda_domain,
            use_dann=use_dann,
            weak_mode=weak_mode,
            use_amp=use_amp,
            steps_per_epoch=steps_per_epoch,
            log_every=log_every,
        )
        global_step += steps_per_epoch

        val_metrics, inf_batches, val_loss = validate_mri(model, mri_val_loader, device)
        val_us_metrics = validate_us_weak(model, us_val_loader, device, weak_mode)
        if inf_batches > 0:
            log.warning("HD95=inf en %d batch(es) de validacion.", inf_batches)

        dice_val = val_metrics["Dice"]
        target_score, val_area_score = compute_target_score(val_us_metrics, dice_val)
        is_best = target_score > best_target_score
        scheduler.step(target_score)

        if is_best:
            best_target_score = target_score
            best_dice = max(best_dice, dice_val)

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_dice,
            val_metrics,
            val_us_metrics,
            alpha,
            lambda_mri,
            lambda_us,
            lambda_domain,
            use_dann,
            weak_mode,
        )
        epoch_ckpt_every = int(CONFIG.get("target_save_epoch_checkpoints", 0))
        if epoch_ckpt_every > 0 and epoch % epoch_ckpt_every == 0:
            epoch_path = str(ckpt_dir / f"epoch_{epoch:03d}_model.pth")
            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                epoch,
                best_dice,
                val_metrics,
                val_us_metrics,
                alpha,
                lambda_mri,
                lambda_us,
                lambda_domain,
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
                val_metrics,
                val_us_metrics,
                alpha,
                lambda_mri,
                lambda_us,
                lambda_domain,
                use_dann,
                weak_mode,
            )

        append_target_metrics(
            [csv_path, run_csv_path],
            run_id,
            epoch,
            train_stats,
            val_metrics,
            val_us_metrics,
            val_loss,
            alpha,
            lambda_mri,
            lambda_us,
            lambda_domain,
            use_dann,
            weak_mode,
            optimizer,
            target_score,
            val_area_score,
            is_best,
        )

        hd95_val = val_metrics["HD95"]
        hd95_str = f"{hd95_val:.2f}px" if np.isfinite(hd95_val) else "inf"
        log.info(
            (
                "Epoca %02d/%02d | total %.4f | seg_mri %.4f | weak_us %.4f "
                "(out %.4f | in %.4f | mean_loss %.4f | mean_prob %.4f | "
                "topk %.4f | area %.4f | under %.4f | over %.4f) | "
                "domain_loss %.4f | val_loss %.4f | Dice %.4f | HD95 %s | dom_acc %.3f | "
                "bbox_in %.3f | val_bbox_in %.3f | area_ratio %.3f | val_area_ratio %.3f | "
                "val_under %.4f | val_over %.4f | area_bounds [%.2f, %.2f] | "
                "val_area_score %.3f | target_score %.4f | %.1fs%s"
            ),
            epoch,
            CONFIG["epochs"],
            train_stats["total_loss"],
            train_stats["seg_loss_mri"],
            train_stats["weak_loss_us"],
            train_stats["outside_loss"],
            train_stats["inside_presence_loss"],
            train_stats["inside_mean_loss"],
            train_stats["mean_prob_inside_bbox"],
            train_stats["topk_inside_prob"],
            train_stats["area_loss"],
            train_stats["under_area_loss"],
            train_stats["over_area_loss"],
            train_stats["domain_loss"],
            val_loss,
            dice_val,
            hd95_str,
            train_stats["domain_acc"],
            train_stats["bbox_inside_ratio"],
            val_us_metrics["bbox_inside_ratio"],
            train_stats["predicted_area_ratio"],
            val_us_metrics["predicted_area_ratio"],
            val_us_metrics["under_area_loss"],
            val_us_metrics["over_area_loss"],
            val_us_metrics["min_area_ratio"],
            val_us_metrics["max_area_ratio"],
            val_area_score,
            target_score,
            time.time() - t0,
            " * BEST" if is_best else "",
        )

    log.info("=" * 60)
    log.info("Entrenamiento target finalizado")
    log.info("Mejor Dice: %.4f", best_dice)
    log.info("Mejor target score: %.4f", best_target_score)
    log.info("Best checkpoint: %s", best_path)
    log.info("Last checkpoint: %s", last_path)
    log.info("Metricas: %s", csv_path)
    log.info("=" * 60)


if __name__ == "__main__":
    train()
