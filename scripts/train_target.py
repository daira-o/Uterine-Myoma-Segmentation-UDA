"""
scripts/train_target.py
Entrenamiento DANN MRI -> US para la Attention U-Net.

Usa MRI con mascaras para segmentation loss y MRI/US para domain loss:
    total = segmentation_loss + lambda_domain * (domain_loss_mri + domain_loss_us) / 2
"""

from __future__ import annotations

import csv
import datetime
import glob
import logging
import os
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CONFIG
from models.attention_unet import bce_dice_loss, compute_all_metrics
from models.dann_unet import DANNUNet
from models.grl import compute_lambda_schedule
from scripts.train_source import SagitalDataset, load_split_paths, log_split_diagnostics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


MRI_DOMAIN = 0
US_DOMAIN = 1


class UltrasoundDataset(Dataset):
    """Dataset target sin mascaras: devuelve imagen US [1, H, W]."""

    def __init__(self, img_paths: list[str]) -> None:
        if not img_paths:
            raise ValueError("UltrasoundDataset recibio una lista vacia.")
        self.img_paths = img_paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = np.load(self.img_paths[idx]).astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 3 and img.shape[0] != 1:
            img = np.moveaxis(img, -1, 0)
        return torch.from_numpy(img)


def load_us_image_paths(base_path: str, split: str = "train") -> list[str]:
    """Carga imagenes US desde <base_path>/<split>/images/*.npy."""
    img_dir = os.path.join(base_path, split, "images")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
    if not imgs:
        raise FileNotFoundError(
            f"No se encontraron .npy en '{img_dir}'. "
            "Ejecuta scripts/build_us_splits_from_clean.py tras la curacion "
            "o revisa US_READY_PATH."
        )
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
        or os.path.join(ROOT, "data_ready_US_curated")
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


_CSV_HEADER = [
    "run_id",
    "epoch",
    "seg_loss",
    "domain_loss_mri",
    "domain_loss_us",
    "domain_loss",
    "total_loss",
    "domain_acc",
    "val_loss",
    "val_dice",
    "val_hd95",
    "val_obj_precision",
    "alpha",
    "lambda_domain",
    "lr_encoder",
    "lr_decoder",
    "lr_discriminator",
    "is_best",
    "timestamp",
]


def init_target_metrics_csv(run_id: str) -> str:
    os.makedirs(CONFIG["logs_path"], exist_ok=True)
    csv_path = os.path.join(CONFIG["logs_path"], "target_training_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(_CSV_HEADER)
        log.info("Metricas target -> %s (archivo nuevo)", csv_path)
    else:
        log.info("Metricas target -> %s (append)", csv_path)
    return csv_path


def append_target_metrics(
    csv_path: str,
    run_id: str,
    epoch: int,
    train_stats: dict[str, float],
    val_metrics: dict,
    val_loss: float,
    alpha: float,
    lambda_domain: float,
    optimizer: torch.optim.Optimizer,
    is_best: bool,
) -> None:
    hd95_val = val_metrics["HD95"]
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            [
                run_id,
                epoch,
                f"{train_stats['seg_loss']:.6f}",
                f"{train_stats['domain_loss_mri']:.6f}",
                f"{train_stats['domain_loss_us']:.6f}",
                f"{train_stats['domain_loss']:.6f}",
                f"{train_stats['total_loss']:.6f}",
                f"{train_stats['domain_acc']:.6f}",
                f"{val_loss:.6f}",
                f"{val_metrics['Dice']:.6f}",
                f"{hd95_val:.4f}" if np.isfinite(hd95_val) else "inf",
                f"{val_metrics['Object_Precision']:.6f}",
                f"{alpha:.6f}",
                f"{lambda_domain:.6f}",
                f"{optimizer.param_groups[0]['lr']:.2e}",
                f"{optimizer.param_groups[1]['lr']:.2e}",
                f"{optimizer.param_groups[2]['lr']:.2e}",
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
    val_metrics: dict,
    alpha: float,
    lambda_domain: float,
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
            "alpha": alpha,
            "lambda_domain": lambda_domain,
            "config": CONFIG,
        },
        path,
    )


def run_epoch(
    model: DANNUNet,
    mri_loader: DataLoader,
    us_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_steps: int,
    start_step: int,
    lambda_domain: float,
    log_every: int = 50,
) -> tuple[dict[str, float], float]:
    model.train()
    us_iter = cycle(us_loader)

    totals = {
        "seg_loss": 0.0,
        "domain_loss_mri": 0.0,
        "domain_loss_us": 0.0,
        "domain_loss": 0.0,
        "total_loss": 0.0,
        "domain_acc": 0.0,
    }
    last_alpha = 0.0

    for step, (mri_images, mri_masks) in enumerate(mri_loader):
        global_step = start_step + step
        alpha = compute_lambda_schedule(global_step, total_steps)
        last_alpha = alpha

        mri_images = mri_images.to(device)
        mri_masks = mri_masks.to(device)
        us_images = next(us_iter).to(device)

        mri_labels = torch.full(
            (mri_images.size(0),), MRI_DOMAIN, dtype=torch.long, device=device
        )
        us_labels = torch.full(
            (us_images.size(0),), US_DOMAIN, dtype=torch.long, device=device
        )

        optimizer.zero_grad()

        seg_logits, domain_logits_mri = model(
            mri_images,
            alpha=alpha,
            return_segmentation=True,
        )
        _, domain_logits_us = model(
            us_images,
            alpha=alpha,
            return_segmentation=False,
        )

        assert seg_logits is not None
        seg_loss = bce_dice_loss(seg_logits, mri_masks)
        domain_loss_mri = F.cross_entropy(domain_logits_mri, mri_labels)
        domain_loss_us = F.cross_entropy(domain_logits_us, us_labels)
        domain_loss = 0.5 * (domain_loss_mri + domain_loss_us)
        total_loss = seg_loss + lambda_domain * domain_loss

        total_loss.backward()
        optimizer.step()

        acc = 0.5 * (
            domain_accuracy(domain_logits_mri, mri_labels)
            + domain_accuracy(domain_logits_us, us_labels)
        )

        totals["seg_loss"] += float(seg_loss.item())
        totals["domain_loss_mri"] += float(domain_loss_mri.item())
        totals["domain_loss_us"] += float(domain_loss_us.item())
        totals["domain_loss"] += float(domain_loss.item())
        totals["total_loss"] += float(total_loss.item())
        totals["domain_acc"] += acc

        if step % log_every == 0:
            log.info(
                "Ep %02d | Step %4d/%d | alpha %.3f | seg %.4f | dom %.4f | total %.4f | dom_acc %.3f",
                epoch,
                step,
                len(mri_loader),
                alpha,
                seg_loss.item(),
                domain_loss.item(),
                total_loss.item(),
                acc,
            )

    n_steps = max(1, len(mri_loader))
    return {key: value / n_steps for key, value in totals.items()}, last_alpha


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    lambda_domain = CONFIG.get("lambda_domain", 0.1)
    freeze_decoder_epochs = CONFIG.get("freeze_decoder_epochs", 0)
    log_every = CONFIG.get("log_every", 50)

    log.info("=" * 60)
    log.info("  MiomaVision - Entrenamiento DANN MRI -> US")
    log.info("  Run ID:      %s", run_id)
    log.info("  Dispositivo: %s", device)
    log.info("  Epocas:      %d", CONFIG["epochs"])
    log.info("  Lambda dom:  %.4f", lambda_domain)
    log.info("=" * 60)

    mri_base_path = CONFIG["base_path"]
    us_base_path = get_us_base_path()

    tr_imgs, tr_masks = load_split_paths(mri_base_path, "train")
    val_imgs, val_masks = load_split_paths(mri_base_path, "val")
    us_train_imgs = load_us_image_paths(us_base_path, "train")

    log.info("- Diagnostico MRI -")
    log_split_diagnostics(tr_imgs, "train")
    log_split_diagnostics(val_imgs, "val")
    log.info("US train: %d imagenes | base: %s", len(us_train_imgs), us_base_path)
    log.info(
        "Dataloaders simultaneos: se itera por MRI y se usa cycle(US). "
        "Asi cada paso tiene mascara MRI y dominio US aunque las longitudes difieran."
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
        UltrasoundDataset(us_train_imgs),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
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

    ckpt_dir = Path(CONFIG["logs_path"]) / "checkpoints_dann"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(ckpt_dir / "best_model_dann.pth")
    last_path = str(ckpt_dir / "last_model_dann.pth")

    best_dice = 0.0
    total_steps = CONFIG["epochs"] * max(1, len(mri_train_loader))
    global_step = 0

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
            lambda_domain=lambda_domain,
            log_every=log_every,
        )
        global_step += len(mri_train_loader)

        val_metrics, inf_batches, val_loss = validate_mri(model, mri_val_loader, device)
        if inf_batches > 0:
            log.warning("HD95=inf en %d batch(es) de validacion.", inf_batches)

        dice_val = val_metrics["Dice"]
        is_best = dice_val > best_dice
        scheduler.step(dice_val)

        if is_best:
            best_dice = dice_val

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_dice,
            val_metrics,
            alpha,
            lambda_domain,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_dice,
                val_metrics,
                alpha,
                lambda_domain,
            )

        append_target_metrics(
            csv_path,
            run_id,
            epoch,
            train_stats,
            val_metrics,
            val_loss,
            alpha,
            lambda_domain,
            optimizer,
            is_best,
        )

        hd95_val = val_metrics["HD95"]
        hd95_str = f"{hd95_val:.2f}px" if np.isfinite(hd95_val) else "inf"
        log.info(
            "Epoca %02d/%02d | total %.4f | seg %.4f | dom %.4f | "
            "val_loss %.4f | Dice %.4f | HD95 %s | dom_acc %.3f | %.1fs%s",
            epoch,
            CONFIG["epochs"],
            train_stats["total_loss"],
            train_stats["seg_loss"],
            train_stats["domain_loss"],
            val_loss,
            dice_val,
            hd95_str,
            train_stats["domain_acc"],
            time.time() - t0,
            " * BEST" if is_best else "",
        )

    log.info("=" * 60)
    log.info("Entrenamiento DANN finalizado")
    log.info("Mejor Dice: %.4f", best_dice)
    log.info("Best checkpoint: %s", best_path)
    log.info("Last checkpoint: %s", last_path)
    log.info("Metricas: %s", csv_path)
    log.info("=" * 60)


if __name__ == "__main__":
    train()
