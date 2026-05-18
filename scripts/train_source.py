"""
scripts/train.py
─────────────────────────────────────────────────────────────────────────────
Entrenamiento del modelo Attention U-Net para segmentación de miomas.

Compatible con la estructura de salida de mri_pipeline.py:
    <base_path>/
        train/images/*.npy  &  train/masks/*.npy
        val/images/*.npy    &  val/masks/*.npy
        test/images/*.npy   &  test/masks/*.npy

Los splits ya fueron realizados a nivel de paciente en el pipeline de
preparación de datos (mri_pipeline.py). Este script los consume directamente,
sin re-dividir, garantizando aislamiento estadístico absoluto.

Uso:
    python scripts/train.py

Requisitos:
    pip install python-dotenv torch numpy scikit-learn scipy scikit-image
"""

from __future__ import annotations

import csv
import datetime
import glob
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Importar config y modelo desde la raíz del proyecto ──────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CONFIG
from models.attention_unet import AttentionUNet, bce_dice_loss, compute_all_metrics

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CARGA DE RUTAS
# ─────────────────────────────────────────────────────────────────────────────

def load_split_paths(
    base_path: str,
    split: str,
) -> tuple[list[str], list[str]]:
    """
    Carga las rutas de imágenes y máscaras para un split dado.

    Espera la estructura generada por mri_pipeline.py:
        <base_path>/<split>/images/*.npy
        <base_path>/<split>/masks/*.npy

    Parámetros
    ----------
    base_path : str
        Carpeta raíz con los splits (salida de mri_pipeline.py).
    split : str
        Nombre del split: "train", "val" o "test".

    Retorna
    -------
    (img_paths, mask_paths) ordenados; ambas listas tienen la misma longitud.

    Lanza
    -----
    FileNotFoundError si no se encuentran archivos .npy para el split.
    ValueError si el número de imágenes y máscaras no coincide.
    """
    img_dir  = os.path.join(base_path, split, "images")
    mask_dir = os.path.join(base_path, split, "masks")

    imgs  = sorted(glob.glob(os.path.join(img_dir,  "*.npy")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))

    if not imgs:
        raise FileNotFoundError(
            f"No se encontraron .npy en '{img_dir}'.\n"
            "Verificá DATA_PATH en tu .env y que mri_pipeline.py haya corrido."
        )
    if len(imgs) != len(masks):
        raise ValueError(
            f"Desbalance en split '{split}': "
            f"{len(imgs)} imágenes vs {len(masks)} máscaras."
        )

    log.info(
        "Split %-5s → %d pares (imágenes/máscaras) cargados.",
        split, len(imgs),
    )
    return imgs, masks


def get_patient_id(path: str) -> str:
    """
    Extrae el ID de paciente del nombre de archivo .npy.

    Formato esperado: {patient_id}_sag_{slice_index}.npy
    Ejemplo: PAT_042_sag_37.npy → "PAT_042"

    Si el nombre no cumple el patrón, retorna el stem completo como fallback.
    """
    stem = Path(path).stem
    m = re.match(r"^(.+)_sag_\d+$", stem)
    return m.group(1) if m else stem


def log_split_diagnostics(imgs: list[str], split: str) -> None:
    """
    Imprime estadísticas de distribución de cortes por paciente para un split.
    Advierte si algún paciente domina más del 10% del split.

    Parámetros
    ----------
    imgs : list[str]
        Lista de rutas de imágenes del split.
    split : str
        Nombre del split (solo para el mensaje de log).
    """
    groups = np.array([get_patient_id(p) for p in imgs])
    unique_patients = np.unique(groups)
    counts = [int((groups == pid).sum()) for pid in unique_patients]

    log.info(
        "  %-5s: %d slices | %d pacientes | "
        "slices/paciente: min=%d max=%d media=%.1f",
        split, len(imgs), len(unique_patients),
        min(counts), max(counts), np.mean(counts),
    )

    dominant = [
        pid for pid, c in zip(unique_patients, counts)
        if c / len(imgs) > 0.10
    ]
    if dominant:
        log.warning(
            "  Pacientes que dominan >10%% del split '%s': %s. "
            "Considera bajar max_slices_per_patient en el pipeline.",
            split, dominant,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SagitalDataset(Dataset):
    """
    Dataset PyTorch para cortes sagitales 2D preprocesados en formato .npy.

    Cada muestra es un par (imagen, máscara) de forma (1, H, W) en float32,
    listos para entrar directamente a la Attention U-Net.

    Parámetros
    ----------
    img_paths : list[str]
        Rutas a los archivos .npy de imagen.
    mask_paths : list[str]
        Rutas a los archivos .npy de máscara (mismo orden que img_paths).
    """

    def __init__(self, img_paths: list[str], mask_paths: list[str]) -> None:
        assert len(img_paths) == len(mask_paths), (
            "img_paths y mask_paths deben tener la misma longitud."
        )
        self.img_paths  = img_paths
        self.mask_paths = mask_paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img  = np.load(self.img_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.float32)

        # Añadir dimensión de canal: (H, W) → (1, H, W)
        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  MÉTRICAS CSV
# ─────────────────────────────────────────────────────────────────────────────

_CSV_HEADER = [
    "run_id", "epoch", "avg_loss",
    "dice", "hd95", "obj_precision",
    "hd95_inf_batches", "is_best", "lr", "timestamp",
]


def init_metrics_csv(run_id: str) -> str:
    """
    Crea (o abre en modo append) el CSV de métricas para este run.

    Escribe la cabecera solo si el archivo no existe previamente.

    Parámetros
    ----------
    run_id : str
        Identificador único del run (timestamp de inicio).

    Retorna
    -------
    Ruta absoluta al archivo CSV.
    """
    os.makedirs(CONFIG["logs_path"], exist_ok=True)
    csv_path = os.path.join(CONFIG["logs_path"], "training_metrics.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(_CSV_HEADER)
        log.info("Métricas → %s (archivo nuevo)", csv_path)
    else:
        log.info("Métricas → %s (append a run anterior)", csv_path)

    return csv_path


def append_epoch_metrics(
    csv_path: str,
    run_id: str,
    epoch: int,
    avg_loss: float,
    metrics: dict,
    hd95_inf_batches: int,
    is_best: bool,
    current_lr: float,
) -> None:
    """
    Añade una fila al CSV con las métricas de la época actual.

    Parámetros
    ----------
    csv_path : str
        Ruta al CSV creado por init_metrics_csv.
    run_id : str
        Identificador del run.
    epoch : int
        Número de época (1-indexed).
    avg_loss : float
        Pérdida promedio de la época de entrenamiento.
    metrics : dict
        Diccionario con claves "Dice", "HD95", "Object_Precision".
    hd95_inf_batches : int
        Número de batches donde HD95 fue infinito.
    is_best : bool
        True si este epoch supera el mejor Dice registrado.
    current_lr : float
        Learning rate actual (útil para detectar ajustes del scheduler).
    """
    hd95_val = metrics["HD95"]
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([
            run_id,
            epoch,
            f"{avg_loss:.6f}",
            f"{metrics['Dice']:.6f}",
            f"{hd95_val:.4f}" if np.isfinite(hd95_val) else "inf",
            f"{metrics['Object_Precision']:.6f}",
            hd95_inf_batches,
            int(is_best),
            f"{current_lr:.2e}",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[dict, int]:
    """
    Evalúa el modelo sobre el conjunto de validación.

    Agrega métricas por batch y promedia al final. Los batches donde HD95
    resulta infinito (predicción vacía frente a GT positivo) se contabilizan
    pero se excluyen del promedio de HD95 para evitar distorsión.

    Parámetros
    ----------
    model : torch.nn.Module
        Modelo en modo eval.
    val_loader : DataLoader
        DataLoader del split de validación.
    device : torch.device
        Dispositivo de cómputo.

    Retorna
    -------
    (metrics_dict, inf_batch_count)
        metrics_dict tiene claves "Dice", "HD95", "Object_Precision".
    """
    model.eval()
    batch_dices, batch_hd95s, batch_oprecs = [], [], []
    inf_count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            metrics = compute_all_metrics(
                model(images), masks,
                threshold=CONFIG["threshold"],
                iou_threshold=CONFIG["iou_threshold"],
            )
            batch_dices.append(metrics["Dice"])
            batch_oprecs.append(metrics["Object_Precision"])

            if np.isfinite(metrics["HD95"]):
                batch_hd95s.append(metrics["HD95"])
            else:
                inf_count += 1

    if inf_count > 0:
        log.warning(
            "HD95=inf en %d batch(es) "
            "(predicción vacía frente a GT positivo). Excluidos del promedio.",
            inf_count,
        )

    return {
        "Dice":             float(np.mean(batch_dices))  if batch_dices  else 0.0,
        "HD95":             float(np.mean(batch_hd95s))  if batch_hd95s  else np.inf,
        "Object_Precision": float(np.mean(batch_oprecs)) if batch_oprecs else 0.0,
    }, inf_count


# ─────────────────────────────────────────────────────────────────────────────
#  LOOP DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 50,
) -> float:
    """
    Ejecuta una época completa de entrenamiento.

    Parámetros
    ----------
    model : torch.nn.Module
    train_loader : DataLoader
    optimizer : torch.optim.Optimizer
    device : torch.device
    epoch : int
        Número de época actual (1-indexed, solo para logging).
    log_every : int
        Frecuencia de logging por steps.

    Retorna
    -------
    avg_loss : float
        Pérdida promedio sobre todos los batches de la época.
    """
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        loss = bce_dice_loss(model(images), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % log_every == 0:
            log.info(
                "Ep %02d | Step %4d/%d | Loss: %.4f",
                epoch, step, len(train_loader), loss.item(),
            )

    return total_loss / len(train_loader)


def train() -> None:
    """
    Función principal de entrenamiento.

    Flujo:
        1. Carga los splits ya preparados por mri_pipeline.py.
        2. Construye DataLoaders sin re-dividir (splits son herméticos).
        3. Inicializa modelo, optimizador, scheduler y CSV de métricas.
        4. Corre el loop de épocas con validación, checkpoint y LR scheduling.
    """
    # ── Dispositivo ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("═" * 60)
    log.info("  MiomaVision — Entrenamiento")
    log.info("  Run ID:      %s", run_id)
    log.info("  Dispositivo: %s", device)
    log.info("  Épocas:      %d", CONFIG["epochs"])
    log.info("  LR inicial:  %s", CONFIG["lr"])
    log.info("═" * 60)

    # ── Carga de datos (splits pre-divididos por mri_pipeline.py) ─────────────
    base_path = CONFIG["base_path"]

    tr_imgs,  tr_masks  = load_split_paths(base_path, "train")
    val_imgs, val_masks = load_split_paths(base_path, "val")

    log.info("─ Diagnóstico de splits ─")
    log_split_diagnostics(tr_imgs,  "train")
    log_split_diagnostics(val_imgs, "val")
    log.info("─" * 60)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    num_workers = CONFIG.get("num_workers", 0)
    pin_mem = device.type == "cuda"

    train_loader = DataLoader(
        SagitalDataset(tr_imgs, tr_masks),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        SagitalDataset(val_imgs, val_masks),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    # ── Modelo, optimizador y scheduler ──────────────────────────────────────
    model     = AttentionUNet(CONFIG["in_channels"], CONFIG["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # ReduceLROnPlateau: reduce el LR en factor 0.5 si el Dice no mejora
    # en 'patience' épocas consecutivas. Ayuda a salir de mesetas sin
    # necesidad de ajuste manual del LR.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",                          # maximizar Dice
        factor=CONFIG.get("lr_factor", 0.5),
        patience=CONFIG.get("lr_patience", 5),
        min_lr=CONFIG.get("lr_min", 1e-6),
    )

    # ── CSV de métricas ───────────────────────────────────────────────────────
    csv_path = init_metrics_csv(run_id)

    # ── Loop de entrenamiento ─────────────────────────────────────────────────
    best_dice = 0.0
    patience_counter = 0
    early_stop_patience = CONFIG.get("early_stop_patience", 0)  # 0 = desactivado

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        # Entrenamiento
        avg_loss = run_epoch(model, train_loader, optimizer, device, epoch)

        # Validación
        val_metrics, inf_batches = validate(model, val_loader, device)

        dice_val  = val_metrics["Dice"]
        hd95_val  = val_metrics["HD95"]
        oprec_val = val_metrics["Object_Precision"]
        hd95_str  = f"{hd95_val:.2f}px" if np.isfinite(hd95_val) else "inf"
        is_best   = dice_val > best_dice
        elapsed   = time.time() - t0

        # Obtener LR actual del optimizador
        current_lr = optimizer.param_groups[0]["lr"]
        prev_lr    = current_lr  # capturar antes del paso del scheduler
        scheduler.step(dice_val)
        new_lr = optimizer.param_groups[0]["lr"]

        log.info(
            "Época %02d/%02d | Loss: %.4f | Dice: %.4f | "
            "HD95: %s | ObjPrec: %.4f | LR: %.2e | %.1fs%s",
            epoch, CONFIG["epochs"],
            avg_loss, dice_val, hd95_str, oprec_val,
            current_lr, elapsed,
            " ★ BEST" if is_best else "",
        )

        if new_lr < prev_lr:
            log.info("  LR reducido: %.2e → %.2e", prev_lr, new_lr)

        # CSV
        append_epoch_metrics(
            csv_path, run_id, epoch,
            avg_loss, val_metrics, inf_batches, is_best, current_lr,
        )

        # Checkpoint
        if is_best:
            best_dice = dice_val
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_path"])
            log.info("  Checkpoint guardado → %s", CONFIG["model_path"])
        else:
            patience_counter += 1

        # Early stopping
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            log.info(
                "Early stopping: Dice no mejoró en %d épocas consecutivas.",
                early_stop_patience,
            )
            break

    log.info("═" * 60)
    log.info("  Entrenamiento finalizado")
    log.info("  Mejor Dice: %.4f", best_dice)
    log.info("  Métricas:   %s", csv_path)
    log.info("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
