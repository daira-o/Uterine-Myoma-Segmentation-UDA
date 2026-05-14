"""
scripts/train.py
─────────────────────────────────────────────────────────────────────────────
Entrenamiento del modelo Attention U-Net para segmentación de miomas.

Uso:
    python scripts/train.py

Requisitos:
    pip install python-dotenv torch numpy scikit-learn scipy scikit-image
"""

import os
import re
import sys
import csv
import glob
import time
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

# ── Importar config y modelo desde la raíz del proyecto ──────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CONFIG
from models.attention_unet import AttentionUNet, bce_dice_loss, compute_all_metrics


def list_processed_arrays(base_path: str, subdir: str):
    """
    Lista arrays procesados dando prioridad a la estructura por vistas.

    Si existen subcarpetas como 1_Anchor/ o 2_Lateral/, se ignoran los .npy
    planos legacy para no mezclar el dataset antiguo de muchos cortes con el
    dataset actual de 3 vistas por volumen.
    """
    nested = sorted(glob.glob(
        os.path.join(base_path, subdir, "*", "*.npy")))
    if nested:
        return nested
    return sorted(glob.glob(os.path.join(base_path, subdir, "*.npy")))


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SagitalDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = np.load(self.img_paths[idx]).astype("float32")
        mask = np.load(self.mask_paths[idx]).astype("float32")
        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0))


# ─────────────────────────────────────────────────────────────────────────────
#  MÉTRICAS CSV
# ─────────────────────────────────────────────────────────────────────────────

def init_metrics_csv(run_id: str) -> str:
    """
    Crea (o abre en modo append) el CSV de métricas para este run.
    Devuelve el path al archivo.

    Columnas:
        run_id, epoch, avg_loss, dice, hd95, obj_precision,
        hd95_inf_batches, timestamp
    """
    os.makedirs(CONFIG["logs_path"], exist_ok=True)
    csv_path = os.path.join(CONFIG["logs_path"], "training_metrics.csv")

    # Escribir cabecera solo si el archivo no existe
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "epoch", "avg_loss",
                "dice", "hd95", "obj_precision",
                "hd95_inf_batches", "is_best", "timestamp"
            ])
        print(f" Métricas → {csv_path} (archivo nuevo)")
    else:
        print(f" Métricas → {csv_path} (append a run anterior)")

    return csv_path


def append_epoch_metrics(csv_path: str, run_id: str, epoch: int,
                         avg_loss: float, metrics: dict,
                         hd95_inf_batches: int, is_best: bool):
    """Añade una fila al CSV con las métricas de la época."""
    hd95_val = metrics["HD95"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id,
            epoch,
            f"{avg_loss:.6f}",
            f"{metrics['Dice']:.6f}",
            f"{hd95_val:.4f}" if np.isfinite(hd95_val) else "inf",
            f"{metrics['Object_Precision']:.6f}",
            hd95_inf_batches,
            int(is_best),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def validate(model, val_loader, device) -> tuple[dict, int]:
    """
    Evalúa el modelo y devuelve (dict de métricas, cantidad de batches con HD95=inf).
    """
    model.eval()
    batch_dices, batch_hd95s, batch_oprecs = [], [], []
    inf_count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits  = model(images)
            metrics = compute_all_metrics(
                logits, masks,
                threshold=CONFIG["threshold"],
                iou_threshold=CONFIG["iou_threshold"],
            )
            batch_dices.append(metrics["Dice"])
            if np.isfinite(metrics["HD95"]):
                batch_hd95s.append(metrics["HD95"])
            else:
                inf_count += 1
            batch_oprecs.append(metrics["Object_Precision"])

    if inf_count > 0:
        print(f"   ⚠  HD95=inf en {inf_count} batch(es) "
              f"(predicción vacía frente a GT positivo). Excluidos del promedio.")

    return {
        "Dice":             float(np.mean(batch_dices))  if batch_dices  else 0.0,
        "HD95":             float(np.mean(batch_hd95s))  if batch_hd95s  else np.inf,
        "Object_Precision": float(np.mean(batch_oprecs)) if batch_oprecs else 0.0,
    }, inf_count


# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES DE SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def get_patient_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(.+)_sag_\d+$", stem)
    return m.group(1) if m else stem


def build_patient_split(imgs, masks):
    """
    Separa imgs/masks en train/val a nivel de paciente.
    Ningún paciente aparece en ambos sets.
    Aplica max_patients y max_slices_per_patient desde CONFIG.
    """
    groups         = np.array([get_patient_id(p) for p in imgs])
    unique_patients = np.unique(groups)

    # ── Limitar número de pacientes ───────────────────────────────────────
    max_p = CONFIG.get("max_patients")
    if max_p and len(unique_patients) > max_p:
        rng = np.random.default_rng(CONFIG["random_state"])
        unique_patients = rng.choice(unique_patients, size=max_p, replace=False)
        keep = np.isin(groups, unique_patients)
        imgs   = [p for p, k in zip(imgs,  keep) if k]
        masks  = [p for p, k in zip(masks, keep) if k]
        groups = groups[keep]

    # ── Diagnóstico de distribución ───────────────────────────────────────
    slices_pp = {pid: int((groups == pid).sum()) for pid in unique_patients}
    counts    = list(slices_pp.values())
    print(f"\n{'─'*55}")
    print(f" Dataset: {len(imgs)} slices | {len(unique_patients)} pacientes")
    print(f" Slices/paciente: min={min(counts)}  max={max(counts)}  "
          f"media={np.mean(counts):.1f}")

    dominantes = [pid for pid, c in slices_pp.items() if c / len(imgs) > 0.10]
    if dominantes:
        print(f" ⚠  Pacientes que dominan >10% del dataset: {dominantes}")
        print(f"    Considera bajar max_slices_per_patient.")
    else:
        print(f" ✓ Distribución balanceada.")

    # ── Balance por paciente ──────────────────────────────────────────────
    max_spp = CONFIG.get("max_slices_per_patient")
    if max_spp:
        rng  = np.random.default_rng(CONFIG["random_state"])
        keep = []
        for pid in unique_patients:
            idx_pid = np.where(groups == pid)[0]
            if len(idx_pid) > max_spp:
                idx_pid = rng.choice(idx_pid, size=max_spp, replace=False)
            keep.extend(idx_pid.tolist())
        keep   = sorted(keep)
        imgs   = [imgs[i]  for i in keep]
        masks  = [masks[i] for i in keep]
        groups = groups[keep]

    # ── Split por paciente ────────────────────────────────────────────────
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=CONFIG["val_size"],
        random_state=CONFIG["random_state"],
    )
    train_idx, val_idx = next(gss.split(imgs, groups=groups))

    tr_i  = [imgs[i]  for i in train_idx]
    tr_m  = [masks[i] for i in train_idx]
    val_i = [imgs[i]  for i in val_idx]
    val_m = [masks[i] for i in val_idx]

    n_tr_p = len(np.unique([get_patient_id(p) for p in tr_i]))
    n_val_p = len(np.unique([get_patient_id(p) for p in val_i]))

    print(f" Train: {len(tr_i):>5} slices | {n_tr_p} pacientes")
    print(f" Val:   {len(val_i):>5} slices | {n_val_p} pacientes")
    print(f" ✓ Separación estricta: cero pacientes compartidos.")
    print(f"{'─'*55}\n")

    return tr_i, tr_m, val_i, val_m


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train():
    # ── Dispositivo ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'═'*55}")
    print(f"  MiomaVision — Entrenamiento")
    print(f"  Run ID:      {run_id}")
    print(f"  Dispositivo: {device}")
    print(f"  Épocas:      {CONFIG['epochs']}")
    print(f"  LR:          {CONFIG['lr']}")
    print(f"{'═'*55}")

    # ── Datos ────────────────────────────────────────────────────────────
    all_imgs = list_processed_arrays(CONFIG["base_path"], "images_npy")
    all_masks = list_processed_arrays(CONFIG["base_path"], "masks_npy")

    if not all_imgs:
        raise FileNotFoundError(
            f"No se encontraron .npy en {CONFIG['base_path']}\n"
            "Verificá DATA_PATH en tu .env")

    tr_i, tr_m, val_i, val_m = build_patient_split(all_imgs, all_masks)

    train_loader = DataLoader(
        SagitalDataset(tr_i, tr_m),
        batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(
        SagitalDataset(val_i, val_m),
        batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"))

    # ── Modelo ───────────────────────────────────────────────────────────
    model     = AttentionUNet(CONFIG["in_channels"], CONFIG["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # ── CSV de métricas ──────────────────────────────────────────────────
    csv_path  = init_metrics_csv(run_id)

    # ── Loop de entrenamiento ─────────────────────────────────────────────
    best_dice = 0.0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            loss = bce_dice_loss(model(images), targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 50 == 0:
                print(f" Ep {epoch+1:02d} | Step {i:>4}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

        # ── Validación ───────────────────────────────────────────────────
        val_metrics, inf_batches = validate(model, val_loader, device)

        dice_val  = val_metrics["Dice"]
        hd95_val  = val_metrics["HD95"]
        oprec_val = val_metrics["Object_Precision"]
        avg_loss  = epoch_loss / len(train_loader)
        hd95_str  = f"{hd95_val:.2f}px" if np.isfinite(hd95_val) else "inf"
        is_best   = dice_val > best_dice

        print(
            f">> Fin Época {epoch+1:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Dice: {dice_val:.4f} | "
            f"HD95: {hd95_str} | "
            f"ObjPrec: {oprec_val:.4f}"
            + (" ★" if is_best else "")
        )

        # ── Guardar CSV ──────────────────────────────────────────────────
        append_epoch_metrics(
            csv_path, run_id, epoch + 1,
            avg_loss, val_metrics, inf_batches, is_best)

        # ── Guardar checkpoint ───────────────────────────────────────────
        if is_best:
            best_dice = dice_val
            torch.save(model.state_dict(), CONFIG["model_path"])
            print(f"   ✓ Checkpoint guardado → {CONFIG['model_path']}")

    print(f"\n{'═'*55}")
    print(f"  Entrenamiento finalizado")
    print(f"  Mejor Dice: {best_dice:.4f}")
    print(f"  Métricas:   {csv_path}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    train()
