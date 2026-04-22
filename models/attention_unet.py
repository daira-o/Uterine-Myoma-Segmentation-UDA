"""
attention_unet.py
Arquitectura Attention U-Net en PyTorch.
Compatible con CUDA 12.8 / RTX 5060 (Blackwell).

Métricas ampliadas según el framework Metrics Reloaded:
  - Dice Coefficient          (overlap semántico)
  - HD95                      (precisión de bordes, percentil 95)
  - Object-level Precision    (detección de instancias / miomas individuales)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.spatial.distance import directed_hausdorff


# ─────────────────────────────────────────────
#  MÉTRICAS — NIVEL SEMÁNTICO
# ─────────────────────────────────────────────

def dice_coef(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """
    Dice coefficient. Valores entre 0 y 1 (1 = predicción perfecta).
    Recibe tensores con probabilidades (ya pasados por sigmoid) o máscaras binarias.
    """
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    return (2.0 * intersection + eps) / (y_pred.sum() + y_true.sum() + eps)


def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_coef(y_pred, y_true)


def bce_dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    """
    Combinación de BCE + Dice. Compatible con AMP (float16).
    y_pred puede llegar en float16; se castea a float32 antes de calcular.
    NO modificar: se usa para gradientes de entrenamiento.
    """
    y_pred = y_pred.float()
    y_true = y_true.float()
    bce  = F.binary_cross_entropy_with_logits(y_pred, y_true)
    prob = torch.sigmoid(y_pred)
    dice = dice_loss(prob, y_true)
    return bce_weight * bce + (1 - bce_weight) * dice


# ─────────────────────────────────────────────
#  MÉTRICAS — NIVEL DE BORDES (HD95)
# ─────────────────────────────────────────────

def compute_hd95(pred_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    """
    Hausdorff Distance al percentil 95 entre dos máscaras binarias 2-D.

    Mide la distancia máxima (robusta) entre los bordes de la predicción
    y la máscara ground truth.  Valores bajos indican mayor precisión de contorno.

    Casos degenerados manejados explícitamente:
      - Si ninguna de las dos máscaras tiene píxeles positivos → devuelve 0.0
        (predicción vacía coincide con GT vacío).
      - Si solo una de las dos es vacía → devuelve np.inf
        (la predicción no tiene la información de bordes mínima).

    Args:
        pred_bin: np.ndarray bool/uint8 [H, W] — máscara predicha umbralizada.
        mask_bin: np.ndarray bool/uint8 [H, W] — máscara ground truth.

    Returns:
        float: HD95 en píxeles.
    """
    pred_pts = np.argwhere(pred_bin)
    gt_pts   = np.argwhere(mask_bin)

    # Caso 1: ambas vacías (verdadero negativo global)
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0

    # Caso 2: una sola vacía — predicción incorrecta de presencia/ausencia
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.inf

    # Distancias de Hausdorff dirigidas en ambas direcciones
    d_pred_to_gt = directed_hausdorff(pred_pts, gt_pts)[0]
    d_gt_to_pred = directed_hausdorff(gt_pts, pred_pts)[0]

    # HD95: percentil 95 sobre todas las distancias punto-a-conjunto más cercano
    from scipy.spatial import cKDTree

    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    dist_pred_to_gt, _ = tree_gt.query(pred_pts)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts)

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def batch_hd95(preds_bin: torch.Tensor, masks_bin: torch.Tensor) -> float:
    """
    Calcula el HD95 promedio sobre un batch entero.

    Args:
        preds_bin: Tensor [B, 1, H, W] binario (uint8 o bool).
        masks_bin: Tensor [B, 1, H, W] binario (uint8 o bool).

    Returns:
        float: media de HD95 por sample (se omiten los np.inf del promedio
               para que una sola predicción vacía no distorsione la época,
               pero se registran en los logs).
    """
    preds_np = preds_bin.squeeze(1).cpu().numpy().astype(bool)  # [B, H, W]
    masks_np = masks_bin.squeeze(1).cpu().numpy().astype(bool)  # [B, H, W]

    hd_values = []
    for p, m in zip(preds_np, masks_np):
        hd_values.append(compute_hd95(p, m))

    finite_vals = [v for v in hd_values if np.isfinite(v)]
    if not finite_vals:
        return np.inf
    return float(np.mean(finite_vals))


# ─────────────────────────────────────────────
#  MÉTRICAS — NIVEL DE INSTANCIA (Object Precision)
# ─────────────────────────────────────────────

def _get_connected_components(binary_mask: np.ndarray):
    """
    Retorna las componentes conexas (objetos) de una máscara binaria 2-D.
    Usa conectividad-4 por defecto (structure=None en scipy).

    Returns:
        labeled: np.ndarray con etiquetas únicas por objeto.
        n_objects: int, número de objetos encontrados.
    """
    structure = np.ones((3, 3), dtype=int)   # conectividad-8 para miomas irregulares
    labeled, n_objects = scipy_label(binary_mask, structure=structure)
    return labeled, n_objects


def compute_object_precision(
    pred_bin: np.ndarray,
    mask_bin: np.ndarray,
    iou_threshold: float = 0.1,
) -> float:
    """
    Precisión a nivel de instancia (Object-level Precision).

    Definición (Metrics Reloaded, Maier-Hein et al. 2022):
        Object Precision = TP_obj / (TP_obj + FP_obj)

    Un objeto predicho se considera TP si su intersección con CUALQUIER objeto
    ground truth supera `iou_threshold` (por área del objeto predicho, no IoU
    simétrico, para tolerar predicciones que cubren parcialmente una lesión real).

    Penaliza explícitamente los FP: predicciones de objetos que no corresponden
    a ningún mioma real.

    Args:
        pred_bin:      Máscara binaria [H, W] predicha.
        mask_bin:      Máscara binaria [H, W] ground truth.
        iou_threshold: Fracción mínima de superposición para contar como TP.

    Returns:
        float: Object Precision en [0, 1]. Retorna 1.0 si no hay predicciones
               (sin predicciones → sin falsos positivos).
    """
    pred_labeled, n_pred = _get_connected_components(pred_bin)
    gt_labeled,   n_gt   = _get_connected_components(mask_bin)

    # Sin predicciones → precisión perfecta (no hay FP)
    if n_pred == 0:
        return 1.0

    tp = 0
    for pred_id in range(1, n_pred + 1):
        pred_obj = (pred_labeled == pred_id)
        pred_area = pred_obj.sum()

        if pred_area == 0:
            continue

        # ¿Cuánto de este objeto predicho se superpone con cualquier GT?
        overlap = (pred_obj & (gt_labeled > 0)).sum()
        overlap_ratio = overlap / pred_area

        if overlap_ratio >= iou_threshold:
            tp += 1

    fp = n_pred - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return float(precision)


def batch_object_precision(
    preds_bin: torch.Tensor,
    masks_bin: torch.Tensor,
    iou_threshold: float = 0.1,
) -> float:
    """
    Object Precision promedio sobre un batch.

    Args:
        preds_bin: Tensor [B, 1, H, W] binario.
        masks_bin: Tensor [B, 1, H, W] binario.
        iou_threshold: umbral de superposición por objeto predicho.

    Returns:
        float: media de Object Precision en el batch.
    """
    preds_np = preds_bin.squeeze(1).cpu().numpy().astype(bool)
    masks_np = masks_bin.squeeze(1).cpu().numpy().astype(bool)

    precisions = [
        compute_object_precision(p, m, iou_threshold)
        for p, m in zip(preds_np, masks_np)
    ]
    return float(np.mean(precisions))


# ─────────────────────────────────────────────
#  MÉTRICAS — FUNCIÓN DE REPORTE UNIFICADA
# ─────────────────────────────────────────────

def compute_all_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    iou_threshold: float = 0.1,
) -> dict:
    """
    Calcula Dice, HD95 y Object Precision a partir de logits crudos.

    Esta función es para REPORTE solamente. No interviene en el cálculo
    de gradientes ni en `bce_dice_loss`.

    Args:
        logits:        Tensor [B, 1, H, W] — salida cruda del modelo (sin sigmoid).
        masks:         Tensor [B, 1, H, W] — máscaras ground truth binarias.
        threshold:     Umbral para binarizar las probabilidades (default 0.5).
        iou_threshold: Umbral de superposición para Object Precision (default 0.1).

    Returns:
        dict con claves:
            "Dice"             → float, promedio del batch.
            "HD95"             → float, promedio del batch (píxeles).
            "Object_Precision" → float, promedio del batch [0, 1].
    """
    with torch.no_grad():
        probs    = torch.sigmoid(logits.float())
        preds_bin = (probs >= threshold).float()

        # ── Dice (opera sobre tensores, rápido) ──────────────────────────
        dice = dice_coef(preds_bin, masks.float()).item()

        # ── HD95 y Object Precision (operan sobre numpy, por sample) ─────
        hd95      = batch_hd95(preds_bin, masks)
        obj_prec  = batch_object_precision(preds_bin, masks, iou_threshold)

    return {
        "Dice":             dice,
        "HD95":             hd95,
        "Object_Precision": obj_prec,
    }


# ─────────────────────────────────────────────
#  BLOQUES CONSTRUCTORES
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Doble convolución con BN y ReLU opcional Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, batch_norm: bool = True):
        super().__init__()
        layers = []
        for i in range(2):
            ch_in = in_ch if i == 0 else out_ch
            layers.append(nn.Conv2d(ch_in, out_ch, kernel_size=3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GatingSignal(nn.Module):
    """Señal de gating para el mecanismo de atención."""

    def __init__(self, in_ch: int, out_ch: int, batch_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.bn   = nn.BatchNorm2d(out_ch) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class AttentionBlock(nn.Module):
    """
    Attention gate: combina la feature map del encoder (x)
    con la señal de gating (g) del decoder.
    """

    def __init__(self, x_ch: int, g_ch: int, inter_ch: int):
        super().__init__()
        self.theta_x = nn.Conv2d(x_ch, inter_ch, kernel_size=2, stride=2, padding=0)
        self.phi_g   = nn.Conv2d(g_ch, inter_ch, kernel_size=1, padding=0)
        self.psi     = nn.Conv2d(inter_ch, 1, kernel_size=1, padding=0)
        self.out_conv = nn.Conv2d(x_ch, x_ch, kernel_size=1, padding=0)
        self.bn       = nn.BatchNorm2d(x_ch)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: feature map del encoder  [B, x_ch, H, W]
        # g: gating signal del decoder [B, g_ch, H/2, W/2]
        theta_x = self.theta_x(x)                                  # [B, inter, H/2, W/2]
        phi_g   = F.interpolate(self.phi_g(g), size=theta_x.shape[2:], mode="bilinear", align_corners=True)
        attn    = torch.sigmoid(self.psi(F.relu(theta_x + phi_g))) # [B, 1, H/2, W/2]
        attn    = F.interpolate(attn, size=x.shape[2:], mode="bilinear", align_corners=True)
        attn    = attn.expand_as(x)
        y       = self.out_conv(x * attn)
        return self.bn(y)


# ─────────────────────────────────────────────
#  RED COMPLETA
# ─────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net para segmentación de imágenes médicas.

    Args:
        in_channels:  canales de entrada (1 = escala de grises, 3 = RGB)
        num_classes:  canales de salida (1 = binario, N = multiclase)
        base_filters: filtros base (se duplican en cada nivel del encoder)
        dropout_rate: tasa de dropout en los ConvBlocks
        batch_norm:   usar BatchNorm
    """

    def __init__(
        self,
        in_channels:  int   = 1,
        num_classes:  int   = 1,
        base_filters: int   = 64,
        dropout_rate: float = 0.0,
        batch_norm:   bool  = True,
    ):
        super().__init__()
        f = base_filters

        # ── Encoder ──────────────────────────────
        self.enc1 = ConvBlock(in_channels, f,    dropout_rate, batch_norm)
        self.enc2 = ConvBlock(f,           f*2,  dropout_rate, batch_norm)
        self.enc3 = ConvBlock(f*2,         f*4,  dropout_rate, batch_norm)
        self.enc4 = ConvBlock(f*4,         f*8,  dropout_rate, batch_norm)

        # ── Bottleneck ───────────────────────────
        self.bottleneck = ConvBlock(f*8, f*16, dropout_rate, batch_norm)

        # ── Gating signals ───────────────────────
        self.gate4 = GatingSignal(f*16, f*8,  batch_norm)
        self.gate3 = GatingSignal(f*8,  f*4,  batch_norm)
        self.gate2 = GatingSignal(f*4,  f*2,  batch_norm)
        self.gate1 = GatingSignal(f*2,  f,    batch_norm)

        # ── Attention blocks ─────────────────────
        self.att4 = AttentionBlock(f*8,  f*8,  f*8)
        self.att3 = AttentionBlock(f*4,  f*4,  f*4)
        self.att2 = AttentionBlock(f*2,  f*2,  f*2)
        self.att1 = AttentionBlock(f,    f,    f)

        # ── Decoder ──────────────────────────────
        self.up4 = nn.ConvTranspose2d(f*16, f*8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f*16, f*8,  dropout_rate, batch_norm)

        self.up3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f*8,  f*4,  dropout_rate, batch_norm)

        self.up2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f*4,  f*2,  dropout_rate, batch_norm)

        self.up1 = nn.ConvTranspose2d(f*2, f,   kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f*2,  f,    dropout_rate, batch_norm)

        # ── Salida ───────────────────────────────
        self.output_conv = nn.Conv2d(f, num_classes, kernel_size=1)
        self.output_act  = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder con attention gates
        g4  = self.gate4(b)
        a4  = self.att4(e4, g4)
        d4  = self.dec4(torch.cat([self.up4(b), a4], dim=1))

        g3  = self.gate3(d4)
        a3  = self.att3(e3, g3)
        d3  = self.dec3(torch.cat([self.up3(d4), a3], dim=1))

        g2  = self.gate2(d3)
        a2  = self.att2(e2, g2)
        d2  = self.dec2(torch.cat([self.up2(d3), a2], dim=1))

        g1  = self.gate1(d2)
        a1  = self.att1(e1, g1)
        d1  = self.dec1(torch.cat([self.up1(d2), a1], dim=1))

        # Salida — devuelve logits crudos (sin sigmoid)
        # El sigmoid se aplica en bce_dice_loss durante el entrenamiento
        # y explícitamente en compute_all_metrics() para inferencia/reporte
        return self.output_conv(d1)
