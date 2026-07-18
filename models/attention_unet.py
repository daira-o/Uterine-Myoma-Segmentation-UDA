"""
Attention U-Net implementation in PyTorch.

The module also provides reporting metrics aligned with the Metrics Reloaded
framework:
  - Dice coefficient for semantic overlap.
  - HD95 for robust boundary accuracy at the 95th percentile.
  - Object-level precision for instance-level myoma detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label


def dice_coef(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """
    Compute the Dice coefficient in [0, 1], where 1 means perfect overlap.

    `y_pred` may contain probabilities after sigmoid or an already-binary mask.
    """
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    return (2.0 * intersection + eps) / (y_pred.sum() + y_true.sum() + eps)


def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_coef(y_pred, y_true)


def bce_dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    """
    Combine BCE-with-logits and Dice loss.

    The logits may arrive as float16 under AMP, so both tensors are promoted to
    float32 before computing the training loss. Keep this function logits-based:
    it is used directly for training gradients.
    """
    y_pred = y_pred.float()
    y_true = y_true.float()
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
    prob = torch.sigmoid(y_pred)
    dice = dice_loss(prob, y_true)
    return bce_weight * bce + (1 - bce_weight) * dice


def compute_hd95(pred_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    """
    Compute the 95th-percentile Hausdorff distance between two 2-D binary masks.

    HD95 measures the robust maximum distance between prediction and ground-truth
    boundaries. Lower values indicate better contour agreement.

    Degenerate cases are handled explicitly:
      - If both masks are empty, return 0.0 because empty prediction matches
        empty ground truth.
      - If only one mask is empty, return np.inf because the boundary evidence
        needed for a finite distance is missing.

    Args:
        pred_bin: Thresholded predicted mask as a bool/uint8 array [H, W].
        mask_bin: Ground-truth binary mask as a bool/uint8 array [H, W].

    Returns:
        HD95 in pixels.
    """
    pred_pts = np.argwhere(pred_bin)
    gt_pts = np.argwhere(mask_bin)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.inf

    from scipy.spatial import cKDTree

    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    dist_pred_to_gt, _ = tree_gt.query(pred_pts)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts)

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def batch_hd95(preds_bin: torch.Tensor, masks_bin: torch.Tensor) -> float:
    """
    Average HD95 over a batch.

    Infinite values are excluded from the mean so that one empty prediction does
    not dominate the epoch-level value. Callers can still count and log those
    cases separately.
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


def _get_connected_components(binary_mask: np.ndarray):
    """
    Return connected components for a 2-D binary mask.

    Eight-connectivity is used because myoma contours can be irregular and
    diagonally connected.
    """
    structure = np.ones((3, 3), dtype=int)
    labeled, n_objects = scipy_label(binary_mask, structure=structure)
    return labeled, n_objects


def compute_object_precision(
    pred_bin: np.ndarray,
    mask_bin: np.ndarray,
    iou_threshold: float = 0.1,
) -> float:
    """
    Compute object-level precision.

    Following the Metrics Reloaded framing, object precision is:
        TP_obj / (TP_obj + FP_obj)

    A predicted object is counted as true positive if its overlap with any
    ground-truth object exceeds `iou_threshold` as a fraction of the predicted
    object's area. This asymmetric criterion tolerates predictions that cover
    only part of a real lesion while still penalizing false-positive objects.

    Returns 1.0 when there are no predicted objects, because no false positives
    were produced.
    """
    pred_labeled, n_pred = _get_connected_components(pred_bin)
    gt_labeled, n_gt = _get_connected_components(mask_bin)

    if n_pred == 0:
        return 1.0

    tp = 0
    for pred_id in range(1, n_pred + 1):
        pred_obj = pred_labeled == pred_id
        pred_area = pred_obj.sum()

        if pred_area == 0:
            continue

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
    """Average object-level precision over a batch."""
    preds_np = preds_bin.squeeze(1).cpu().numpy().astype(bool)
    masks_np = masks_bin.squeeze(1).cpu().numpy().astype(bool)

    precisions = [
        compute_object_precision(p, m, iou_threshold)
        for p, m in zip(preds_np, masks_np)
    ]
    return float(np.mean(precisions))


def compute_all_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    iou_threshold: float = 0.1,
) -> dict:
    """
    Compute Dice, HD95, and object-level precision from raw logits.

    This function is for reporting only. It does not participate in gradient
    computation or in `bce_dice_loss`.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.float())
        preds_bin = (probs >= threshold).float()

        dice = dice_coef(preds_bin, masks.float()).item()
        hd95 = batch_hd95(preds_bin, masks)
        obj_prec = batch_object_precision(preds_bin, masks, iou_threshold)

    return {
        "Dice": dice,
        "HD95": hd95,
        "Object_Precision": obj_prec,
    }


class ConvBlock(nn.Module):
    """Two convolution layers with optional batch normalization and dropout."""

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
    """Decoder gating signal used by the attention blocks."""

    def __init__(self, in_ch: int, out_ch: int, batch_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out_ch) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class AttentionBlock(nn.Module):
    """Attention gate that combines an encoder feature map with decoder context."""

    def __init__(self, x_ch: int, g_ch: int, inter_ch: int):
        super().__init__()
        self.theta_x = nn.Conv2d(x_ch, inter_ch, kernel_size=2, stride=2, padding=0)
        self.phi_g = nn.Conv2d(g_ch, inter_ch, kernel_size=1, padding=0)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, padding=0)
        self.out_conv = nn.Conv2d(x_ch, x_ch, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(x_ch)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x is the encoder feature map; g is the decoder gating signal.
        theta_x = self.theta_x(x)
        phi_g = F.interpolate(self.phi_g(g), size=theta_x.shape[2:], mode="bilinear", align_corners=True)
        attn = torch.sigmoid(self.psi(F.relu(theta_x + phi_g)))
        attn = F.interpolate(attn, size=x.shape[2:], mode="bilinear", align_corners=True)
        attn = attn.expand_as(x)
        y = self.out_conv(x * attn)
        return self.bn(y)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for 2-D medical image segmentation.

    Args:
        in_channels: Number of input channels. Use 1 for grayscale images.
        num_classes: Number of output channels. Use 1 for binary segmentation.
        base_filters: Base channel count, doubled at each encoder level.
        dropout_rate: Dropout rate inside convolution blocks.
        batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 64,
        dropout_rate: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock(in_channels, f, dropout_rate, batch_norm)
        self.enc2 = ConvBlock(f, f * 2, dropout_rate, batch_norm)
        self.enc3 = ConvBlock(f * 2, f * 4, dropout_rate, batch_norm)
        self.enc4 = ConvBlock(f * 4, f * 8, dropout_rate, batch_norm)

        self.bottleneck = ConvBlock(f * 8, f * 16, dropout_rate, batch_norm)

        self.gate4 = GatingSignal(f * 16, f * 8, batch_norm)
        self.gate3 = GatingSignal(f * 8, f * 4, batch_norm)
        self.gate2 = GatingSignal(f * 4, f * 2, batch_norm)
        self.gate1 = GatingSignal(f * 2, f, batch_norm)

        self.att4 = AttentionBlock(f * 8, f * 8, f * 8)
        self.att3 = AttentionBlock(f * 4, f * 4, f * 4)
        self.att2 = AttentionBlock(f * 2, f * 2, f * 2)
        self.att1 = AttentionBlock(f, f, f)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8, dropout_rate, batch_norm)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4, dropout_rate, batch_norm)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2, dropout_rate, batch_norm)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f, dropout_rate, batch_norm)

        self.output_conv = nn.Conv2d(f, num_classes, kernel_size=1)
        self.output_act = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottleneck(F.max_pool2d(e4, 2))

        g4 = self.gate4(b)
        a4 = self.att4(e4, g4)
        d4 = self.dec4(torch.cat([self.up4(b), a4], dim=1))

        g3 = self.gate3(d4)
        a3 = self.att3(e3, g3)
        d3 = self.dec3(torch.cat([self.up3(d4), a3], dim=1))

        g2 = self.gate2(d3)
        a2 = self.att2(e2, g2)
        d2 = self.dec2(torch.cat([self.up2(d3), a2], dim=1))

        g1 = self.gate1(d2)
        a1 = self.att1(e1, g1)
        d1 = self.dec1(torch.cat([self.up1(d2), a1], dim=1))

        # Return raw logits. Training loss and reporting metrics apply sigmoid
        # explicitly so numerical behavior stays stable and easy to audit.
        return self.output_conv(d1)
