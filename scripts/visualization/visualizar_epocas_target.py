"""
Visualizador Streamlit para comparar checkpoints/epocas del entrenamiento target.

Uso:
    streamlit run scripts/visualization/visualizar_epocas_target.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import CONFIG
from models.attention_unet import AttentionUNet
from scripts.inference.infer_us_production import extract_segmenter_state_dict


DEFAULT_US_PATH = CONFIG.get("us_ready_path") or str(ROOT / "data_ready_US")
DEFAULT_CHECKPOINT_ROOT = str(Path(CONFIG.get("logs_path", ROOT / "logs")) / "checkpoints")


st.set_page_config(
    page_title="Comparador de Epocas Target",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stSidebar"] { border-right: 1px solid #263331; }
.small-note { color: #708481; font-size: .82rem; }
</style>
""",
    unsafe_allow_html=True,
)


def project_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"epoch[_\-]?(\d+)", path.stem, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def checkpoint_label(path: Path, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    epoch = metadata.get("epoch") or checkpoint_epoch(path)
    dice = None
    metrics = metadata.get("metrics")
    if isinstance(metrics, dict):
        dice = metrics.get("Dice") or metrics.get("dice") or metrics.get("val_dice")
    pieces = [path.stem]
    if epoch is not None:
        pieces.append(f"ep {int(epoch):03d}")
    if dice is not None:
        pieces.append(f"Dice {float(dice):.4f}")
    return " | ".join(pieces)


def list_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []
    paths = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: (checkpoint_epoch(p) is None, checkpoint_epoch(p) or 999999, p.stat().st_mtime, p.name))
    return [path for path in paths if path.is_file()]


def list_run_dirs(checkpoint_root: Path) -> list[Path]:
    if not checkpoint_root.exists():
        return []
    dirs = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and any(path.glob("*.pth"))
    ]
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def run_label(path: Path) -> str:
    metrics = path / "metrics.csv"
    marker = "metrics" if metrics.exists() else "sin metrics"
    return f"{path.name} ({marker})"


def list_us_images(base_path: Path, split: str) -> list[Path]:
    if split == "all":
        patterns = [
            str(base_path / "*" / "images" / "*.npy"),
            str(base_path / "images" / "*.npy"),
        ]
    else:
        patterns = [str(base_path / split / "images" / "*.npy")]
    return sorted({Path(path) for pattern in patterns for path in glob.glob(pattern)})


def bbox_json_path(image_path: Path) -> Path:
    if image_path.parent.name == "images":
        return image_path.parent.parent / "bboxes" / f"{image_path.stem}.json"
    return image_path.parent / "bboxes" / f"{image_path.stem}.json"


def load_bbox_mask(image_path: Path, size: int = 256) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    json_path = bbox_json_path(image_path)
    if not json_path.exists():
        return mask
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    for row in payload.get("bbox_256", [])[:1]:
        xmin = int(np.floor(float(row["xmin"])))
        ymin = int(np.floor(float(row["ymin"])))
        xmax = int(np.ceil(float(row["xmax"])))
        ymax = int(np.ceil(float(row["ymax"])))
        xmin, xmax = np.clip([xmin, xmax], 0, size)
        ymin, ymax = np.clip([ymin, ymax], 0, size)
        if xmax > xmin and ymax > ymin:
            mask[ymin:ymax, xmin:xmax] = True
    return mask


def load_us_image(path: Path) -> np.ndarray:
    image = np.load(path).astype(np.float32)
    if image.ndim == 3:
        image = image[0] if image.shape[0] == 1 else image[..., 0]
    if image.shape != (256, 256):
        raise ValueError(f"Se esperaba US 256x256, recibido {image.shape}")
    if image.max() > 1.0 or image.min() < 0.0:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return np.clip(image, 0.0, 1.0)


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    return checkpoint if isinstance(checkpoint, dict) else {}


@st.cache_resource(show_spinner="Cargando checkpoint...")
def load_model(checkpoint_path: str, device_name: str) -> tuple[AttentionUNet, dict[str, Any]]:
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_segmenter_state_dict(checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = AttentionUNet(
        in_channels=int(config.get("in_channels", CONFIG.get("in_channels", 1))),
        num_classes=int(config.get("num_classes", CONFIG.get("num_classes", 1))),
        base_filters=int(config.get("base_filters", CONFIG.get("base_filters", 64))),
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model, metadata


@torch.no_grad()
def predict(model: AttentionUNet, image: np.ndarray, device_name: str) -> np.ndarray:
    device = torch.device(device_name)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits[0, 0].float()).cpu().numpy()


def compute_metrics(prob: np.ndarray, bbox: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= threshold
    pred_area = float(pred.sum())
    bbox_area = float(bbox.sum())
    intersection = float((pred & bbox).sum())
    union = float((pred | bbox).sum())
    inside_probs = prob[bbox]
    return {
        "pred_area": pred_area,
        "area_ratio": pred_area / max(bbox_area, 1.0),
        "bbox_in": intersection / max(pred_area, 1.0),
        "bbox_iou": intersection / max(union, 1.0),
        "inside_mean": float(inside_probs.mean()) if inside_probs.size else 0.0,
        "inside_max": float(inside_probs.max()) if inside_probs.size else 0.0,
    }


def render_panel(image: np.ndarray, prob: np.ndarray, bbox: np.ndarray, threshold: float, title: str):
    pred = prob >= threshold
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    if bbox.any():
        axes[0].contour(bbox.astype(np.float32), levels=[0.5], colors=["#00c853"], linewidths=1.5)
    axes[0].set_title("US + bbox")

    axes[1].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Probabilidad")

    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(np.ma.masked_where(~pred, pred), cmap="Reds", alpha=0.42, vmin=0, vmax=1)
    if bbox.any():
        axes[2].contour(bbox.astype(np.float32), levels=[0.5], colors=["#00c853"], linewidths=1.3)
    axes[2].set_title("Overlay binario")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    return fig


st.title("Comparador de epocas target")
st.markdown(
    '<div class="small-note">Compara checkpoints del entrenamiento actual usando solo el segmentador; GRL y discriminador se ignoran.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    checkpoint_root = project_path(st.text_input("Raiz checkpoints", DEFAULT_CHECKPOINT_ROOT))
    run_dirs = list_run_dirs(checkpoint_root)
    if run_dirs:
        selected_run = st.selectbox(
            "Corrida",
            options=[str(path) for path in run_dirs],
            format_func=lambda value: run_label(Path(value)),
            index=0,
        )
        checkpoint_dir = Path(selected_run)
        st.caption(f"Modelo actual: {checkpoint_dir}")
    else:
        checkpoint_dir = checkpoint_root
    us_dir = project_path(st.text_input("Dataset US", DEFAULT_US_PATH))
    split = st.selectbox("Split US", ["train", "val", "test", "all"], index=1)
    threshold = st.slider("Threshold", 0.05, 0.95, float(CONFIG.get("threshold", 0.5)), 0.05)
    device_name = st.selectbox(
        "Device",
        ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
        index=0,
    )

checkpoints = list_checkpoints(checkpoint_dir)
if not checkpoints:
    st.warning(f"No encontre checkpoints .pth en {checkpoint_dir}")
    st.stop()

metadata_by_path = {str(path): load_checkpoint_metadata(path) for path in checkpoints}
labels_by_path = {str(path): checkpoint_label(path, metadata_by_path[str(path)]) for path in checkpoints}
default_selection = [str(path) for path in checkpoints[-min(4, len(checkpoints)):]]
selected = st.multiselect(
    "Epocas/checkpoints a comparar",
    options=[str(path) for path in checkpoints],
    default=default_selection,
    format_func=lambda value: labels_by_path[value],
)
if not selected:
    st.info("Selecciona al menos un checkpoint.")
    st.stop()

images = list_us_images(us_dir, split)
if not images:
    st.warning(f"No encontre imagenes .npy para split={split} en {us_dir}")
    st.stop()

image_path = st.selectbox(
    "Imagen US",
    options=images,
    format_func=lambda path: f"{path.parent.parent.name}/{path.name}" if path.parent.name == "images" else path.name,
)

image = load_us_image(image_path)
bbox = load_bbox_mask(image_path)

metric_rows = []
cols = st.columns(min(3, len(selected)))
for idx, checkpoint_path in enumerate(selected):
    model, metadata = load_model(checkpoint_path, device_name)
    prob = predict(model, image, device_name)
    metrics = compute_metrics(prob, bbox, threshold)
    label = checkpoint_label(Path(checkpoint_path), metadata)
    metric_rows.append({"checkpoint": label, **metrics})

    with cols[idx % len(cols)]:
        st.pyplot(render_panel(image, prob, bbox, threshold, label), clear_figure=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("bbox_in", f"{metrics['bbox_in']:.3f}")
        m2.metric("IoU bbox", f"{metrics['bbox_iou']:.3f}")
        m3.metric("area/bbox", f"{metrics['area_ratio']:.2f}")

st.subheader("Metricas comparativas")
st.dataframe(metric_rows, use_container_width=True)
