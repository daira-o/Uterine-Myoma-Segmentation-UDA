"""
scripts/visualization/visualizar_us_modelo.py
Visualizador Streamlit para Fase 3: inferencia de miomas en ultrasonido.

Carga un checkpoint target, descarta GRL/DomainDiscriminator si existen y usa
solo el segmentador adaptado para predecir mascaras sobre US 256x256.

Uso:
    streamlit run scripts/visualization/visualizar_us_modelo.py
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, label as scipy_label
from skimage import measure
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from config import CONFIG
except Exception:
    CONFIG = {
        "logs_path": "logs",
        "threshold": 0.5,
        "in_channels": 1,
        "num_classes": 1,
        "base_filters": 64,
    }

from models.attention_unet import AttentionUNet
from scripts.inference.infer_us_production import extract_segmenter_state_dict
from scripts.inference.postprocessing import (
    BBox,
    postprocess_prediction,
    save_postprocessing_debug_overlay,
)


DEFAULT_US_PATH = CONFIG.get("us_ready_path") or os.path.join(str(ROOT), "data_ready_US")


def project_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def checkpoint_search_roots() -> list[Path]:
    logs_path = project_path(CONFIG.get("logs_path", "logs"))
    return [
        logs_path / "checkpoints",
    ]


def best_checkpoints_under(root: Path) -> list[Path]:
    candidates: dict[str, Path] = {}
    if not root.exists():
        return []
    for pattern in ("**/best_model.pth", "**/best_model_dann.pth"):
        for path in root.glob(pattern):
            if path.is_file():
                candidates[str(path.resolve())] = path
    for path in (root / "best_model.pth", root / "best_model_dann.pth"):
        if path.is_file():
            candidates[str(path.resolve())] = path
    return list(candidates.values())


def latest_best_checkpoint() -> str:
    # Usa una sola raiz canonica: PROJECT_ROOT/logs.
    for root in checkpoint_search_roots():
        candidates = best_checkpoints_under(root)
        if candidates:
            return str(max(candidates, key=lambda path: path.stat().st_mtime))
    else:
        return str(ROOT / "logs" / "checkpoints" / "best_model.pth")


DEFAULT_CHECKPOINT = latest_best_checkpoint()

CMAP_US = LinearSegmentedColormap.from_list(
    "us_gray",
    ["#050505", "#1c1c1c", "#5d5d5d", "#b8b8b8", "#ffffff"],
    N=256,
)


st.set_page_config(
    page_title="Visualizador US Target",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@500;700;800&display=swap');
:root {
    --bg: #070909; --surface: #101414; --border: #22302f;
    --accent: #00d1b2; --accent2: #ff4d7d; --text: #dcebea;
    --muted: #708481; --warn: #f2a93b;
}
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }
.main-title {
    font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800;
    color: var(--text); margin-bottom: 0;
}
.subtitle { color: var(--muted); font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; }
.section-header {
    font-family: 'Syne', sans-serif; font-size: .72rem; font-weight: 700;
    letter-spacing: .16em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 12px;
}
.metric-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 16px; text-align: center;
}
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: var(--accent); line-height: 1; }
.metric-label { font-size: .68rem; color: var(--muted); letter-spacing: .12em; text-transform: uppercase; margin-top: 5px; }
.info-box { background: rgba(0,209,178,.08); border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; padding: 10px 14px; font-size: .78rem; }
.warn-box { background: rgba(242,169,59,.10); border-left: 3px solid var(--warn); border-radius: 0 6px 6px 0; padding: 10px 14px; font-size: .78rem; color: var(--warn); }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
</style>
""",
    unsafe_allow_html=True,
)


def list_us_images(base_path: str, split: str) -> list[str]:
    if split == "all":
        patterns = [
            os.path.join(base_path, "*", "images", "*.npy"),
            os.path.join(base_path, "images", "*.npy"),
        ]
    else:
        patterns = [os.path.join(base_path, split, "images", "*.npy")]
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    return paths


def load_us_npy(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)
    if arr.shape != (256, 256):
        raise ValueError(f"US debe ser 256x256. Shape recibido: {arr.shape}")
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.clip(arr, 0.0, 1.0)


def bbox_json_path(image_path: str) -> Path:
    path = Path(image_path)
    if path.parent.name == "images":
        return path.parent.parent / "bboxes" / f"{path.stem}.json"
    return path.parent / "bboxes" / f"{path.stem}.json"


def load_bboxes(image_path: str) -> list[BBox]:
    json_path = bbox_json_path(image_path)
    if not json_path.exists():
        return []

    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    boxes: list[BBox] = []
    for row in payload.get("bbox_256", []):
        try:
            box = BBox(
                xmin=float(row["xmin"]),
                ymin=float(row["ymin"]),
                xmax=float(row["xmax"]),
                ymax=float(row["ymax"]),
            ).clipped()
            if box is not None:
                boxes.append(box)
        except (KeyError, TypeError, ValueError):
            continue
    return boxes[:1]


@st.cache_resource(show_spinner="Cargando segmentador adaptado...")
def load_segmenter(checkpoint_path: str, device_name: str) -> AttentionUNet:
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_segmenter_state_dict(checkpoint)

    model = AttentionUNet(
        in_channels=CONFIG.get("in_channels", 1),
        num_classes=CONFIG.get("num_classes", 1),
        base_filters=CONFIG.get("base_filters", 64),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        st.warning(f"Pesos faltantes: {list(missing)}")
    if unexpected:
        st.warning(f"Pesos inesperados ignorados: {list(unexpected)}")
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model: AttentionUNet, image_np: np.ndarray, device_name: str) -> np.ndarray:
    device = torch.device(device_name)
    tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="#070909",
        edgecolor="none",
    )
    buf.seek(0)
    return buf.read()


def metric_card(container, value: str, label: str) -> None:
    container.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_contours(ax, prob_map: np.ndarray, threshold: float, color: str = "#00ffcc") -> None:
    smooth = gaussian_filter(prob_map.astype(float), sigma=1.3)
    for contour in measure.find_contours(smooth, level=threshold):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2.2, alpha=0.95)


def draw_mask_contours(ax, mask: np.ndarray, color: str = "#00ffcc") -> None:
    for contour in measure.find_contours(mask.astype(float), level=0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2.2, alpha=0.95)


def draw_bboxes(ax, boxes: list[BBox], color: str = "#00d1b2", linewidth: float = 2.3) -> None:
    for box in boxes:
        xmin = max(0.0, min(256.0, box.xmin))
        ymin = max(0.0, min(256.0, box.ymin))
        xmax = max(0.0, min(256.0, box.xmax))
        ymax = max(0.0, min(256.0, box.ymax))
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            continue
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=0.95,
            )
        )


def make_panel(title: str):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#070909")
    ax.set_title(title, color="#708481", fontsize=8, fontfamily="monospace", pad=6)
    ax.axis("off")
    ax.set_facecolor("#000000")
    return fig, ax


with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.35rem">US Target</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Fase 3 · Produccion</div>', unsafe_allow_html=True)
    st.markdown("---")

    base_path = st.text_input("Directorio US procesado", value=DEFAULT_US_PATH)
    split = st.selectbox("Split", ["test", "val", "train", "all"], index=0)
    checkpoint_path = st.text_input("Checkpoint target", value=DEFAULT_CHECKPOINT)
    st.caption(f"Default autodetectado: {DEFAULT_CHECKPOINT}")

    st.markdown("---")
    threshold_slider = st.slider(
        "Umbral de mascara",
        min_value=0.0,
        max_value=0.95,
        value=float(CONFIG.get("threshold", 0.5)),
        step=0.01,
    )
    threshold = st.number_input(
        "Umbral fino",
        min_value=0.0,
        max_value=1.0,
        value=float(threshold_slider),
        step=0.001,
        format="%.3f",
        help="Util para diagnostico cuando prob_max queda muy por debajo de 0.5.",
    )
    opacity = st.slider("Opacidad overlay", 0.05, 0.75, 0.35, 0.05)

    st.markdown("---")
    min_area_px = st.number_input(
        "Min area postprocess px",
        min_value=1,
        max_value=5000,
        value=50,
        step=5,
    )
    bbox_margin_px = st.number_input(
        "Margen bbox px",
        min_value=0,
        max_value=128,
        value=8,
        step=1,
    )
    closing_kernel_px = st.select_slider(
        "Closing kernel px",
        options=[1, 3, 5, 7, 9],
        value=3,
    )

    st.markdown("---")
    use_gpu = st.checkbox("Usar GPU si disponible", value=True)
    device_name = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    st.markdown(f'<div class="info-box">Dispositivo: <b>{device_name}</b></div>', unsafe_allow_html=True)


st.markdown('<h1 class="main-title">Visualizador de Ultrasonido</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Attention U-Net adaptada a US - Inferencia Fase 3</p>',
    unsafe_allow_html=True,
)

if not os.path.exists(checkpoint_path):
    st.markdown(f'<div class="warn-box">No se encontro el checkpoint: {checkpoint_path}</div>', unsafe_allow_html=True)
    st.stop()
if not os.path.isdir(base_path):
    st.markdown(f'<div class="warn-box">No se encontro el directorio: {base_path}</div>', unsafe_allow_html=True)
    st.stop()

all_images = list_us_images(base_path, split)
if not all_images:
    st.error(f"No se encontraron .npy para split={split}.")
    st.stop()

model = load_segmenter(checkpoint_path, device_name)

st.markdown(
    f'<div class="info-box">Checkpoint cargado usando solo el segmentador - {len(all_images):,} imagenes US disponibles</div>',
    unsafe_allow_html=True,
)

labels = [os.path.relpath(p, base_path) for p in all_images]
selected_idx = st.selectbox(
    "Seleccionar ecografia",
    range(len(labels)),
    format_func=lambda i: labels[i],
)

img_path = all_images[int(selected_idx)]
image_np = load_us_npy(img_path)
bboxes = load_bboxes(img_path)

with st.spinner("Generando mascara..."):
    prob_map = predict(model, image_np, device_name)
post = postprocess_prediction(
    prob=prob_map,
    boxes=bboxes,
    threshold=threshold,
    min_area_px=int(min_area_px),
    bbox_margin_px=int(bbox_margin_px),
    closing_kernel_px=int(closing_kernel_px),
)
raw_mask = post.mask_before
mask = post.mask_final

_, n_objects = scipy_label(mask, np.ones((3, 3), dtype=int))
area_px = int(mask.sum())
area_mm2 = area_px * (0.8 ** 2)
max_prob = float(prob_map.max())
mean_prob_in_mask = float(prob_map[mask.astype(bool)].mean()) if area_px > 0 else 0.0

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
metric_card(m1, f"{area_px}", "Area px")
metric_card(m2, f"{area_mm2:.1f}", "Area mm2 aprox")
metric_card(m3, f"{post.stats['objects_before']} -> {n_objects}", "Objetos")
metric_card(m4, f"{max_prob:.3f}", "Prob max")

if max_prob < threshold:
    st.markdown(
        f'<div class="warn-box">Mascara vacia: prob_max={max_prob:.4f} esta por debajo '
        f'del umbral={threshold:.3f}. Para diagnostico, baja el umbral fino o revisa '
        f'si el checkpoint target ya entreno suficientes epocas.</div>',
        unsafe_allow_html=True,
    )

if not bboxes:
    st.markdown(
        f'<div class="warn-box">No se encontro bbox para esta imagen: {bbox_json_path(img_path)}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown('<div class="section-header">Visualizacion</div>', unsafe_allow_html=True)

cols = st.columns(3)

with cols[0]:
    fig, ax = make_panel("US procesado + bbox")
    ax.imshow(image_np, cmap=CMAP_US, vmin=0, vmax=1, interpolation="bicubic")
    draw_bboxes(ax, bboxes)
    draw_bboxes(ax, post.expanded_bboxes, color="#f2a93b", linewidth=1.4)
    st.image(fig_to_bytes(fig), width="stretch")
    plt.close(fig)

with cols[1]:
    fig, ax = make_panel(f"Antes postprocess - thr={threshold:.2f}")
    ax.imshow(image_np, cmap=CMAP_US, vmin=0, vmax=1, interpolation="bicubic")
    overlay = np.ma.masked_where(raw_mask == 0, raw_mask)
    ax.imshow(overlay, cmap="autumn", alpha=opacity, interpolation="nearest")
    draw_bboxes(ax, bboxes)
    draw_bboxes(ax, post.expanded_bboxes, color="#f2a93b", linewidth=1.4)
    st.image(fig_to_bytes(fig), width="stretch")
    plt.close(fig)

with cols[2]:
    fig, ax = make_panel("Mascara final postprocess")
    ax.imshow(image_np, cmap=CMAP_US, vmin=0, vmax=1, interpolation="bicubic")
    overlay = np.ma.masked_where(mask == 0, mask)
    ax.imshow(overlay, cmap="autumn", alpha=opacity, interpolation="nearest")
    draw_mask_contours(ax, mask)
    st.image(fig_to_bytes(fig), width="stretch")
    plt.close(fig)

st.markdown("---")
with st.expander("Guardar prediccion actual", expanded=False):
    output_dir = st.text_input("Carpeta de salida", value=os.path.join(str(ROOT), "outputs", "phase3_viewer_exports"))
    if st.button("Guardar mascara, probabilidad y overlay"):
        out = Path(output_dir)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        (out / "probabilities").mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem
        np.save(out / "masks" / f"{stem}_mask.npy", mask.astype(np.uint8))
        np.save(out / "probabilities" / f"{stem}_prob.npy", prob_map.astype(np.float32))

        fig, ax = make_panel(f"{stem} overlay")
        ax.imshow(image_np, cmap=CMAP_US, vmin=0, vmax=1, interpolation="bicubic")
        overlay = np.ma.masked_where(mask == 0, mask)
        ax.imshow(overlay, cmap="autumn", alpha=opacity, interpolation="nearest")
        draw_mask_contours(ax, mask)
        fig.savefig(out / f"{stem}_overlay.png", dpi=220, bbox_inches="tight", facecolor="#070909")
        plt.close(fig)
        save_postprocessing_debug_overlay(
            image_np=image_np,
            result=post,
            boxes=bboxes,
            output_dir=Path(str(ROOT)) / "outputs" / "postprocessing_debug",
            stem=stem,
        )

        st.success(f"Prediccion guardada en {out}")

with st.expander("Detalles tecnicos", expanded=False):
    st.write(
        {
            "archivo": img_path,
            "shape": tuple(image_np.shape),
            "rango_intensidad": [float(image_np.min()), float(image_np.max())],
            "threshold": threshold,
            "min_area_px": int(min_area_px),
            "bbox_margin_px": int(bbox_margin_px),
            "closing_kernel_px": int(closing_kernel_px),
            "postprocessing": post.stats,
            "area_px": area_px,
            "area_mm2_aprox_0_8mm_px": area_mm2,
            "prob_max": max_prob,
            "prob_media_en_mascara": mean_prob_in_mask,
            "checkpoint": checkpoint_path,
            "grl_domain_discriminator": "descartados en Fase 3",
        }
    )
