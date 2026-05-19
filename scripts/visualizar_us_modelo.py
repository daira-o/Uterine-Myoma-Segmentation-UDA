"""
scripts/visualizar_us_modelo.py
Visualizador Streamlit para Fase 3: inferencia de miomas en ultrasonido.

Carga un checkpoint DANN, descarta GRL/DomainDiscriminator y usa solo el
segmentador adaptado para predecir mascaras sobre US 256x256.

Uso:
    streamlit run scripts/visualizar_us_modelo.py
"""

from __future__ import annotations

import glob
import io
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

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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
from scripts.infer_us_production import extract_segmenter_state_dict


DEFAULT_US_PATH = (
    os.getenv("US_READY_PATH")
    or CONFIG.get("us_ready_path")
    or os.path.join(ROOT, "data_ready_US_curated")
)
DEFAULT_US_ORIGINAL_PATH = os.path.join(ROOT, "data", "Ultrasound")
DEFAULT_CHECKPOINT = os.path.join(
    ROOT,
    CONFIG.get("logs_path", "logs"),
    "checkpoints_dann",
    "best_model_dann.pth",
)

CMAP_US = LinearSegmentedColormap.from_list(
    "us_gray",
    ["#050505", "#1c1c1c", "#5d5d5d", "#b8b8b8", "#ffffff"],
    N=256,
)


st.set_page_config(
    page_title="Visualizador US DANN",
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


def parse_processed_us_name(path: str) -> tuple[str | None, str | None]:
    """
    Extrae FOV y stem original desde nombres como:
        11cm_1.2.826....npy -> ("11cm", "1.2.826...")
    """
    stem = Path(path).stem
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def find_original_us_image(processed_path: str, original_root: str) -> str | None:
    """Busca la imagen original previa al pipeline usando FOV + stem."""
    fov, source_stem = parse_processed_us_name(processed_path)
    if fov is None or source_stem is None:
        return None

    candidates = []
    for ext in (".jpg", ".jpeg", ".png"):
        candidates.append(os.path.join(original_root, fov, f"{source_stem}{ext}"))
        candidates.append(os.path.join(original_root, fov, f"{source_stem}{ext.upper()}"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    recursive = glob.glob(os.path.join(original_root, "**", f"{source_stem}.*"), recursive=True)
    for candidate in sorted(recursive):
        if Path(candidate).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return candidate
    return None


def orient_original_like_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Aplica la orientacion final del pipeline US:
        rotacion 90 grados horario + espejo horizontal.
    """
    return np.fliplr(np.rot90(image, k=-1))


def load_original_us_image(path: str) -> np.ndarray:
    """Carga la ecografia original y la rota para compararla con el tile procesado."""
    if not PIL_OK:
        raise ImportError("Pillow no esta instalado; no se puede cargar la imagen original.")
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return orient_original_like_pipeline(arr)


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


def make_panel(title: str):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#070909")
    ax.set_title(title, color="#708481", fontsize=8, fontfamily="monospace", pad=6)
    ax.axis("off")
    ax.set_facecolor("#000000")
    return fig, ax


with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.35rem">US DANN</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Fase 3 · Produccion</div>', unsafe_allow_html=True)
    st.markdown("---")

    base_path = st.text_input("Directorio US curado", value=DEFAULT_US_PATH)
    original_root = st.text_input("Directorio US original", value=DEFAULT_US_ORIGINAL_PATH)
    split = st.selectbox("Split", ["test", "val", "train", "all"], index=0)
    checkpoint_path = st.text_input("Checkpoint DANN", value=DEFAULT_CHECKPOINT)

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
    auto_prob_scale = st.checkbox(
        "Auto-contraste probabilidad",
        value=True,
        help="Escala el mapa de probabilidad al maximo de esta imagen para ver senales debiles.",
    )
    show_prob = st.checkbox("Mostrar mapa de probabilidad", value=True)
    show_binary = st.checkbox("Mostrar mascara binaria", value=True)

    st.markdown("---")
    use_gpu = st.checkbox("Usar GPU si disponible", value=True)
    device_name = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    st.markdown(f'<div class="info-box">Dispositivo: <b>{device_name}</b></div>', unsafe_allow_html=True)


st.markdown('<h1 class="main-title">Visualizador de Ultrasonido</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Attention U-Net adaptada con DANN · Inferencia Fase 3</p>',
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
    f'<div class="info-box">Checkpoint cargado sin GRL ni discriminator · {len(all_images):,} imagenes US disponibles</div>',
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
original_path = find_original_us_image(img_path, original_root)
original_np = None
original_status = "Original no encontrado"
if original_path is not None:
    try:
        original_np = load_original_us_image(original_path)
        original_status = os.path.relpath(original_path, original_root)
    except Exception as exc:
        original_status = f"No se pudo cargar original: {exc}"

with st.spinner("Generando mascara..."):
    prob_map = predict(model, image_np, device_name)
mask = (prob_map >= threshold).astype(np.uint8)

_, n_objects = scipy_label(mask, np.ones((3, 3), dtype=int))
area_px = int(mask.sum())
area_mm2 = area_px * (0.8 ** 2)
max_prob = float(prob_map.max())
mean_prob_in_mask = float(prob_map[mask.astype(bool)].mean()) if area_px > 0 else 0.0

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
metric_card(m1, f"{area_px}", "Area px")
metric_card(m2, f"{area_mm2:.1f}", "Area mm2 aprox")
metric_card(m3, f"{n_objects}", "Objetos")
metric_card(m4, f"{max_prob:.3f}", "Prob max")

if max_prob < threshold:
    st.markdown(
        f'<div class="warn-box">Mascara vacia: prob_max={max_prob:.4f} esta por debajo '
        f'del umbral={threshold:.3f}. Para diagnostico, baja el umbral fino o revisa '
        f'si el checkpoint DANN ya entreno suficientes epocas.</div>',
        unsafe_allow_html=True,
    )

if original_np is None:
    st.markdown(
        f'<div class="warn-box">{original_status}. Se esperaba buscar en: {original_root}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown('<div class="section-header">Visualizacion</div>', unsafe_allow_html=True)

num_panels = 3 + int(show_prob) + int(show_binary)
cols = st.columns(num_panels)

with cols[0]:
    fig, ax = make_panel("US original rotado")
    if original_np is not None:
        ax.imshow(original_np, cmap=CMAP_US, vmin=0, vmax=1, interpolation="bicubic")
        ax.set_title(f"Original · {original_status}", color="#708481", fontsize=6, fontfamily="monospace", pad=6)
    else:
        ax.text(
            0.5,
            0.5,
            "Original no encontrado",
            color="#f2a93b",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            fontfamily="monospace",
        )
    st.image(fig_to_bytes(fig), width="stretch")
    plt.close(fig)

with cols[1]:
    fig, ax = make_panel("US estandarizado 256x256")
    ax.imshow(image_np, cmap=CMAP_US, vmin=0, vmax=1, interpolation="bicubic")
    st.image(fig_to_bytes(fig), width="stretch")
    plt.close(fig)

with cols[2]:
    fig, ax = make_panel(f"Overlay mascara · thr={threshold:.2f}")
    ax.imshow(image_np, cmap=CMAP_US, vmin=0, vmax=1, interpolation="bicubic")
    overlay = np.ma.masked_where(mask == 0, mask)
    ax.imshow(overlay, cmap="autumn", alpha=opacity, interpolation="nearest")
    draw_contours(ax, prob_map, threshold)
    st.image(fig_to_bytes(fig), width="stretch")
    plt.close(fig)

panel_idx = 3
if show_prob:
    with cols[panel_idx]:
        fig, ax = make_panel("Mapa de probabilidad")
        vmax = max(float(prob_map.max()), 1e-6) if auto_prob_scale else 1.0
        im = ax.imshow(prob_map, cmap="magma", vmin=0, vmax=vmax, interpolation="bicubic")
        draw_contours(ax, prob_map, threshold)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#708481", labelsize=6)
        st.image(fig_to_bytes(fig), width="stretch")
        plt.close(fig)
    panel_idx += 1

if show_binary:
    with cols[panel_idx]:
        fig, ax = make_panel("Mascara binaria")
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        st.image(fig_to_bytes(fig), width="stretch")
        plt.close(fig)

st.markdown("---")
with st.expander("Guardar prediccion actual", expanded=False):
    output_dir = st.text_input("Carpeta de salida", value=os.path.join(ROOT, "outputs", "phase3_viewer_exports"))
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
        draw_contours(ax, prob_map, threshold)
        fig.savefig(out / f"{stem}_overlay.png", dpi=220, bbox_inches="tight", facecolor="#070909")
        plt.close(fig)

        st.success(f"Prediccion guardada en {out}")

with st.expander("Detalles tecnicos", expanded=False):
    st.write(
        {
            "archivo": img_path,
            "shape": tuple(image_np.shape),
            "rango_intensidad": [float(image_np.min()), float(image_np.max())],
            "threshold": threshold,
            "area_px": area_px,
            "area_mm2_aprox_0_8mm_px": area_mm2,
            "prob_max": max_prob,
            "prob_media_en_mascara": mean_prob_in_mask,
            "checkpoint": checkpoint_path,
            "grl_domain_discriminator": "descartados en Fase 3",
        }
    )
