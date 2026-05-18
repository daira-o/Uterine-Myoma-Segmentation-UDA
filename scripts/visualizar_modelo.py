"""
visualizar_modelo.py
Dashboard interactivo para el modelo Attention U-Net de segmentacion de miomas.

ESTRATEGIA DE ALTA FIDELIDAD:
  El modelo opera sobre .npy de 256x256.
  El visualizador recupera el NIfTI original (resolucion nativa) y proyecta
  el contorno vectorial de la prediccion escalado sobre el.
  Los contornos son poligonos vectoriales (marching squares), no pixeles.

Requisitos:
    pip install streamlit torch numpy matplotlib scipy scikit-image nibabel Pillow

Uso:
    streamlit run visualizar_modelo.py
"""

import os
import re
import glob
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import streamlit as st
import io
from scipy.ndimage import gaussian_filter
from skimage import measure

try:
    import nibabel as nib
    NIBABEL_OK = True
except ImportError:
    NIBABEL_OK = False

warnings.filterwarnings("ignore", category=UserWarning)

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CONFIG
from models.attention_unet import AttentionUNet, compute_all_metrics

TARGET_SPACING_MM = 0.8
TILE_SIZE = 256


# -----------------------------------------------------------------------------
#  PAGINA
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Visualizador de Miomas",
    page_icon="ðŸ”¬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg: #080c10; --surface: #0e1420; --border: #1e2d42;
    --accent: #00c9a7; --accent2: #e05c8a; --text: #c8d8e8;
    --muted: #4a6070; --warn: #f5a623;
}
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }
.main-title {
    font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00c9a7 0%, #4facfe 50%, #e05c8a 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0;
}
.subtitle { color: var(--muted); font-size: 0.78rem; letter-spacing: 0.15em; text-transform: uppercase; }
.metric-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; text-align: center;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: var(--accent); line-height: 1; }
.metric-label { font-size: 0.7rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px; }
.metric-sub   { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }
.section-header {
    font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 12px;
}
.info-box  { background: rgba(0,201,167,.06); border-left: 3px solid var(--accent);  border-radius: 0 6px 6px 0; padding: 10px 14px; font-size: .78rem; margin-bottom: 12px; }
.warn-box  { background: rgba(245,166,35,.08); border-left: 3px solid var(--warn);   border-radius: 0 6px 6px 0; padding: 10px 14px; font-size: .78rem; color: var(--warn); margin-bottom: 12px; }
.hires-badge {
    display: inline-block; background: rgba(0,201,167,.15);
    border: 1px solid var(--accent); border-radius: 4px;
    padding: 2px 8px; font-size: .68rem; color: var(--accent);
    letter-spacing: .1em; text-transform: uppercase;
}
.lores-badge {
    display: inline-block; background: rgba(245,166,35,.12);
    border: 1px solid var(--warn); border-radius: 4px;
    padding: 2px 8px; font-size: .68rem; color: var(--warn);
    letter-spacing: .1em; text-transform: uppercase;
}
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
.stButton > button {
    background: transparent !important; border: 1px solid var(--accent) !important;
    color: var(--accent) !important; font-family: 'DM Mono', monospace !important;
    font-size: .78rem !important; letter-spacing: .08em !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
#  COLORMAPS
# -----------------------------------------------------------------------------

CMAP_MRI = LinearSegmentedColormap.from_list(
    "mri", ["#020408", "#0a2540", "#1a4a7a", "#4a9aca", "#c8e8f8"], N=256)


# -----------------------------------------------------------------------------
#  HELPERS DE CARGA
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(model_path, device):
    model = AttentionUNet(in_channels=1, num_classes=1)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@st.cache_data(show_spinner=False)
def list_samples(base_path, split="val"):
    if split and split != "all":
        img_patterns = [
            os.path.join(base_path, split, "images", "*.npy"),
            os.path.join(base_path, split, "images_npy", "*.npy"),
        ]
        mask_patterns = [
            os.path.join(base_path, split, "masks", "*.npy"),
            os.path.join(base_path, split, "masks_npy", "*.npy"),
        ]
    else:
        img_patterns = [
            os.path.join(base_path, "*", "images", "*.npy"),
            os.path.join(base_path, "images", "*.npy"),
            os.path.join(base_path, "images_npy", "*", "*.npy"),
            os.path.join(base_path, "images_npy", "*.npy"),
        ]
        mask_patterns = [
            os.path.join(base_path, "*", "masks", "*.npy"),
            os.path.join(base_path, "masks", "*.npy"),
            os.path.join(base_path, "masks_npy", "*", "*.npy"),
            os.path.join(base_path, "masks_npy", "*.npy"),
        ]

    imgs = sorted({p for pattern in img_patterns for p in glob.glob(pattern)})
    masks = sorted({p for pattern in mask_patterns for p in glob.glob(pattern)})

    mask_by_name = {os.path.basename(p): p for p in masks}
    paired_masks = [mask_by_name.get(os.path.basename(p)) for p in imgs]
    return imgs, paired_masks


def sample_label(path, base_path):
    rel = os.path.relpath(path, base_path)
    parts = rel.split(os.sep)
    if len(parts) >= 3 and parts[1] in ("images", "images_npy"):
        return f"{parts[0]} / {parts[-1]}"
    return rel


def load_npy(path):
    arr = np.load(path).astype("float32")
    if arr.max() > 1.0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


@st.cache_data(show_spinner=False)
def load_hires_slice(nii_path: str, slice_idx: int):
    """
    Carga el mismo corte del NIfTI que usa mri_pipeline.py.

    El pipeline actual extrae eje 0:
        img_slice = data_img[i, :, :]

    Para superponer prediccion/GT sobre el NIfTI original, tambien devuelve
    la transformacion inversa del tile 256x256 al corte original, deshaciendo
    el resampling fisico y el center crop/padding.
    """
    if not NIBABEL_OK:
        return None, "nibabel no instalado  ->  pip install nibabel"
    if not os.path.exists(nii_path):
        return None, f"archivo no encontrado: {nii_path}"
    try:
        img_nib = nib.as_closest_canonical(nib.load(nii_path))
        vol     = img_nib.get_fdata(dtype=np.float32)
    except Exception as e:
        return None, str(e)

    if slice_idx < 0 or slice_idx >= vol.shape[0]:
        return None, f"sag_{slice_idx} fuera de rango para eje 0 (n={vol.shape[0]})"

    slc = vol[slice_idx, :, :].astype("float32")
    spacing_row, spacing_col = [float(v) for v in img_nib.header.get_zooms()[1:3]]
    orig_h, orig_w = slc.shape
    res_h = max(1, round(orig_h * spacing_row / TARGET_SPACING_MM))
    res_w = max(1, round(orig_w * spacing_col / TARGET_SPACING_MM))

    transform = {
        "orig_shape": (orig_h, orig_w),
        "res_shape": (res_h, res_w),
        "tile_shape": (TILE_SIZE, TILE_SIZE),
        "display_rotation": "left",
    }

    slc = np.rot90(slc, k=1)

    # -- Normalizar a [0,1] ------------------------------------------------
    d = slc.max() - slc.min()
    slc = (slc - slc.min()) / (d if d != 0 else 1.0)

    info = (
        f"{slc.shape[0]}x{slc.shape[1]} eje=0 corte={slice_idx} | "
        f"resample={res_h}x{res_w} -> tile={TILE_SIZE}x{TILE_SIZE}"
    )
    return {"image": slc, "transform": transform}, info


def parse_npy_name(npy_path: str):
    """
    Extrae patient_id y slice_idx del nombre {patient_id}_sag_{idx}.npy
    generado por procesador_imagenes.py.
    """
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    m = re.match(r"^(.+)_sag_(\d+)$", stem)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def find_nii(nifti_root: str, patient_id: str, suffix: str):
    hits = glob.glob(os.path.join(nifti_root, patient_id, f"*{suffix}.nii*"))
    return hits[0] if hits else None


@torch.no_grad()
def predict(model, img_np, device, threshold):
    t    = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    return prob, (prob >= threshold).astype("float32")


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                facecolor="#080c10", edgecolor="none")
    buf.seek(0)
    return buf.read()


def make_fig(size=(5, 5)):
    return plt.subplots(figsize=size, facecolor="#080c10")


# -----------------------------------------------------------------------------
#  HELPERS DE RENDERIZADO VECTORIAL
# -----------------------------------------------------------------------------

def render_base(ax, img_hires, img_lores):
    """Dibuja la imagen base (hires si existe, lores como fallback)."""
    img = img_hires if img_hires is not None else img_lores
    ax.imshow(img, cmap=CMAP_MRI, vmin=0, vmax=1,
              interpolation="bicubic", aspect="equal")
    ax.set_facecolor("#000000")
    ax.axis("off")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    return img


def tile_contour_to_display(contour, source_shape, display_shape, transform=None):
    if transform is None:
        src_h, src_w = source_shape
        dst_h, dst_w = display_shape
        sx = dst_w / src_w
        sy = dst_h / src_h
        return np.column_stack([contour[:, 1] * sx, contour[:, 0] * sy])

    orig_h, orig_w = transform["orig_shape"]
    res_h, res_w = transform["res_shape"]
    tile_h, tile_w = transform["tile_shape"]

    if res_h <= tile_h:
        res_y = contour[:, 0] - ((tile_h - res_h) / 2.0)
    else:
        res_y = contour[:, 0] + ((res_h - tile_h) / 2.0)

    if res_w <= tile_w:
        res_x = contour[:, 1] - ((tile_w - res_w) / 2.0)
    else:
        res_x = contour[:, 1] + ((res_w - tile_w) / 2.0)

    x = res_x * (orig_w / res_w)
    y = res_y * (orig_h / res_h)

    if transform.get("display_rotation") == "left":
        x, y = y, orig_w - x

    return np.column_stack([x, y])


def draw_contours(ax, prob_map, thr, display_shape,
                  fill_color, line_color, fill_alpha,
                  line_width=2.0, label="", transform=None):
    """
    Proyecta la segmentacion del modelo (256x256) como contornos vectoriales
    sobre una imagen de resolucion arbitraria (display_shape).

    El escalado es proporcional: cada punto del contorno se multiplica por
    (dst / src) en x e y. Como los contornos son poligonos (no pixeles),
    la calidad es completamente independiente de la resolucion destino.
    """
    smooth   = gaussian_filter(prob_map.astype(float), sigma=1.5)
    contours = measure.find_contours(smooth, level=thr)
    if not contours:
        return

    for i, c in enumerate(contours):
        xy = tile_contour_to_display(c, prob_map.shape, display_shape, transform)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(xy) - 1) + [Path.CLOSEPOLY]
        path  = Path(np.vstack([xy, xy[0]]), codes)

        ax.add_patch(PathPatch(path,
                               facecolor=fill_color, edgecolor="none",
                               alpha=fill_alpha, linewidth=0, zorder=3))
        ax.plot(xy[:, 0], xy[:, 1],
                color=line_color, linewidth=line_width,
                alpha=0.95, zorder=4,
                label=label if i == 0 else "")


def draw_error_contours(ax, prob_map, mask_np, thr, display_shape, transform=None):
    """TP / FP / FN como contornos vectoriales escalados a display_shape."""
    smooth    = gaussian_filter(prob_map.astype(float), sigma=1.5)
    pred_soft = (smooth >= thr).astype("float32")
    gt_soft   = (mask_np >= 0.5).astype("float32")

    regions = {
        "TP": (pred_soft * gt_soft,            "#00c9a7", "#00ffcc", "TP"),
        "FP": (pred_soft * (1 - gt_soft),      "#ff8c42", "#ffb347", "FP (falso positivo)"),
        "FN": ((1 - pred_soft) * gt_soft,      "#c979c7", "#e8a0e8", "FN (falso negativo)"),
    }
    for _, (region, fill, line, lbl) in regions.items():
        if region.sum() < 4:
            continue
        rs    = gaussian_filter(region.astype(float), sigma=0.8)
        first = True
        for c in measure.find_contours(rs, level=0.5):
            xy = tile_contour_to_display(c, prob_map.shape, display_shape, transform)
            codes = [Path.MOVETO] + [Path.LINETO] * (len(xy) - 1) + [Path.CLOSEPOLY]
            path  = Path(np.vstack([xy, xy[0]]), codes)
            ax.add_patch(PathPatch(path, facecolor=fill, edgecolor="none",
                                   alpha=0.35, zorder=3))
            ax.plot(xy[:, 0], xy[:, 1], color=line, linewidth=1.8,
                    alpha=0.9, zorder=4, label=lbl if first else "")
            first = False


# -----------------------------------------------------------------------------
#  SIDEBAR
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.4rem">Visualizador de Miomas</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Attention U-Net Inspector</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Rutas (.npy)</div>', unsafe_allow_html=True)
    base_path  = st.text_input(
        "Directorio de datos (.npy)",
        value=CONFIG["base_path"])
    split_filter = st.selectbox(
        "Split",
        ["val", "test", "train", "all"],
        index=0,
        help="Val es el default porque son casos no vistos durante entrenamiento.")
    model_path = st.text_input("Checkpoint (.pth)", value=CONFIG["model_path"])

    st.markdown("---")
    st.markdown('<div class="section-header">NIfTI original (alta resolucion)</div>',
                unsafe_allow_html=True)
    nifti_root = st.text_input(
        "Carpeta raiz NIfTI",
        value=CONFIG["nifti_root"],
        help="El mismo base_path de procesador_imagenes.py")
    img_suffix = st.text_input("Sufijo imagen NIfTI", value=CONFIG["nifti_img_suffix"],
                               help="El mismo img_suffix de procesador_imagenes.py")
    use_hires  = st.checkbox("Usar NIfTI original como base", value=True,
                             help="ON = imagen nativa. OFF = .npy 256x256.\n"
                                  "El contorno siempre se escala correctamente en ambos casos.")

    st.markdown("---")
    st.markdown('<div class="section-header">Inferencia</div>', unsafe_allow_html=True)
    threshold = st.slider("Umbral de binarizacion", 0.1, 0.9, 0.5, 0.05)
    iou_thr   = st.slider("IoU threshold (Object Precision)", 0.05, 0.5, 0.10, 0.05)

    st.markdown("---")
    st.markdown('<div class="section-header">Visualizacion</div>', unsafe_allow_html=True)
    overlay_alpha = st.slider("Opacidad relleno", 0.05, 0.6, 0.25, 0.05)
    line_width    = st.slider("Grosor contorno", 0.5, 5.0, 2.2, 0.5)
    show_gt       = st.checkbox("Mostrar Ground Truth", value=True)
    show_prob_map = st.checkbox("Mostrar mapa de probabilidades", value=True)
    show_diff_map = st.checkbox("Mostrar mapa de diferencias", value=True)

    st.markdown("---")
    use_gpu = st.checkbox("Usar GPU (si disponible)", value=True)
    device  = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    st.markdown(f'<div class="info-box">Dispositivo: <b>{device}</b></div>',
                unsafe_allow_html=True)
    if not NIBABEL_OK:
        st.markdown(
            '<div class="warn-box">nibabel no instalado.<br>'
            'Ejecuta: pip install nibabel</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
#  HEADER
# -----------------------------------------------------------------------------

st.markdown('<h1 class="main-title">Visualizador de Miomas</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Visualizador â€” Attention U-Net Â· Miomas Uterinos Â· RM Sagital</p>',
    unsafe_allow_html=True)
st.markdown("")


# -----------------------------------------------------------------------------
#  VALIDACION DE RUTAS
# -----------------------------------------------------------------------------

if not os.path.exists(model_path):
    st.markdown(f'<div class="warn-box">No se encontro el checkpoint: {model_path}</div>',
                unsafe_allow_html=True)
    st.stop()
if not os.path.isdir(base_path):
    st.markdown(f'<div class="warn-box">No se encontro el directorio: {base_path}</div>',
                unsafe_allow_html=True)
    st.stop()

model = load_model(model_path, device)
all_imgs, all_masks = list_samples(base_path, split_filter)

if not all_imgs:
    st.error(f"No se encontraron archivos .npy para split='{split_filter}'.")
    st.stop()

st.markdown(
    f'<div class="info-box">Modelo cargado Â· split: <b>{split_filter}</b> Â· '
    f'{len(all_imgs):,} muestras disponibles</div>',
            unsafe_allow_html=True)


# -----------------------------------------------------------------------------
#  SELECTOR DE MUESTRA
# -----------------------------------------------------------------------------

col_sel1, col_sel2, col_sel3, col_r = st.columns([3, 1, 1, 1])

with col_sel1:
    sample_names  = [sample_label(p, base_path) for p in all_imgs]
    selected_idx = st.selectbox(
        "Seleccionar imagen",
        range(len(sample_names)),
        index=0,
        format_func=lambda i: sample_names[i],
    )
    idx = int(selected_idx)

with col_sel2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Anterior"):
        st.session_state["idx"] = max(0, idx - 1)

with col_sel3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Siguiente"):
        st.session_state["idx"] = min(len(all_imgs) - 1, idx + 1)

with col_r:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Aleatoria"):
        st.session_state["idx"] = int(np.random.randint(0, len(all_imgs)))

if "idx" in st.session_state:
    idx = st.session_state["idx"]


# -----------------------------------------------------------------------------
#  INFERENCIA
# -----------------------------------------------------------------------------

img_path = all_imgs[idx]
mask_path = all_masks[idx] if idx < len(all_masks) else None
img_npy  = load_npy(img_path)
mask_npy = load_npy(mask_path) if mask_path else np.zeros_like(img_npy)

with st.spinner("Generando prediccion..."):
    prob_map, pred_bin = predict(model, img_npy, device, threshold)


# -----------------------------------------------------------------------------
#  RECUPERAR NIFTI ORIGINAL
# -----------------------------------------------------------------------------

patient_id, slice_idx = parse_npy_name(img_path)
img_hires    = None
hires_transform = None
hires_status = "desactivado"

if use_hires and NIBABEL_OK and patient_id is not None:
    nii_path = find_nii(nifti_root, patient_id, img_suffix)
    if nii_path:
        hires_data, info = load_hires_slice(nii_path, slice_idx)
        if isinstance(hires_data, dict):
            img_hires = hires_data["image"]
            hires_transform = hires_data["transform"]
        else:
            img_hires = hires_data
        hires_status = f"OK ({info})" if img_hires is not None else info
    else:
        hires_status = f"NIfTI no encontrado para paciente '{patient_id}'"
elif not NIBABEL_OK:
    hires_status = "nibabel no instalado"
elif patient_id is None:
    hires_status = "nombre .npy no coincide con patron patient_id_sag_N"

# Forma efectiva para escalar los contornos
display_shape = img_hires.shape if img_hires is not None else img_npy.shape

if img_hires is not None:
    badge = (f'<span class="hires-badge">'
             f'Alta res {display_shape[0]}x{display_shape[1]}</span>')
else:
    badge = (f'<span class="lores-badge">'
             f'256x256 â€” {hires_status}</span>')

st.markdown(f"Resolucion de visualizacion: {badge}", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
#  METRICAS
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Metricas de esta muestra</div>',
            unsafe_allow_html=True)

with torch.no_grad():
    logits_t = model(torch.from_numpy(img_npy).unsqueeze(0).unsqueeze(0).to(device))
mask_t  = torch.from_numpy(mask_npy).unsqueeze(0).unsqueeze(0).to(device)
metrics = compute_all_metrics(logits_t, mask_t, threshold=threshold, iou_threshold=iou_thr)

dice_val  = metrics["Dice"]
hd95_val  = metrics["HD95"]
oprec_val = metrics["Object_Precision"]


def metric_card(container, value_str, label, sub="", color="var(--accent)"):
    container.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value_str}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


mc1, mc2, mc3, mc4 = st.columns(4)
metric_card(mc1, f"{dice_val:.4f}", "Dice Coefficient", "Overlap semantico")
hd95_str = f"{hd95_val:.2f}px" if np.isfinite(hd95_val) else "inf"
metric_card(mc2, hd95_str, "HD95", "Precision de bordes (menor = mejor)",
            color="var(--accent)" if np.isfinite(hd95_val) and hd95_val < 10 else "var(--warn)")
metric_card(mc3, f"{oprec_val:.4f}", "Object Precision", "Por instancia",
            color="var(--accent)" if oprec_val > 0.85 else "var(--warn)")

from scipy.ndimage import label as scipy_label
_, n_pred = scipy_label(pred_bin, np.ones((3, 3), int))
_, n_gt   = scipy_label((mask_npy > 0.5).astype(np.uint8), np.ones((3, 3), int))
metric_card(mc4, f"{n_pred} / {n_gt}", "Objetos Pred / GT", "Miomas detectados vs reales",
            color="var(--accent)" if n_pred == n_gt else "var(--accent2)")


# -----------------------------------------------------------------------------
#  VISUALIZACIONES
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-header">Visualizacion</div>', unsafe_allow_html=True)

n_panels = 2 + int(show_prob_map) + int(show_diff_map)
cols     = st.columns(n_panels)


# ---- Panel 1: imagen base ---------------------------------------------------
with cols[0]:
    titulo = ("MRI original (NIfTI nativo)" if img_hires is not None
              else "MRI procesado (.npy 256x256)")
    st.markdown(f'<div class="section-header" style="text-align:center">{titulo}</div>',
                unsafe_allow_html=True)
    fig, ax = make_fig()
    used = render_base(ax, img_hires, img_npy)
    ax.set_title(f"{os.path.basename(img_path)}   {used.shape[0]}x{used.shape[1]}",
                 color="#4a6070", fontsize=6, fontfamily="monospace", pad=4)
    st.image(fig_to_bytes(fig), width="stretch")
    plt.close(fig)


# ---- Panel 2: overlay con contornos escalados -------------------------------
with cols[1]:
    lbl2 = "Pred (teal) + GT (rosa)" if show_gt else "Prediccion del modelo"
    st.markdown(f'<div class="section-header" style="text-align:center">{lbl2}</div>',
                unsafe_allow_html=True)
    fig, ax = make_fig()
    render_base(ax, img_hires, img_npy)

    if show_gt:
        draw_contours(ax, mask_npy, 0.5, display_shape,
                      fill_color="#e05c8a", line_color="#ff9ec0",
                      fill_alpha=overlay_alpha * 0.8,
                      line_width=line_width, label="Ground Truth",
                      transform=hires_transform)

    draw_contours(ax, prob_map, threshold, display_shape,
                  fill_color="#00c9a7", line_color="#00ffcc",
                  fill_alpha=overlay_alpha,
                  line_width=line_width + 0.4,
                  label=f"Prediccion (thr={threshold})",
                  transform=hires_transform)

    ax.legend(loc="lower right", fontsize=6,
              facecolor="#0e1420", edgecolor="#1e2d42",
              labelcolor="#c8d8e8", framealpha=0.9)
    st.image(fig_to_bytes(fig), width="stretch")
    plt.close(fig)


# ---- Panel 3: mapa de probabilidades ----------------------------------------
panel_idx = 2
if show_prob_map and panel_idx < len(cols):
    with cols[panel_idx]:
        st.markdown(
            '<div class="section-header" style="text-align:center">Probabilidad sigma(logits)</div>',
            unsafe_allow_html=True)
        fig, ax = make_fig()
        render_base(ax, img_hires, img_npy)

        if hires_transform is None:
            im = ax.imshow(prob_map, cmap="plasma", vmin=0, vmax=1,
                           alpha=0.60, interpolation="bicubic",
                           extent=[0, display_shape[1], display_shape[0], 0],
                           aspect="auto", zorder=2)
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.ax.yaxis.set_tick_params(color="#4a6070", labelsize=5, labelcolor="#4a6070")

        # Contorno del umbral vectorial escalado
        smooth = gaussian_filter(prob_map.astype(float), sigma=1.5)
        for c in measure.find_contours(smooth, level=threshold):
            xy = tile_contour_to_display(c, prob_map.shape, display_shape, hires_transform)
            ax.plot(xy[:, 0], xy[:, 1],
                    color="#00ffcc", linewidth=2.2, alpha=0.95, zorder=6)

        ax.set_title(f"contorno = umbral {threshold}",
                     color="#4a6070", fontsize=6, fontfamily="monospace", pad=4)
        st.image(fig_to_bytes(fig), width="stretch")
        plt.close(fig)
    panel_idx += 1


# ---- Panel 4: error map vectorial -------------------------------------------
if show_diff_map and panel_idx < len(cols):
    with cols[panel_idx]:
        st.markdown(
            '<div class="section-header" style="text-align:center">Error Map (FP / FN)</div>',
            unsafe_allow_html=True)
        fig, ax = make_fig()
        render_base(ax, img_hires, img_npy)
        draw_error_contours(ax, prob_map, mask_npy, threshold, display_shape,
                            transform=hires_transform)

        smooth_   = gaussian_filter(prob_map.astype(float), sigma=1.5)
        pred_bin_ = (smooth_ >= threshold).astype("float32")
        gt_bin_   = (mask_npy >= 0.5).astype("float32")
        fp_px = int((pred_bin_ * (1 - gt_bin_)).sum())
        fn_px = int(((1 - pred_bin_) * gt_bin_).sum())

        ax.legend(loc="lower right", fontsize=6,
                  facecolor="#0e1420", edgecolor="#1e2d42",
                  labelcolor="#c8d8e8", framealpha=0.9)
        ax.set_title(f"FP={fp_px}px  FN={fn_px}px (espacio 256x256)",
                     color="#4a6070", fontsize=6, fontfamily="monospace", pad=4)
        st.image(fig_to_bytes(fig), width="stretch")
        plt.close(fig)


# -----------------------------------------------------------------------------
#  HISTOGRAMA
# -----------------------------------------------------------------------------

st.markdown("---")
with st.expander("Distribucion de probabilidades predichas", expanded=False):
    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="#080c10")
    ax.set_facecolor("#0e1420")
    vals = prob_map.ravel()
    n, bins, patches = ax.hist(vals, bins=80, color="#1e2d42", edgecolor="none")
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor("#00c9a7" if left >= threshold else "#4a6070")
        patch.set_alpha(0.85)
    ax.axvline(threshold, color="#e05c8a", lw=1.5, linestyle="--",
               label=f"Umbral = {threshold}")
    ax.set_xlabel("Probabilidad", color="#4a6070", fontsize=8, fontfamily="monospace")
    ax.set_ylabel("N pixeles", color="#4a6070", fontsize=8, fontfamily="monospace")
    ax.tick_params(colors="#4a6070", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d42")
    ax.legend(fontsize=7, facecolor="#0e1420", edgecolor="#1e2d42", labelcolor="#c8d8e8")
    pct = (vals >= threshold).mean() * 100
    ax.set_title(f"{pct:.1f}% pixeles como mioma  |  max prob={vals.max():.4f}",
                 color="#4a6070", fontsize=7, fontfamily="monospace", pad=6)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# -----------------------------------------------------------------------------
#  EVALUACION DE BATCH
# -----------------------------------------------------------------------------

st.markdown("---")
with st.expander("Evaluar batch aleatorio (N muestras)", expanded=False):
    n_batch = st.slider("Cantidad de muestras", 5, 100, 20, 5)

    if st.button("Ejecutar evaluacion"):
        idxs    = np.random.choice(len(all_imgs), size=min(n_batch, len(all_imgs)), replace=False)
        batch_m = {"Dice": [], "HD95": [], "Object_Precision": []}
        prog    = st.progress(0.0, text="Evaluando...")

        for k, bi in enumerate(idxs):
            bimg  = load_npy(all_imgs[bi])
            bmask = (load_npy(all_masks[bi]) if bi < len(all_masks) and all_masks[bi]
                     else np.zeros_like(bimg))
            with torch.no_grad():
                bl = model(torch.from_numpy(bimg).unsqueeze(0).unsqueeze(0).to(device))
            bm = compute_all_metrics(
                bl,
                torch.from_numpy(bmask).unsqueeze(0).unsqueeze(0).to(device),
                threshold=threshold, iou_threshold=iou_thr)
            for key in batch_m:
                if np.isfinite(bm[key]):
                    batch_m[key].append(bm[key])
            prog.progress((k + 1) / len(idxs), text=f"Evaluando {k+1}/{len(idxs)}...")

        prog.empty()
        bc1, bc2, bc3 = st.columns(3)
        metric_card(bc1, f"{np.mean(batch_m['Dice']):.4f}", "Dice medio",
                    f"sigma={np.std(batch_m['Dice']):.4f}")
        hd_m = np.mean(batch_m["HD95"]) if batch_m["HD95"] else np.inf
        metric_card(bc2,
                    f"{hd_m:.2f}px" if np.isfinite(hd_m) else "inf",
                    "HD95 medio")
        metric_card(bc3, f"{np.mean(batch_m['Object_Precision']):.4f}",
                    "Obj Precision media")

        fig, axes = plt.subplots(1, 3, figsize=(12, 3), facecolor="#080c10")
        for ax, d, c, lbl in zip(
                axes,
                [batch_m["Dice"], batch_m["HD95"], batch_m["Object_Precision"]],
                ["#00c9a7", "#4facfe", "#e05c8a"],
                ["Dice", "HD95 (px)", "Object Precision"]):
            ax.set_facecolor("#0e1420")
            if d:
                ax.boxplot(d, patch_artist=True,
                           medianprops={"color": c, "lw": 2},
                           boxprops={"facecolor": c, "alpha": 0.25, "edgecolor": c},
                           whiskerprops={"color": "#4a6070"},
                           capprops={"color": "#4a6070"},
                           flierprops={"marker": "o", "markersize": 3,
                                       "markerfacecolor": c, "alpha": 0.5})
                ax.scatter(
                    np.ones(len(d)) + np.random.uniform(-0.1, 0.1, len(d)),
                    d, color=c, alpha=0.35, s=12, zorder=5)
            ax.set_title(lbl, color="#c8d8e8", fontsize=8, fontfamily="monospace", pad=4)
            ax.tick_params(colors="#4a6070", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2d42")
            ax.set_xticks([])
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


# -----------------------------------------------------------------------------
#  FOOTER
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    '<p style="color:#2a3d50;font-size:.68rem;font-family:monospace;text-align:center">'
    'Visualizador de Miomas Â· Attention U-Net Â· Dice 0.8962 Â· HD95 4.24px Â· ObjPrec 0.9220 Â· Epoch 6'
    '</p>', unsafe_allow_html=True)

