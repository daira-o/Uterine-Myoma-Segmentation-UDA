import nibabel as nib
import numpy as np
import os
import cv2
from glob import glob
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

MRI_BASE_PATH = os.getenv("MRI_DATA_PATH", os.getenv("NIFTI_ROOT", str(PROJECT_ROOT / "data" / "UMD")))
MRI_OUTPUT_PATH = os.getenv("MRI_OUTPUT_PATH", str(PROJECT_ROOT / "data_ready_RM"))
MRI_IMG_SUFFIX = os.getenv("NIFTI_IMG_SUFFIX", "_t2")
MRI_MASK_SUFFIX = os.getenv("NIFTI_MASK_SUFFIX", "_seg")

US_BASE_PATH = os.getenv("US_DATA_PATH", str(PROJECT_ROOT / "data" / "Ultrasound"))
US_OUTPUT_PATH = os.getenv("US_OUTPUT_PATH", str(PROJECT_ROOT / "data_ready_US"))
IMAGE_SIZE = int(os.getenv("PROCESSOR_IMAGE_SIZE", "256"))

# Resolución física objetivo: 1 píxel = TARGET_SPACING_MM mm de tejido real
TARGET_SPACING_MM = 0.8

SLICE_VIEWS = (
    ("mid", "1_Anchor"),
    ("left", "2_Lateral"),
    ("right", "2_Lateral"),
)

US_VIEW_MAP = {
    "F28": ("mid", "1_Anchor"),
    "F18": ("left", "2_Lateral"),
    "F38": ("right", "2_Lateral"),
}

US_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


def display_path(path):
    """Devuelve una ruta relativa al proyecto para logs sin exponer rutas locales."""
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return os.path.basename(path)


def get_sagittal_slice(volume, sag_axis, idx):
    if sag_axis == 0:
        return volume[idx, :, :]
    if sag_axis == 1:
        return volume[:, idx, :]
    return volume[:, :, idx]


def get_in_plane_spacings(header, sag_axis):
    """
    Extrae los spacings (mm/px) de los dos ejes en el plano del slice sagital.

    NIfTI pixdim[1:4] = (spacing_axis0, spacing_axis1, spacing_axis2).
    El slice sagital elimina sag_axis, quedando los otros dos ejes en el plano.

    Returns:
        (spacing_row, spacing_col): resolución física en mm/px para cada
        dimensión del slice 2-D tal como lo devuelve get_sagittal_slice.
    """
    pixdim = header.get_zooms()          # tupla (sx, sy, sz) en mm/px
    in_plane_axes = [i for i in range(3) if i != sag_axis]
    spacing_row = float(pixdim[in_plane_axes[0]])
    spacing_col = float(pixdim[in_plane_axes[1]])
    return spacing_row, spacing_col


def pad_or_crop_to_fixed(arr, target_h, target_w):
    """
    Lleva un array 2-D a (target_h, target_w) con relleno de ceros centrado
    si es más pequeño, o recorte central si es más grande.
    No deforma ni escala el contenido: garantiza la conservación del spacing.
    """
    h, w = arr.shape
    out = np.zeros((target_h, target_w), dtype=arr.dtype)

    # --- Eje vertical (filas) ---
    if h <= target_h:
        pad_top = (target_h - h) // 2
        src_row_start, src_row_end = 0, h
        dst_row_start, dst_row_end = pad_top, pad_top + h
    else:
        crop_top = (h - target_h) // 2
        src_row_start, src_row_end = crop_top, crop_top + target_h
        dst_row_start, dst_row_end = 0, target_h

    # --- Eje horizontal (columnas) ---
    if w <= target_w:
        pad_left = (target_w - w) // 2
        src_col_start, src_col_end = 0, w
        dst_col_start, dst_col_end = pad_left, pad_left + w
    else:
        crop_left = (w - target_w) // 2
        src_col_start, src_col_end = crop_left, crop_left + target_w
        dst_col_start, dst_col_end = 0, target_w

    out[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
        arr[src_row_start:src_row_end, src_col_start:src_col_end]

    return out


def preprocess_pair(img_slice, seg_slice, header, sag_axis):
    """
    Preprocesa un par (imagen, máscara) de un slice sagital MRI:

    1. Rotación 90° horario + espejo horizontal (convención A-L / P-R).
    2. Normalización Min-Max de la imagen.
    3. Resampling físico a TARGET_SPACING_MM mm/px:
         new_dim = round(dim_px * spacing_mm / TARGET_SPACING_MM)
       - Imagen  → INTER_CUBIC  (interpolación suave para señal continua).
       - Máscara → INTER_NEAREST (preserva bordes binarios sin artefactos).
    4. Padding con ceros o Center Crop al tamaño fijo IMAGE_SIZE × IMAGE_SIZE,
       garantizando que 1 px ≡ TARGET_SPACING_MM mm al entrar a la red.

    Args:
        img_slice : np.ndarray 2-D – slice de imagen MRI en unidades originales.
        seg_slice : np.ndarray 2-D – slice de máscara de segmentación.
        header    : nibabel header – contiene pixdim con los spacings reales.
        sag_axis  : int – eje sagital eliminado al extraer el slice (0, 1 ó 2).

    Returns:
        img_out (np.float32, IMAGE_SIZE×IMAGE_SIZE) – imagen normalizada.
        seg_out (np.float32, IMAGE_SIZE×IMAGE_SIZE) – máscara binarizada.
    """
    # ── 1. Orientación: 90° horario + espejo horizontal ──────────────────────
    img_slice = np.rot90(img_slice, k=3)
    seg_slice = np.rot90(seg_slice, k=3)
    img_slice = np.fliplr(img_slice)
    seg_slice = np.fliplr(seg_slice)

    # ── 2. Normalización Min-Max ──────────────────────────────────────────────
    diff = np.max(img_slice) - np.min(img_slice)
    img_norm = (img_slice - np.min(img_slice)) / (diff if diff != 0 else 1)

    # ── 3. Resampling físico a TARGET_SPACING_MM mm/px ───────────────────────
    # Tras la rotación 90°, las filas del slice corresponden al eje-columna
    # original y las columnas al eje-fila original; los spacings se intercambian.
    spacing_row_orig, spacing_col_orig = get_in_plane_spacings(header, sag_axis)

    # Después de rot90(k=3): filas ← cols originales, cols ← filas originales
    spacing_row = spacing_col_orig
    spacing_col = spacing_row_orig

    h_px, w_px = img_norm.shape

    # Tamaño físico total del slice (mm)
    h_mm = h_px * spacing_row
    w_mm = w_px * spacing_col

    # Nuevas dimensiones a TARGET_SPACING_MM mm/px
    new_h = max(1, round(h_mm / TARGET_SPACING_MM))
    new_w = max(1, round(w_mm / TARGET_SPACING_MM))

    # cv2.resize espera (width, height)
    img_resampled = cv2.resize(
        img_norm.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )
    seg_resampled = cv2.resize(
        seg_slice.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # ── 4. Padding / Center Crop → IMAGE_SIZE × IMAGE_SIZE ───────────────────
    img_out = pad_or_crop_to_fixed(img_resampled, IMAGE_SIZE, IMAGE_SIZE)
    seg_out = pad_or_crop_to_fixed(seg_resampled, IMAGE_SIZE, IMAGE_SIZE)

    # Binarización final de la máscara (robusta ante artefactos de float)
    seg_out = (seg_out > 0).astype(np.float32)

    return img_out.astype(np.float32), seg_out


def preprocess_ultrasound_image(img):
    
    # 1. Volteo vertical para poner el transductor abajo (MRI-style)
    # 0 = vertical flip, 1 = horizontal, -1 = ambos
    img = cv2.flip(img, 0)

    # 2. Normalización Min-Max
    diff = np.max(img) - np.min(img)
    img_norm = (img - np.min(img)) / (diff if diff != 0 else 1)

    # 3. Resize al tamaño estándar (256x256)
    img_res = cv2.resize(img_norm, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    
    return img_res.astype(np.float32)


def universal_processor(base_path, output_path, img_suffix='_t2', mask_suffix='_seg'):
    """
    base_path: Carpeta donde están las subcarpetas de pacientes.
    output_path: Carpeta de destino para los .npy.
    img_suffix: El final del nombre del archivo de imagen (ej: '_t2' o '_mri').
    mask_suffix: El final del nombre del archivo de máscara (ej: '_seg' o '_mask').
    """
    out_img = os.path.join(output_path, 'images_npy')
    out_mask = os.path.join(output_path, 'masks_npy')
    for _, view_folder in SLICE_VIEWS:
        os.makedirs(os.path.join(out_img, view_folder), exist_ok=True)
        os.makedirs(os.path.join(out_mask, view_folder), exist_ok=True)

    # Buscamos todas las subcarpetas de pacientes
    patient_folders = [f for f in glob(os.path.join(base_path, "*")) if os.path.isdir(f)]
    
    for folder in tqdm(patient_folders, desc="Procesando Dataset"):
        patient_id = os.path.basename(folder)
        
        # 1. Validación de existencia: ¿Ya procesamos a este paciente?
        if glob(os.path.join(out_img, "*", f"{patient_id}_*")):
            continue 

        # 2. Búsqueda de archivos
        img_nii = glob(os.path.join(folder, f"*{img_suffix}.nii*"))
        seg_nii = glob(os.path.join(folder, f"*{mask_suffix}.nii*"))

        if not img_nii or not seg_nii:
            continue 

        # 3. Carga con "Escudo" contra archivos corruptos
        try:
            nii_img = nib.load(img_nii[0])
            data_img = nii_img.get_fdata()
            data_seg = nib.load(seg_nii[0]).get_fdata()
        except Exception as e:
            print(f"\n Error en {patient_id}: {e}. Salteando...")
            continue

        # Guardamos el header para acceder al pixdim en preprocess_pair
        header = nii_img.header

        # ── Detectar el eje sagital via affine ───────────────────────────────
        affine = nii_img.affine
        codes  = nib.aff2axcodes(affine)
        # Sagital = dirección Left/Right
        sag_axis = next((i for i, c in enumerate(codes) if c in ("L", "R")), None)
        if sag_axis is None:
            # Fallback: eje con menos slices (típicamente el sagital en pelvis)
            sag_axis = int(np.argmin(data_img.shape))

        n_sag = data_img.shape[sag_axis]

        # ── Detectar el rango sagital con mioma y extraer 3 vistas ──────────
        tissue_indices = [
            i for i in range(n_sag)
            if np.sum(get_sagittal_slice(data_seg, sag_axis, i)) > 0
        ]

        if not tissue_indices:
            continue

        first_idx = tissue_indices[0]
        last_idx = tissue_indices[-1]
        myoma_width = last_idx - first_idx + 1
        mid_idx = (first_idx + last_idx) // 2
        offset = max(1, int(round(myoma_width * 0.25)))

        selected_slices = {
            "mid": mid_idx,
            "left": max(first_idx, mid_idx - offset),
            "right": min(last_idx, mid_idx + offset),
        }

        for suffix, view_folder in SLICE_VIEWS:
            i = selected_slices[suffix]
            img_slice = get_sagittal_slice(data_img, sag_axis, i)
            seg_slice = get_sagittal_slice(data_seg, sag_axis, i)

            # Resampling físico a TARGET_SPACING_MM mm/px + pad/crop a IMAGE_SIZE
            img_res, seg_res = preprocess_pair(img_slice, seg_slice, header, sag_axis)

            # Guardar — formato: {patient_id}_sag_{i}_{vista}
            file_id = f"{patient_id}_sag_{i}_{suffix}"
            np.save(os.path.join(out_img, view_folder, f"{file_id}.npy"), img_res)
            np.save(os.path.join(out_mask, view_folder, f"{file_id}.npy"), seg_res)


def ultrasound_processor(base_path, output_path):
    """
    Procesa ultrasonidos F18/F28/F38 para alinear las vistas con MRI:
      - F28 -> images_npy/1_Anchor/*_mid.npy
      - F18 -> images_npy/2_Lateral/*_left.npy
      - F38 -> images_npy/2_Lateral/*_right.npy

    No genera máscaras para US porque el dominio target es no supervisado.
    """
    out_img = os.path.join(output_path, 'images_npy')
    for _, view_folder in US_VIEW_MAP.values():
        os.makedirs(os.path.join(out_img, view_folder), exist_ok=True)

    image_files = []
    for ext in US_EXTENSIONS:
        image_files.extend(glob(os.path.join(base_path, "**", ext), recursive=True))
    image_files = sorted(set(image_files))
    
    saved = 0
    skipped = 0

    for img_path in tqdm(image_files, desc="Procesando Ultrasonido"):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        view_key = next((view for view in US_VIEW_MAP if view in filename.upper()), None)
        if view_key is None:
            skipped += 1
            continue

        suffix, view_folder = US_VIEW_MAP[view_key]
        file_id = f"{filename}_{suffix}"
        out_file = os.path.join(out_img, view_folder, f"{file_id}.npy")
        if os.path.exists(out_file):
            skipped += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"\n No se pudo leer {img_path}. Salteando...")
            skipped += 1
            continue

        img_res = preprocess_ultrasound_image(img)
        np.save(out_file, img_res)
        saved += 1

    print(f"\nUltrasonido finalizado. Guardados: {saved} | Salteados: {skipped}")
    print(f"Destino: {display_path(out_img)}")


# --- CONFIGURACIÓN: ---
if __name__ == "__main__":
    universal_processor(
        base_path=MRI_BASE_PATH,
        output_path=MRI_OUTPUT_PATH,
        img_suffix=MRI_IMG_SUFFIX,
        mask_suffix=MRI_MASK_SUFFIX,
    )

    ultrasound_processor(
        base_path=US_BASE_PATH,
        output_path=US_OUTPUT_PATH,
    )
