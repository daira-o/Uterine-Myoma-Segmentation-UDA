"""
normalizar_us_08mm.py
=====================
Normaliza el dataset de Ultrasonido a resolución física 0.8 mm/px para que sea
comparable con el MRI procesado previamente.

Para cada imagen:
  1. Determina la profundidad real (mm) según la carpeta de origen.
  2. Calcula la resolución actual: spacing_actual = D_mm / altura_px
  3. Calcula el factor de escala:  factor = spacing_actual / 0.8
  4. Redimensiona con INTER_CUBIC preservando relación de aspecto.
  5. Aplica Center Padding (negro) o Center Crop para llegar a IMAGE_SIZE × IMAGE_SIZE.

Estructura de entrada esperada:
  US_BASE_PATH/
    Escala_10cm/   ← profundidad 100 mm
    Escala_12cm/   ← profundidad 120 mm
    Escala_15cm/   ← profundidad 150 mm
    Escala_16cm/   ← profundidad 160 mm

Estructura de salida generada (espejo):
  US_OUTPUT_PATH/
    Escala_10cm/
    Escala_12cm/
    Escala_15cm/
    Escala_16cm/
"""

import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv


# ── Configuración ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

US_BASE_PATH = Path(
    os.getenv("US_DATA_PATH", str(PROJECT_ROOT / "data" / "Ultrasound"))
)
US_OUTPUT_PATH = Path(
    os.getenv("US_OUTPUT_08MM_PATH", str(PROJECT_ROOT / "US_procesado_08mm"))
)

IMAGE_SIZE = int(os.getenv("PROCESSOR_IMAGE_SIZE", "256"))
TARGET_SPACING_MM = 0.8          # resolución objetivo: 1 px = 0.8 mm de tejido real

# Mapa carpeta → profundidad en mm
DEPTH_MAP: dict[str, float] = {
    "Escala_10cm": 100.0,
    "Escala_12cm": 120.0,
    "Escala_15cm": 150.0,
    "Escala_16cm": 160.0,
}

US_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


# ── Utilidades ────────────────────────────────────────────────────────────────

def display_path(path: Path | str) -> str:
    """Ruta relativa al proyecto para logs sin exponer rutas locales."""
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return os.path.basename(str(path))


def pad_or_crop_center(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Ajusta un array 2-D a (target_h, target_w) sin redimensionar el contenido.

    - Si la imagen es más pequeña → padding de ceros centrado (negro).
    - Si la imagen es más grande  → center crop desde el centro.

    Garantiza que 1 px sigue valiendo TARGET_SPACING_MM mm tras la operación.
    """
    h, w = img.shape
    out = np.zeros((target_h, target_w), dtype=img.dtype)

    # Eje vertical
    if h <= target_h:
        pad_top = (target_h - h) // 2
        src_r0, src_r1 = 0, h
        dst_r0, dst_r1 = pad_top, pad_top + h
    else:
        crop_top = (h - target_h) // 2
        src_r0, src_r1 = crop_top, crop_top + target_h
        dst_r0, dst_r1 = 0, target_h

    # Eje horizontal
    if w <= target_w:
        pad_left = (target_w - w) // 2
        src_c0, src_c1 = 0, w
        dst_c0, dst_c1 = pad_left, pad_left + w
    else:
        crop_left = (w - target_w) // 2
        src_c0, src_c1 = crop_left, crop_left + target_w
        dst_c0, dst_c1 = 0, target_w

    out[dst_r0:dst_r1, dst_c0:dst_c1] = img[src_r0:src_r1, src_c0:src_c1]
    return out


def physical_scale(img: np.ndarray, depth_mm: float) -> np.ndarray:
    """
    Escala la imagen a TARGET_SPACING_MM mm/px a partir de su profundidad real.

    Parámetros
    ----------
    img      : array 2-D (grayscale, float32 en [0, 1]).
    depth_mm : profundidad total del campo ecográfico en mm (eje vertical).

    Proceso
    -------
    spacing_actual = depth_mm / img.shape[0]   # mm por píxel original
    factor_escala  = spacing_actual / TARGET_SPACING_MM
    new_h = round(img.shape[0] * factor_escala)
    new_w = round(img.shape[1] * factor_escala)

    El mismo factor se aplica a ambas dimensiones para no distorsionar la
    relación de aspecto (el ecógrafo preserva píxeles cuadrados en la imagen).
    """
    h_orig, w_orig = img.shape

    spacing_actual_mm_px = depth_mm / h_orig           # resolución actual
    factor_escala = spacing_actual_mm_px / TARGET_SPACING_MM

    new_h = max(1, round(h_orig * factor_escala))
    new_w = max(1, round(w_orig * factor_escala))

    img_scaled = cv2.resize(
        img,
        (new_w, new_h),          # cv2.resize: (width, height)
        interpolation=cv2.INTER_CUBIC,
    )
    return img_scaled


def preprocess_us_image(img_bgr: np.ndarray, depth_mm: float) -> np.ndarray:
    """
    Pipeline completo para una imagen de US:

    1. Convertir a escala de grises.
    2. Volteo vertical (transductor arriba → transductor abajo, estilo MRI).
    3. Normalización Min-Max → float32 en [0, 1].
    4. Resampling físico a TARGET_SPACING_MM mm/px (INTER_CUBIC).
    5. Padding / Center Crop → IMAGE_SIZE × IMAGE_SIZE.

    Returns
    -------
    np.ndarray float32, shape (IMAGE_SIZE, IMAGE_SIZE), valores en [0, 1].
    """
    # 1. Escala de grises
    if img_bgr.ndim == 3:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = img_bgr.copy()

    # 2. Volteo vertical (alineación con convención MRI)
    img = cv2.flip(img, 0)

    # 3. Normalización Min-Max
    img = img.astype(np.float32)
    diff = img.max() - img.min()
    img = (img - img.min()) / (diff if diff != 0 else 1.0)

    # 4. Resampling físico
    img = physical_scale(img, depth_mm)

    # 5. Padding / Center Crop al tamaño fijo
    img = pad_or_crop_center(img, IMAGE_SIZE, IMAGE_SIZE)

    return img


# ── Procesador principal ──────────────────────────────────────────────────────

def normalizar_us(base_path: Path, output_path: Path) -> None:
    """
    Recorre las subcarpetas de `base_path` definidas en DEPTH_MAP,
    procesa cada imagen y guarda el resultado como .npy en la estructura
    espejo dentro de `output_path`.

    Reanudación automática: si el archivo .npy de destino ya existe, se omite.
    """
    # ── Estadísticas globales
    total_saved = 0
    total_skipped = 0
    total_errors = 0

    for folder_name, depth_mm in DEPTH_MAP.items():
        src_folder = base_path / folder_name
        dst_folder = output_path / folder_name

        if not src_folder.exists():
            print(f"[AVISO] Carpeta no encontrada, omitiendo: {display_path(src_folder)}")
            continue

        dst_folder.mkdir(parents=True, exist_ok=True)

        # Recopilar imágenes de la carpeta (sin recursión: estructura plana por escala)
        image_files: list[Path] = []
        for ext in US_EXTENSIONS:
            image_files.extend(src_folder.glob(ext))
        image_files = sorted(set(image_files))

        if not image_files:
            print(f"[AVISO] Sin imágenes en {display_path(src_folder)}")
            continue

        saved = skipped = errors = 0

        for img_path in tqdm(image_files, desc=f"{folder_name} ({depth_mm:.0f} mm)"):
            stem = img_path.stem
            out_file = dst_folder / f"{stem}.npy"

            # Reanudación: saltar si ya fue procesada
            if out_file.exists():
                skipped += 1
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"\n  [ERROR] No se pudo leer: {display_path(img_path)}")
                errors += 1
                continue

            try:
                img_out = preprocess_us_image(img_bgr, depth_mm)
                np.save(str(out_file), img_out)
                saved += 1
            except Exception as exc:
                print(f"\n  [ERROR] {display_path(img_path)}: {exc}")
                errors += 1

        print(
            f"  → Guardadas: {saved:4d} | Saltadas: {skipped:4d} | Errores: {errors:2d}"
            f"  ({display_path(dst_folder)})"
        )
        total_saved  += saved
        total_skipped += skipped
        total_errors  += errors

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print(f"  Guardadas : {total_saved}")
    print(f"  Saltadas  : {total_skipped}")
    print(f"  Errores   : {total_errors}")
    print(f"  Destino   : {display_path(output_path)}")
    print("=" * 60)


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Fuente : {display_path(US_BASE_PATH)}")
    print(f"Destino: {display_path(US_OUTPUT_PATH)}")
    print(f"Target : {TARGET_SPACING_MM} mm/px  →  {IMAGE_SIZE}×{IMAGE_SIZE} px\n")

    normalizar_us(US_BASE_PATH, US_OUTPUT_PATH)
