"""
mri_pipeline.py
===============
Pipeline de preparación de datos MRI (NIfTI → .npy) para entrenamiento de
Attention U-Net 2D con consistencia isométrica estricta (0.8 mm/px) y
aislamiento estadístico absoluto (sin data leakage).

Orden de ejecución:
    1. División física a nivel de paciente (train/val/test)
    2. Canonización de orientación (RAS)
    3. Extracción 2D por el Eje 0 (plano sagital)
    4. Filtrado por área mínima de máscara (≥ 150 px)
    5. Resampling físico a 0.8 mm/px
    6. Center Crop / Padding a 256 × 256
    7. Guardado individual como .npy

Uso:
    python mri_pipeline.py

Variables de entorno (opcional, con valores por defecto):
    MRI_DATA_PATH      → carpeta raíz con subcarpetas de pacientes
    MRI_OUTPUT_PATH    → carpeta de salida para los splits
    NIFTI_IMG_SUFFIX   → sufijo del archivo imagen  (default: _t2)
    NIFTI_MASK_SUFFIX  → sufijo del archivo máscara (default: _seg)
    PROCESSOR_IMAGE_SIZE → lado del tile cuadrado final (default: 256)
"""

from __future__ import annotations

import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 0. Configuración global
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MRI_BASE_PATH: str = os.getenv(
    "MRI_DATA_PATH",
    os.getenv("NIFTI_ROOT", str(PROJECT_ROOT / "data" / "UMD")),
)
MRI_OUTPUT_PATH: str = os.getenv(
    "MRI_OUTPUT_PATH", str(PROJECT_ROOT / "data_ready_RM")
)
IMG_SUFFIX: str = os.getenv("NIFTI_IMG_SUFFIX", "_t2")
MASK_SUFFIX: str = os.getenv("NIFTI_MASK_SUFFIX", "_seg")
IMAGE_SIZE: int = int(os.getenv("PROCESSOR_IMAGE_SIZE", "256"))

TARGET_SPACING_MM: float = 0.8   # resolución isométrica objetivo
MIN_MASK_AREA_PX: int = 150      # umbral mínimo de píxeles útiles en la máscara
RANDOM_STATE: int = 42           # semilla para reproducibilidad de la división

SPLITS: dict[str, float] = {"train": 0.80, "val": 0.10, "test": 0.10}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. División física a nivel de paciente
# ─────────────────────────────────────────────────────────────────────────────

def split_patients(
    base_path: str,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = RANDOM_STATE,
) -> dict[str, list[str]]:
    """
    Divide los directorios de pacientes en train / val / test ANTES de
    extraer ningún corte, garantizando que un mismo paciente no aparezca
    jamás en dos splits distintos.

    Parámetros
    ----------
    base_path : str
        Carpeta raíz que contiene una subcarpeta por paciente.
    val_ratio : float
        Fracción del total asignada a validación (default 0.10).
    test_ratio : float
        Fracción del total asignada a test (default 0.10).
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    dict con claves "train", "val", "test" y listas de rutas absolutas.
    """
    all_folders: list[str] = sorted(
        f for f in glob(os.path.join(base_path, "*")) if os.path.isdir(f)
    )
    if not all_folders:
        raise FileNotFoundError(f"No se encontraron subcarpetas en: {base_path}")

    # Primera división: train vs (val + test)
    holdout_ratio = val_ratio + test_ratio
    train_folders, holdout_folders = train_test_split(
        all_folders,
        test_size=holdout_ratio,
        random_state=random_state,
        shuffle=True,
    )

    # Segunda división: val vs test dentro del holdout
    relative_test_ratio = test_ratio / holdout_ratio
    val_folders, test_folders = train_test_split(
        holdout_folders,
        test_size=relative_test_ratio,
        random_state=random_state,
        shuffle=True,
    )

    splits = {"train": train_folders, "val": val_folders, "test": test_folders}

    log.info(
        "División de pacientes — train: %d | val: %d | test: %d",
        len(train_folders), len(val_folders), len(test_folders),
    )
    return splits


def save_split_manifest(splits: dict[str, list[str]], output_path: str) -> None:
    """
    Exporta un JSON con los IDs de paciente asignados a cada split para
    trazabilidad y reproducibilidad del experimento.

    Parámetros
    ----------
    splits : dict
        Resultado de split_patients().
    output_path : str
        Directorio raíz de salida; el JSON se guarda en su interior.
    """
    manifest = {
        split_name: [os.path.basename(p) for p in paths]
        for split_name, paths in splits.items()
    }
    manifest_path = os.path.join(output_path, "patient_splits.json")
    os.makedirs(output_path, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    log.info("Manifiesto de splits guardado en: %s", manifest_path)


def build_output_dirs(output_path: str) -> dict[str, dict[str, str]]:
    """
    Crea la estructura de carpetas de salida:
        <output_path>/<split>/images/
        <output_path>/<split>/masks/

    Retorna un dict anidado con las rutas absolutas para uso posterior.
    """
    paths: dict[str, dict[str, str]] = {}
    for split_name in SPLITS:
        img_dir = os.path.join(output_path, split_name, "images")
        msk_dir = os.path.join(output_path, split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)
        paths[split_name] = {"images": img_dir, "masks": msk_dir}
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# 2. Carga y canonización de orientación
# ─────────────────────────────────────────────────────────────────────────────

def load_canonical_nifti(
    nifti_path: str,
) -> Optional[tuple[np.ndarray, nib.nifti1.Nifti1Header]]:
    """
    Carga un archivo NIfTI y lo convierte a la orientación canónica RAS+
    mediante `nib.as_closest_canonical`.

    Por qué RAS+ obligatorio
    ------------------------
    Los NIfTIs del mundo real pueden almacenarse en cualquier orientación
    (LAS, PIR, etc.) dependiendo del escáner y el centro hospitalario.
    Aplicar `as_closest_canonical` garantiza que el eje 0 siempre apunte
    a Right, el eje 1 a Anterior y el eje 2 a Superior,
    sin importar el origen del archivo.  Esto hace que el código sea
    completamente independiente del equipo de adquisición y elimina
    la necesidad de rotaciones manuales frágiles (rot90, fliplr), que son
    propensas a errores silenciosos cuando la orientación original varía.

    Parámetros
    ----------
    nifti_path : str
        Ruta al archivo .nii o .nii.gz.

    Retorna
    -------
    Tupla (array 3-D float32, header del NIfTI canonizado) o None si falla.
    """
    try:
        img = nib.load(nifti_path)
        img_canonical = nib.as_closest_canonical(img)   # ← orientación RAS+
        data = img_canonical.get_fdata(dtype=np.float32)
        return data, img_canonical.header
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo cargar '%s': %s. Saltando...", nifti_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3 + 4. Extracción y filtrado de cortes 2D
# ─────────────────────────────────────────────────────────────────────────────

SAG_AXIS: int = 0  # Eje sagital tras canonizacion RAS+


def extract_valid_slices(
    vol_img: np.ndarray,
    vol_seg: np.ndarray,
    min_mask_area: int = MIN_MASK_AREA_PX,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Extrae todos los cortes 2D a lo largo del Eje 0 (SAG_AXIS) y retiene
    únicamente los pares cuya máscara supera el umbral mínimo de área.

    Por qué Eje 0
    ----------------------------
    Tras la canonización RAS+, iterar sobre el eje 0 produce cortes 2D
    `vol[i, :, :]`. Las dimensiones del plano resultante corresponden a los
    ejes físicos 1 y 2, que son los spacings usados para el resampling.

    Parámetros
    ----------
    vol_img : np.ndarray  (H, W, D)
        Volumen de imagen MRI canonizado.
    vol_seg : np.ndarray  (H, W, D)
        Volumen de máscara de segmentación canonizado.
    min_mask_area : int
        Número mínimo de píxeles activos para conservar el corte.

    Retorna
    -------
    Lista de tuplas (índice_corte, slice_imagen_2D, slice_máscara_2D).
    """
    valid: list[tuple[int, np.ndarray, np.ndarray]] = []
    n_slices = vol_img.shape[SAG_AXIS]

    for i in range(n_slices):
        img_slice = vol_img[i, :, :]   # Eje 0 -> vol[i, :, :]
        seg_slice = vol_seg[i, :, :]

        mask_area = int(np.sum(seg_slice > 0))
        if mask_area >= min_mask_area:
            valid.append((i, img_slice, seg_slice))

    return valid


# ─────────────────────────────────────────────────────────────────────────────
# 5. Resampling físico a TARGET_SPACING_MM
# ─────────────────────────────────────────────────────────────────────────────

def get_inplane_spacings(
    header: nib.nifti1.Nifti1Header,
) -> tuple[float, float]:
    """
    Extrae los spacings (mm/px) de los ejes en el plano del corte sagital
    (Eje 1 y Eje 2 del volumen canonizado) directamente del header NIfTI.

    Retorna
    -------
    (spacing_row, spacing_col) en mm/px.
    """
    pixdim = header.get_zooms()          # (sx, sy, sz) mm/px tras canonizacion
    # Los cortes son vol[i, :, :] -> dimensiones en el plano son ejes 1 y 2.
    return float(pixdim[1]), float(pixdim[2])


def resample_slice(
    img_slice: np.ndarray,
    seg_slice: np.ndarray,
    spacing_row: float,
    spacing_col: float,
    target_spacing: float = TARGET_SPACING_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Redimensiona un par imagen/máscara para que 1 px ≡ target_spacing mm.

    Cálculo de nuevas dimensiones
    ------------------------------
        tamaño_físico_mm = dim_px * spacing_mm_px
        nueva_dim_px     = round(tamaño_físico_mm / target_spacing)

    Elección de interpolación
    -------------------------
    - Imagen  → INTER_CUBIC (bicúbico):
        La señal MRI es continua y suave.  La interpolación bicúbica preserva
        los gradientes de intensidad sin introducir aliasing perceptible,
        lo que es crítico para que la red aprenda texturas reales.

    - Máscara → INTER_NEAREST (vecino más cercano):
        La máscara es un mapa binario (0/1).  Cualquier interpolación
        que genere valores intermedios (0.3, 0.7…) corrompería las etiquetas
        al redondear o binarizar después.  INTER_NEAREST garantiza bordes
        nítidos y clases puras, preservando la integridad del ground truth.

    Parámetros
    ----------
    img_slice : np.ndarray 2-D   – corte de imagen normalizado.
    seg_slice : np.ndarray 2-D   – corte de máscara.
    spacing_row : float          – resolución física original del eje de filas (mm/px).
    spacing_col : float          – resolución física original del eje de columnas (mm/px).
    target_spacing : float       – resolución física objetivo (mm/px).

    Retorna
    -------
    (img_resampled, seg_resampled) como float32.
    """
    h_px, w_px = img_slice.shape

    # Tamaño físico total en mm
    h_mm = h_px * spacing_row
    w_mm = w_px * spacing_col

    # Nuevas dimensiones a la resolución objetivo
    new_h = max(1, round(h_mm / target_spacing))
    new_w = max(1, round(w_mm / target_spacing))

    # cv2.resize recibe (width, height)
    img_res = cv2.resize(
        img_slice.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,      # suaviza sin aliasing
    )
    seg_res = cv2.resize(
        seg_slice.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,    # preserva etiquetas binarias
    )
    return img_res, seg_res


# ─────────────────────────────────────────────────────────────────────────────
# 6. Center Crop / Padding a IMAGE_SIZE × IMAGE_SIZE
# ─────────────────────────────────────────────────────────────────────────────

def pad_or_crop_center(
    arr: np.ndarray,
    target_h: int = IMAGE_SIZE,
    target_w: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Lleva un array 2-D a (target_h, target_w) sin deformar ni escalar
    el contenido:

    - Si la dimensión es MENOR que el objetivo → padding de ceros centrado.
    - Si la dimensión es MAYOR que el objetivo → recorte central.

    Esta operación garantiza que 1 px sigue equivaliendo a TARGET_SPACING_MM mm
    después del resampling, ya que NO altera la resolución física.

    La misma función se aplica a imagen y máscara con idénticos parámetros,
    asegurando alineación perfecta entre ambos arrays.

    Parámetros
    ----------
    arr : np.ndarray 2-D
        Array a ajustar.
    target_h, target_w : int
        Dimensiones finales deseadas.

    Retorna
    -------
    np.ndarray de forma (target_h, target_w) con el mismo dtype que arr.
    """
    h, w = arr.shape
    out = np.zeros((target_h, target_w), dtype=arr.dtype)

    # ── Eje vertical (filas) ─────────────────────────────────────────────────
    if h <= target_h:
        pad_top = (target_h - h) // 2
        src_r0, src_r1 = 0, h
        dst_r0, dst_r1 = pad_top, pad_top + h
    else:
        crop_top = (h - target_h) // 2
        src_r0, src_r1 = crop_top, crop_top + target_h
        dst_r0, dst_r1 = 0, target_h

    # ── Eje horizontal (columnas) ────────────────────────────────────────────
    if w <= target_w:
        pad_left = (target_w - w) // 2
        src_c0, src_c1 = 0, w
        dst_c0, dst_c1 = pad_left, pad_left + w
    else:
        crop_left = (w - target_w) // 2
        src_c0, src_c1 = crop_left, crop_left + target_w
        dst_c0, dst_c1 = 0, target_w

    out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. Normalización Min-Max de la imagen
# ─────────────────────────────────────────────────────────────────────────────

def normalize_minmax(img: np.ndarray) -> np.ndarray:
    """
    Normaliza un array al rango [0, 1] usando Min-Max scaling.
    Si el rango es cero (corte completamente negro), retorna el array tal cual.

    Parámetros
    ----------
    img : np.ndarray   – imagen MRI en unidades de intensidad originales.

    Retorna
    -------
    np.ndarray float32 con valores en [0.0, 1.0].
    """
    vmin, vmax = float(img.min()), float(img.max())
    diff = vmax - vmin
    if diff == 0:
        return img.astype(np.float32)
    return ((img - vmin) / diff).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Clase principal del pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MRIPipelineProcessor:
    """
    Orquesta el pipeline completo de preparación de datos MRI NIfTI → .npy.

    Orden de ejecución:
        1. División física a nivel de paciente  (sin data leakage)
        2. Creación de directorios de salida
        3. Exportación del manifiesto JSON
        4. Para cada split y cada paciente:
            a. Carga y canonización RAS+
            b. Extracción y filtrado de cortes 2D
            c. Normalización Min-Max
            d. Resampling físico a 0.8 mm/px
            e. Center Crop / Padding a 256×256
            f. Guardado como .npy individual

    Parámetros
    ----------
    base_path : str
        Carpeta raíz con subcarpetas de pacientes.
    output_path : str
        Carpeta raíz de salida para los splits.
    img_suffix : str
        Sufijo del archivo imagen (e.g. '_t2').
    mask_suffix : str
        Sufijo del archivo máscara (e.g. '_seg').
    image_size : int
        Lado del tile cuadrado final en píxeles.
    target_spacing : float
        Resolución física objetivo en mm/px.
    min_mask_area : int
        Área mínima de máscara en píxeles para conservar un corte.
    """

    def __init__(
        self,
        base_path: str = MRI_BASE_PATH,
        output_path: str = MRI_OUTPUT_PATH,
        img_suffix: str = IMG_SUFFIX,
        mask_suffix: str = MASK_SUFFIX,
        image_size: int = IMAGE_SIZE,
        target_spacing: float = TARGET_SPACING_MM,
        min_mask_area: int = MIN_MASK_AREA_PX,
    ) -> None:
        self.base_path = base_path
        self.output_path = output_path
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.image_size = image_size
        self.target_spacing = target_spacing
        self.min_mask_area = min_mask_area

    # ── Métodos auxiliares internos ──────────────────────────────────────────

    def _find_nifti(self, folder: str, suffix: str) -> Optional[str]:
        """
        Busca el primer archivo .nii o .nii.gz cuyo nombre termina en `suffix`.
        Retorna la ruta o None si no existe.
        """
        matches = glob(os.path.join(folder, f"*{suffix}.nii*"))
        return matches[0] if matches else None

    def _process_patient(
        self,
        folder: str,
        out_dirs: dict[str, str],
    ) -> int:
        """
        Procesa todos los cortes válidos de un único paciente y los guarda.

        Parámetros
        ----------
        folder : str
            Ruta a la carpeta del paciente.
        out_dirs : dict
            Sub-diccionario {'images': ruta, 'masks': ruta} para el split.

        Retorna
        -------
        Número de cortes guardados (0 si el paciente fue saltado).
        """
        patient_id = os.path.basename(folder)

        # ── Búsqueda de archivos NIfTI ────────────────────────────────────
        img_path = self._find_nifti(folder, self.img_suffix)
        seg_path = self._find_nifti(folder, self.mask_suffix)

        if img_path is None or seg_path is None:
            log.warning("Paciente '%s': falta imagen o máscara. Saltando.", patient_id)
            return 0

        # ── Carga y canonización RAS+ ─────────────────────────────────────
        result_img = load_canonical_nifti(img_path)
        result_seg = load_canonical_nifti(seg_path)

        if result_img is None or result_seg is None:
            return 0  # error ya logueado dentro de load_canonical_nifti

        vol_img, header = result_img
        vol_seg, _ = result_seg

        # Verificación de consistencia de dimensiones
        if vol_img.shape != vol_seg.shape:
            log.warning(
                "Paciente '%s': imagen %s y máscara %s tienen formas distintas. Saltando.",
                patient_id, vol_img.shape, vol_seg.shape,
            )
            return 0

        # ── Spacings físicos en el plano (ejes 1 y 2 tras canonización) ───
        spacing_row, spacing_col = get_inplane_spacings(header)

        # ── Extracción y filtrado de cortes 2D ────────────────────────────
        valid_slices = extract_valid_slices(vol_img, vol_seg, self.min_mask_area)

        saved_count = 0
        for slice_idx, img_slice, seg_slice in valid_slices:

            # ── Normalización Min-Max ─────────────────────────────────────
            img_norm = normalize_minmax(img_slice)

            # ── Resampling físico a TARGET_SPACING_MM mm/px ───────────────
            img_res, seg_res = resample_slice(
                img_norm, seg_slice,
                spacing_row, spacing_col,
                self.target_spacing,
            )

            # ── Center Crop / Padding a IMAGE_SIZE × IMAGE_SIZE ───────────
            img_out = pad_or_crop_center(img_res, self.image_size, self.image_size)
            seg_out = pad_or_crop_center(seg_res, self.image_size, self.image_size)

            # Binarización robusta de la máscara (elimina artefactos float)
            seg_out = (seg_out > 0).astype(np.float32)

            # ── Guardado individual como .npy ─────────────────────────────
            file_id = f"{patient_id}_sag_{slice_idx}"
            np.save(
                os.path.join(out_dirs["images"], f"{file_id}.npy"),
                img_out,
            )
            np.save(
                os.path.join(out_dirs["masks"], f"{file_id}.npy"),
                seg_out,
            )
            saved_count += 1

        return saved_count

    # ── Método público principal ─────────────────────────────────────────────

    def run(self) -> None:
        """
        Ejecuta el pipeline completo en el orden definido.
        Loguea estadísticas de cortes guardados por split al finalizar.
        """
        log.info("═" * 60)
        log.info("  MRI Pipeline — Inicio de procesamiento")
        log.info("  Base path : %s", self.base_path)
        log.info("  Output    : %s", self.output_path)
        log.info("  Spacing   : %.1f mm/px  |  Tile: %d px", self.target_spacing, self.image_size)
        log.info("═" * 60)

        # ── Paso 1: División de pacientes ────────────────────────────────────
        splits = split_patients(
            self.base_path,
            val_ratio=SPLITS["val"],
            test_ratio=SPLITS["test"],
        )

        # ── Paso 2: Creación de directorios ──────────────────────────────────
        out_dirs = build_output_dirs(self.output_path)

        # ── Paso 3: Exportar manifiesto JSON ─────────────────────────────────
        save_split_manifest(splits, self.output_path)

        # ── Pasos 4–7: Procesar cada split ───────────────────────────────────
        total_stats: dict[str, int] = {}

        for split_name, folders in splits.items():
            split_slices = 0
            desc = f"[{split_name.upper():5s}] Procesando pacientes"

            for folder in tqdm(folders, desc=desc, unit="paciente"):
                saved = self._process_patient(folder, out_dirs[split_name])
                split_slices += saved

            total_stats[split_name] = split_slices
            log.info("Split %-5s → %d cortes guardados", split_name, split_slices)

        # ── Resumen final ────────────────────────────────────────────────────
        log.info("─" * 60)
        log.info("  RESUMEN FINAL")
        for split_name, count in total_stats.items():
            log.info("    %-6s : %d cortes .npy", split_name, count)
        log.info("  Total   : %d cortes .npy", sum(total_stats.values()))
        log.info("═" * 60)
        log.info("  Pipeline completado. Datos listos en: %s", self.output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = MRIPipelineProcessor(
        base_path=MRI_BASE_PATH,
        output_path=MRI_OUTPUT_PATH,
        img_suffix=IMG_SUFFIX,
        mask_suffix=MASK_SUFFIX,
        image_size=IMAGE_SIZE,
        target_spacing=TARGET_SPACING_MM,
        min_mask_area=MIN_MASK_AREA_PX,
    )
    pipeline.run()
