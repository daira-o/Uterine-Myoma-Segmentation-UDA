import numpy as np
import os
import cv2
from glob import glob
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_US_DATA_PATH = os.getenv("US_DATA_PATH", str(PROJECT_ROOT / "data" / "Ultrasound"))
DEFAULT_US_OUTPUT_PATH = os.getenv("US_OUTPUT_PATH", str(PROJECT_ROOT / "data_ready_US"))


def _display_path(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return os.path.basename(path)
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _crop_fixed(img: np.ndarray,
                top: float = 0.08,
                bottom: float = 0.10,
                left: float = 0.05,
                right: float = 0.10) -> np.ndarray:
    """
    Crop asimétrico fijo que respeta la geometría real de las imágenes US:
 
      - Superior  8 %  : elimina fecha/hora y texto de equipo (barra superior).
      - Inferior 10 %  : elimina medidas y escala de profundidad (cm).
      - Izquierdo  5 %  : elimina parámetros técnicos (frecuencia, modo, etc.).
      - Derecho   10 %  : elimina la escala de grises lateral (barra de ganancia).
 
    Por qué asimétrico y no uniforme:
      El borde izquierdo tiene menos artefacto que el derecho (la escala lateral
      ocupa más espacio). Un crop uniforme del 10 % recortaría tejido útil
      innecesariamente en el lado izquierdo.
    """
    h, w = img.shape[:2]
    r0 = max(0, int(h * top))
    r1 = min(h, h - int(h * bottom))
    c0 = max(0, int(w * left))
    c1 = min(w, w - int(w * right))
    return img[r0:r1, c0:c1]
 
 
def _inpaint_annotations(img: np.ndarray) -> np.ndarray:
    """
    Elimina anotaciones de texto y líneas de medición con umbral 200.
 
    Sin máscara de cono: opera sobre toda la imagen.
    El umbral 200 (vs 255) captura blancos degradados por compresión JPG
    y los grises-claros de las líneas de medición con flechas (~200-239).
 
    Se aplica una dilatación de 1px sobre la máscara antes del inpainting
    para capturar el halo de 1-2px que rodea cada anotación.
    """
    mask = (img >= 200).astype(np.uint8) * 255
 
    if mask.sum() > 0:
        # Dilatar la máscara para cubrir halos de borde de las anotaciones
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, k, iterations=1)
        img = cv2.inpaint(img, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
 
    return img
 
 
def _normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalización en dos etapas:
 
    Etapa 1 — CLAHE (Contrast Limited Adaptive Histogram Equalization):
      Mejora el contraste local del tejido de forma adaptativa antes de
      normalizar. Parámetros conservadores (clipLimit=2.0, tileGrid=8×8)
      para no amplificar el ruido speckle del US.
 
    Etapa 2 — Percentil 2–98 + clip:
      Calculado sobre todos los píxeles no-negros (> 5) para no sesgar
      el rango dinámico con el fondo negro residual post-crop.
      np.clip satura cualquier resto de ruido blanco a 1.0.
    """
    # CLAHE sobre imagen uint8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
 
    # Percentiles solo sobre píxeles de tejido (excluir negro residual)
    tissue = img_clahe[img_clahe > 5].astype(np.float32)
    if tissue.size < 100:
        tissue = img_clahe.astype(np.float32).ravel()
 
    p2  = np.percentile(tissue, 2)
    p98 = np.percentile(tissue, 98)
    denom = float(p98 - p2) if (p98 - p2) != 0 else 1.0
 
    img_norm = np.clip((img_clahe.astype(np.float32) - p2) / denom, 0.0, 1.0)
 
    return img_norm
 
 
# ── Procesador principal ──────────────────────────────────────────────────────
 
def us_nosano_processor(base_path, output_path):
    """
    Procesador legacy para el dataset de Ultrasonido (US) de Mendeley.
    Solo contiene casos No Sanos; no utiliza máscaras de segmentación.

    Nota:
      El pipeline principal actual vive en procesador_imagenes.py, que organiza
      F18/F28/F38 en vistas alineadas con MRI. Mantener este script solo si se
      necesita el preprocesamiento historico con crop, inpainting y CLAHE.
 
    base_path:   Carpeta raíz donde están las imágenes JPG/PNG del dataset.
    output_path: Carpeta destino; se creará la estructura US_NO_SANO/images_npy.
 
    Pipeline por imagen:
      1. Carga en escala de grises.
      2. Crop asimétrico fijo (top 8%, bottom 10%, left 5%, right 10%).
      3. Inpainting con umbral 200 + dilatación de máscara 1px + radio 6.
      4. CLAHE (clipLimit 2.0, tileGrid 8×8) + normalización percentil 2–98
         calculada solo sobre píxeles > 5 (excluye negro residual post-crop).
      5. Resize a 256 × 256 (INTER_AREA).
      6. Guardado como float32 con prefijo 'us_nosano_'.
 
    Por qué se abandonó la detección automática del cono:
      Las imágenes del dataset Mendeley presentan conos pegados al borde sin
      marco negro bien definido. Tanto Otsu+morfología como flood-fill desde
      esquinas fallan de forma variable (máscara desbordada o cono completamente
      negro). El crop fijo asimétrico es predecible, reproducible y suficiente
      para eliminar todos los artefactos periféricos observados en validación.
    """
 
    out_img = os.path.join(output_path, "US_NO_SANO", "images_npy")
    os.makedirs(out_img, exist_ok=True)
 
    # Recolectar imágenes (JPG y PNG, recursivo)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(base_path, "**", ext), recursive=True))
        image_files.extend(glob(os.path.join(base_path, ext)))
 
    seen = set()
    image_files = [p for p in image_files if not (p in seen or seen.add(p))]
 
    if not image_files:
        print(f"No se encontraron imágenes JPG/PNG en: {_display_path(base_path)}")
        return
 
    print(f"Imágenes encontradas: {len(image_files)}")
 
    saved   = 0
    skipped = 0
 
    for img_path in tqdm(image_files, desc="Procesando US No Sano"):
 
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_file = os.path.join(out_img, f"us_nosano_{img_name}.npy")
 
        if os.path.exists(out_file):
            skipped += 1
            continue
 
        try:
            # 1. Carga en escala de grises
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"\n  Advertencia: no se pudo leer '{img_path}'. Salteando...")
                skipped += 1
                continue
 
            # 2. Crop asimétrico fijo
            img = _crop_fixed(img)
 
            # 3. Inpainting: umbral 200, dilatación 1px, radio 6
            img = _inpaint_annotations(img)
 
            # 4. CLAHE + normalización percentil 2–98 sobre tejido real
            img_norm = _normalize(img)
 
            # 5. Resize a 256 × 256
            img_res = cv2.resize(img_norm, (256, 256), interpolation=cv2.INTER_AREA)
 
            # 6. Guardar
            np.save(out_file, img_res.astype(np.float32))
            saved += 1
 
        except Exception as e:
            print(f"\n  Error procesando '{img_path}': {e}. Salteando...")
            skipped += 1
            continue
 
    print(f"\nProcesamiento finalizado.")
    print(f"  Guardados : {saved}")
    print(f"  Salteados : {skipped}")
    print(f"  Destino   : {_display_path(out_img)}")
 
# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    us_nosano_processor(
        base_path=DEFAULT_US_DATA_PATH,
        output_path=DEFAULT_US_OUTPUT_PATH,
    )
