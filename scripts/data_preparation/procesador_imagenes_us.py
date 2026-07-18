"""
Normalize ultrasound images to 0.8 mm/px.

This legacy utility makes ultrasound images physically comparable with the
previously processed MRI slices.

For each image:
  1. Infer the real field depth in millimeters from the source folder.
  2. Compute current spacing: current_spacing = depth_mm / image_height_px.
  3. Compute scale factor: scale = current_spacing / 0.8.
  4. Resize with INTER_CUBIC while preserving aspect ratio.
  5. Center pad or center crop to IMAGE_SIZE x IMAGE_SIZE.

Expected input structure:
  US_BASE_PATH/
    Escala_10cm/   depth 100 mm
    Escala_12cm/   depth 120 mm
    Escala_15cm/   depth 150 mm
    Escala_16cm/   depth 160 mm

Generated output mirrors the same folder structure under US_OUTPUT_PATH.
"""

import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

US_BASE_PATH = Path(
    os.getenv("US_DATA_PATH", str(PROJECT_ROOT / "data" / "Ultrasound"))
)
US_OUTPUT_PATH = Path(
    os.getenv("US_OUTPUT_08MM_PATH", str(PROJECT_ROOT / "US_procesado_08mm"))
)

IMAGE_SIZE = int(os.getenv("PROCESSOR_IMAGE_SIZE", "256"))
TARGET_SPACING_MM = 0.8  # Target resolution: 1 px = 0.8 mm of real tissue.

DEPTH_MAP: dict[str, float] = {
    "Escala_10cm": 100.0,
    "Escala_12cm": 120.0,
    "Escala_15cm": 150.0,
    "Escala_16cm": 160.0,
}

US_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


def display_path(path: Path | str) -> str:
    """Return a project-relative path for logs without exposing local roots."""
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return os.path.basename(str(path))


def pad_or_crop_center(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Fit a 2-D array to `(target_h, target_w)` without scaling content.

    Smaller images are centered with zero padding. Larger images are center
    cropped. Because this step does not rescale the image, the physical
    resolution remains TARGET_SPACING_MM mm/px.
    """
    h, w = img.shape
    out = np.zeros((target_h, target_w), dtype=img.dtype)

    if h <= target_h:
        pad_top = (target_h - h) // 2
        src_r0, src_r1 = 0, h
        dst_r0, dst_r1 = pad_top, pad_top + h
    else:
        crop_top = (h - target_h) // 2
        src_r0, src_r1 = crop_top, crop_top + target_h
        dst_r0, dst_r1 = 0, target_h

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
    Scale an ultrasound image to TARGET_SPACING_MM mm/px.

    The vertical field depth gives the original physical spacing:
        current_spacing = depth_mm / img.shape[0]
        scale_factor = current_spacing / TARGET_SPACING_MM

    The same factor is applied to height and width to avoid aspect-ratio
    distortion.
    """
    h_orig, w_orig = img.shape

    spacing_actual_mm_px = depth_mm / h_orig
    factor_escala = spacing_actual_mm_px / TARGET_SPACING_MM

    new_h = max(1, round(h_orig * factor_escala))
    new_w = max(1, round(w_orig * factor_escala))

    img_scaled = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )
    return img_scaled


def preprocess_us_image(img_bgr: np.ndarray, depth_mm: float) -> np.ndarray:
    """
    Run the complete preprocessing path for one ultrasound image.

    Steps:
        1. Convert to grayscale.
        2. Flip vertically to align ultrasound orientation with MRI convention.
        3. Min-max normalize to float32 in [0, 1].
        4. Resample physically to TARGET_SPACING_MM mm/px.
        5. Center pad or crop to IMAGE_SIZE x IMAGE_SIZE.
    """
    if img_bgr.ndim == 3:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = img_bgr.copy()

    # Align ultrasound orientation with the convention used by the MRI pipeline.
    img = cv2.flip(img, 0)

    img = img.astype(np.float32)
    diff = img.max() - img.min()
    img = (img - img.min()) / (diff if diff != 0 else 1.0)

    img = physical_scale(img, depth_mm)
    img = pad_or_crop_center(img, IMAGE_SIZE, IMAGE_SIZE)

    return img


def normalizar_us(base_path: Path, output_path: Path) -> None:
    """
    Process folders listed in DEPTH_MAP and save `.npy` images.

    Resume behavior: if the destination `.npy` already exists, the image is
    skipped instead of recomputed.
    """
    total_saved = 0
    total_skipped = 0
    total_errors = 0

    for folder_name, depth_mm in DEPTH_MAP.items():
        src_folder = base_path / folder_name
        dst_folder = output_path / folder_name

        if not src_folder.exists():
            print(f"[WARN] Folder not found, skipping: {display_path(src_folder)}")
            continue

        dst_folder.mkdir(parents=True, exist_ok=True)

        image_files: list[Path] = []
        for ext in US_EXTENSIONS:
            image_files.extend(src_folder.glob(ext))
        image_files = sorted(set(image_files))

        if not image_files:
            print(f"[WARN] No images found in {display_path(src_folder)}")
            continue

        saved = skipped = errors = 0

        for img_path in tqdm(image_files, desc=f"{folder_name} ({depth_mm:.0f} mm)"):
            stem = img_path.stem
            out_file = dst_folder / f"{stem}.npy"

            if out_file.exists():
                skipped += 1
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"\n  [ERROR] Could not read: {display_path(img_path)}")
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
            f"  -> Saved: {saved:4d} | Skipped: {skipped:4d} | Errors: {errors:2d}"
            f"  ({display_path(dst_folder)})"
        )
        total_saved += saved
        total_skipped += skipped
        total_errors += errors

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print(f"  Saved   : {total_saved}")
    print(f"  Skipped : {total_skipped}")
    print(f"  Errors  : {total_errors}")
    print(f"  Output  : {display_path(output_path)}")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Source : {display_path(US_BASE_PATH)}")
    print(f"Output : {display_path(US_OUTPUT_PATH)}")
    print(f"Target : {TARGET_SPACING_MM} mm/px  ->  {IMAGE_SIZE}x{IMAGE_SIZE} px\n")

    normalizar_us(US_BASE_PATH, US_OUTPUT_PATH)
