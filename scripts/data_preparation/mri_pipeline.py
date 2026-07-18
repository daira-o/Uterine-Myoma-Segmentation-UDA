"""
MRI preprocessing pipeline for Attention U-Net training.

This script converts patient-level NIfTI volumes into 2-D `.npy` slices with
strict physical consistency: 0.8 mm/px spacing and fixed 256 x 256 tiles.
Patient-level train/validation/test splitting is performed before slice
extraction to prevent data leakage.

Execution order:
    1. Split patient folders into train/validation/test sets.
    2. Canonicalize NIfTI orientation to RAS+.
    3. Extract 2-D slices along axis 0, corresponding to sagittal slices after
       canonicalization.
    4. Keep slices whose mask area is at least 150 pixels.
    5. Resample image and mask to 0.8 mm/px.
    6. Center crop or pad each slice to 256 x 256.
    7. Save each image/mask pair as `.npy`.

Usage:
    python scripts/data_preparation/mri_pipeline.py

Optional environment variables:
    MRI_DATA_PATH          Root folder with one subfolder per patient.
    MRI_OUTPUT_PATH        Output folder for processed splits.
    NIFTI_IMG_SUFFIX       Image filename suffix, default `_t2`.
    NIFTI_MASK_SUFFIX      Mask filename suffix, default `_seg`.
    PROCESSOR_IMAGE_SIZE   Final square tile size, default 256.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

TARGET_SPACING_MM: float = 0.8
MIN_MASK_AREA_PX: int = 150
RANDOM_STATE: int = 42

SPLITS: dict[str, float] = {"train": 0.80, "val": 0.10, "test": 0.10}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def split_patients(
    base_path: str,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = RANDOM_STATE,
) -> dict[str, list[str]]:
    """
    Split patient folders before extracting any slices.

    This guarantees that slices from the same patient can never appear in more
    than one split, preserving statistical independence between train,
    validation, and test sets.
    """
    all_folders: list[str] = sorted(
        f for f in glob(os.path.join(base_path, "*")) if os.path.isdir(f)
    )
    if not all_folders:
        raise FileNotFoundError(f"No patient subfolders were found in: {base_path}")

    holdout_ratio = val_ratio + test_ratio
    train_folders, holdout_folders = train_test_split(
        all_folders,
        test_size=holdout_ratio,
        random_state=random_state,
        shuffle=True,
    )

    relative_test_ratio = test_ratio / holdout_ratio
    val_folders, test_folders = train_test_split(
        holdout_folders,
        test_size=relative_test_ratio,
        random_state=random_state,
        shuffle=True,
    )

    splits = {"train": train_folders, "val": val_folders, "test": test_folders}

    log.info(
        "Patient split - train: %d | val: %d | test: %d",
        len(train_folders), len(val_folders), len(test_folders),
    )
    return splits


def save_split_manifest(splits: dict[str, list[str]], output_path: str) -> None:
    """Save patient IDs assigned to each split for traceability."""
    manifest = {
        split_name: [os.path.basename(p) for p in paths]
        for split_name, paths in splits.items()
    }
    manifest_path = os.path.join(output_path, "patient_splits.json")
    os.makedirs(output_path, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    log.info("Saved split manifest to: %s", manifest_path)


def build_output_dirs(output_path: str) -> dict[str, dict[str, str]]:
    """
    Create the output directory structure:
        <output_path>/<split>/images/
        <output_path>/<split>/masks/
    """
    paths: dict[str, dict[str, str]] = {}
    for split_name in SPLITS:
        img_dir = os.path.join(output_path, split_name, "images")
        msk_dir = os.path.join(output_path, split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)
        paths[split_name] = {"images": img_dir, "masks": msk_dir}
    return paths


def load_canonical_nifti(
    nifti_path: str,
) -> Optional[tuple[np.ndarray, nib.nifti1.Nifti1Header]]:
    """
    Load a NIfTI file and convert it to canonical RAS+ orientation.

    Real-world NIfTI files may be stored in different orientations depending on
    scanner and institution. `nib.as_closest_canonical` makes axis 0 point to
    Right, axis 1 to Anterior, and axis 2 to Superior. That removes the need for
    fragile manual rotations or flips, which can silently fail when the original
    orientation changes.

    Returns:
        `(data, header)` for the canonicalized image, or `None` if loading fails.
    """
    try:
        img = nib.load(nifti_path)
        img_canonical = nib.as_closest_canonical(img)
        data = img_canonical.get_fdata(dtype=np.float32)
        return data, img_canonical.header
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load '%s': %s. Skipping...", nifti_path, exc)
        return None


SAG_AXIS: int = 0  # Sagittal axis after RAS+ canonicalization.


def extract_valid_slices(
    vol_img: np.ndarray,
    vol_seg: np.ndarray,
    min_mask_area: int = MIN_MASK_AREA_PX,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Extract 2-D sagittal slices and keep only slices with sufficient mask area.

    After RAS+ canonicalization, iterating over axis 0 yields slices of the form
    `vol[i, :, :]`. The in-plane dimensions are physical axes 1 and 2, which are
    the spacings used for resampling.
    """
    valid: list[tuple[int, np.ndarray, np.ndarray]] = []
    n_slices = vol_img.shape[SAG_AXIS]

    for i in range(n_slices):
        img_slice = vol_img[i, :, :]
        seg_slice = vol_seg[i, :, :]

        mask_area = int(np.sum(seg_slice > 0))
        if mask_area >= min_mask_area:
            valid.append((i, img_slice, seg_slice))

    return valid


def get_inplane_spacings(
    header: nib.nifti1.Nifti1Header,
) -> tuple[float, float]:
    """
    Read sagittal in-plane spacings from the canonicalized NIfTI header.

    Returns:
        `(spacing_row, spacing_col)` in mm/px.
    """
    # Slices are vol[i, :, :], so in-plane dimensions are axes 1 and 2.
    pixdim = header.get_zooms()
    return float(pixdim[1]), float(pixdim[2])


def resample_slice(
    img_slice: np.ndarray,
    seg_slice: np.ndarray,
    spacing_row: float,
    spacing_col: float,
    target_spacing: float = TARGET_SPACING_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize an image/mask pair so one pixel represents `target_spacing` mm.

    New dimensions are computed from physical size:
        physical_size_mm = dim_px * spacing_mm_px
        new_dim_px = round(physical_size_mm / target_spacing)

    Interpolation choices:
      - Image: INTER_CUBIC. MRI intensity is continuous, and cubic interpolation
        preserves smooth gradients without obvious aliasing.
      - Mask: INTER_NEAREST. The mask is binary; intermediate values would
        corrupt labels after rounding or thresholding.
    """
    h_px, w_px = img_slice.shape

    h_mm = h_px * spacing_row
    w_mm = w_px * spacing_col

    new_h = max(1, round(h_mm / target_spacing))
    new_w = max(1, round(w_mm / target_spacing))

    img_res = cv2.resize(
        img_slice.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )
    seg_res = cv2.resize(
        seg_slice.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return img_res, seg_res


def pad_or_crop_center(
    arr: np.ndarray,
    target_h: int = IMAGE_SIZE,
    target_w: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Convert a 2-D array to the target shape without scaling or distortion.

    Smaller dimensions are centered with zero padding. Larger dimensions are
    center-cropped. Because this step happens after physical resampling, it does
    not change the mm/px resolution. The same crop/pad operation is applied to
    image and mask, preserving their alignment.
    """
    h, w = arr.shape
    out = np.zeros((target_h, target_w), dtype=arr.dtype)

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

    out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


def normalize_minmax(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image to [0, 1] with min-max scaling.

    If the intensity range is zero, return the image as float32 unchanged.
    """
    vmin, vmax = float(img.min()), float(img.max())
    diff = vmax - vmin
    if diff == 0:
        return img.astype(np.float32)
    return ((img - vmin) / diff).astype(np.float32)


class MRIPipelineProcessor:
    """
    Orchestrate the complete MRI NIfTI-to-`.npy` preprocessing pipeline.

    Execution order:
        1. Patient-level split without leakage.
        2. Output directory creation.
        3. Split-manifest export.
        4. Per-patient processing:
            a. Load and canonicalize to RAS+.
            b. Extract and filter 2-D slices.
            c. Apply min-max normalization.
            d. Resample to 0.8 mm/px.
            e. Center crop or pad to 256 x 256.
            f. Save individual `.npy` image/mask pairs.
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

    def _find_nifti(self, folder: str, suffix: str) -> Optional[str]:
        """Return the first `.nii` or `.nii.gz` file ending in `suffix`."""
        matches = glob(os.path.join(folder, f"*{suffix}.nii*"))
        return matches[0] if matches else None

    def _process_patient(
        self,
        folder: str,
        out_dirs: dict[str, str],
    ) -> int:
        """
        Process all valid slices from one patient folder.

        Returns:
            Number of saved slices, or zero if the patient is skipped.
        """
        patient_id = os.path.basename(folder)

        img_path = self._find_nifti(folder, self.img_suffix)
        seg_path = self._find_nifti(folder, self.mask_suffix)

        if img_path is None or seg_path is None:
            log.warning("Patient '%s': missing image or mask. Skipping.", patient_id)
            return 0

        result_img = load_canonical_nifti(img_path)
        result_seg = load_canonical_nifti(seg_path)

        if result_img is None or result_seg is None:
            return 0

        vol_img, header = result_img
        vol_seg, _ = result_seg

        if vol_img.shape != vol_seg.shape:
            log.warning(
                "Patient '%s': image %s and mask %s have different shapes. Skipping.",
                patient_id, vol_img.shape, vol_seg.shape,
            )
            return 0

        spacing_row, spacing_col = get_inplane_spacings(header)

        valid_slices = extract_valid_slices(vol_img, vol_seg, self.min_mask_area)

        saved_count = 0
        for slice_idx, img_slice, seg_slice in valid_slices:

            img_norm = normalize_minmax(img_slice)

            img_res, seg_res = resample_slice(
                img_norm, seg_slice,
                spacing_row, spacing_col,
                self.target_spacing,
            )

            img_out = pad_or_crop_center(img_res, self.image_size, self.image_size)
            seg_out = pad_or_crop_center(seg_res, self.image_size, self.image_size)

            # Restore binary labels after resize and crop/pad.
            seg_out = (seg_out > 0).astype(np.float32)

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

    def run(self) -> None:
        """Run the full pipeline and log saved-slice counts per split."""
        log.info("=" * 60)
        log.info("  MRI Pipeline - start")
        log.info("  Base path : %s", self.base_path)
        log.info("  Output    : %s", self.output_path)
        log.info("  Spacing   : %.1f mm/px  |  Tile: %d px", self.target_spacing, self.image_size)
        log.info("=" * 60)

        splits = split_patients(
            self.base_path,
            val_ratio=SPLITS["val"],
            test_ratio=SPLITS["test"],
        )

        out_dirs = build_output_dirs(self.output_path)
        save_split_manifest(splits, self.output_path)

        total_stats: dict[str, int] = {}

        for split_name, folders in splits.items():
            split_slices = 0
            desc = f"[{split_name.upper():5s}] Processing patients"

            for folder in tqdm(folders, desc=desc, unit="patient"):
                saved = self._process_patient(folder, out_dirs[split_name])
                split_slices += saved

            total_stats[split_name] = split_slices
            log.info("Split %-5s -> %d saved slices", split_name, split_slices)

        log.info("-" * 60)
        log.info("  FINAL SUMMARY")
        for split_name, count in total_stats.items():
            log.info("    %-6s : %d .npy slices", split_name, count)
        log.info("  Total   : %d .npy slices", sum(total_stats.values()))
        log.info("=" * 60)
        log.info("  Pipeline completed. Data ready in: %s", self.output_path)


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
