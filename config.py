"""Central project configuration and filesystem paths.

This file is intentionally kept at the repository root because the training,
preprocessing and visualization entry points live under ``scripts/``.
All default paths are derived from ``Path(__file__)`` so the project works from
any current working directory on Windows and Linux.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

DATA_DIR = PROJECT_ROOT / "data"
MRI_READY_DIR = PROJECT_ROOT / "data_ready_RM"
US_READY_DIR = PROJECT_ROOT / "data_ready_US"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _path_env(key: str, default: Path) -> str:
    """Return env paths resolved from PROJECT_ROOT when they are relative."""
    value = os.getenv(key)
    if not value:
        return str(default)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _float_list_env(key: str, default: str) -> list[float]:
    """Parse a comma-separated float list from env."""
    raw = os.getenv(key, default)
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


CONFIG = {
    # Paths
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(DATA_DIR),
    "outputs_path": _path_env("OUTPUTS_PATH", OUTPUTS_DIR),
    "base_path": _path_env("DATA_PATH", MRI_READY_DIR),
    "nifti_root": _path_env("NIFTI_ROOT", DATA_DIR / "UMD"),
    "mri_data_path": _path_env("MRI_DATA_PATH", DATA_DIR / "UMD"),
    "mri_output_path": _path_env("MRI_OUTPUT_PATH", MRI_READY_DIR),
    "us_data_path": _path_env("US_DATA_PATH", DATA_DIR / "Ultrasound"),
    "us_output_path": _path_env("US_OUTPUT_PATH", US_READY_DIR),
    "us_ready_path": _path_env("US_READY_PATH", US_READY_DIR),
    "model_path": _path_env("MODEL_PATH", PROJECT_ROOT / "best_model_sagital.pth"),
    "logs_path": _path_env("LOGS_PATH", LOGS_DIR),
    # Preprocessing
    "nifti_img_suffix": os.getenv("NIFTI_IMG_SUFFIX", "_t2"),
    "nifti_mask_suffix": os.getenv("NIFTI_MASK_SUFFIX", "_seg"),
    "processor_image_size": int(os.getenv("PROCESSOR_IMAGE_SIZE", "256")),
    # Architecture
    "in_channels": 1,
    "num_classes": 1,
    "base_filters": int(os.getenv("BASE_FILTERS", "64")),
    # Training
    "batch_size": int(os.getenv("BATCH_SIZE", "8")),
    "epochs": int(os.getenv("EPOCHS", "30")),
    "lr": float(os.getenv("LR", "1e-4")),
    "use_dann": os.getenv("USE_DANN", "0").lower() in {"1", "true", "yes", "on"},
    "lambda_domain": float(os.getenv("LAMBDA_DOMAIN", "0.0")),
    "target_steps_per_epoch": int(os.getenv("TARGET_STEPS_PER_EPOCH", "0")),
    "target_save_epoch_checkpoints": int(os.getenv("TARGET_SAVE_EPOCH_CHECKPOINTS", "1")),
    "val_size": float(os.getenv("VAL_SIZE", "0.2")),
    "random_state": int(os.getenv("RANDOM_STATE", "42")),
    "max_patients": None,
    "max_slices_per_patient": None,
    # Weak US bbox supervision
    "weak_loss_type": os.getenv("WEAK_LOSS_TYPE", "soft_bbox"),
    "weak_bbox_margin_px": int(os.getenv("WEAK_BBOX_MARGIN_PX", "4")),
    "weak_outside_weight": float(os.getenv("WEAK_OUTSIDE_WEIGHT", "1.0")),
    "weak_inside_weight": float(os.getenv("WEAK_INSIDE_WEIGHT", "2.0")),
    "weak_area_weight": float(os.getenv("WEAK_AREA_WEIGHT", "0.10")),
    "weak_min_inside_activation": float(os.getenv("WEAK_MIN_INSIDE_ACTIVATION", "0.35")),
    "weak_min_inside_mean": float(os.getenv("WEAK_MIN_INSIDE_MEAN", "0.08")),
    "weak_inside_mean_weight": float(os.getenv("WEAK_INSIDE_MEAN_WEIGHT", "0.5")),
    "weak_min_area_ratio": float(os.getenv("WEAK_MIN_AREA_RATIO", "0.25")),
    "weak_max_area_ratio": float(os.getenv("WEAK_MAX_AREA_RATIO", "1.0")),
    "weak_under_area_weight": float(os.getenv("WEAK_UNDER_AREA_WEIGHT", "4.0")),
    "weak_over_area_weight": float(os.getenv("WEAK_OVER_AREA_WEIGHT", "1.0")),
    "weak_debug_dir": _path_env("WEAK_DEBUG_DIR", LOGS_DIR / "checkpoints"),
    "weak_debug_every": int(os.getenv("WEAK_DEBUG_EVERY", "1")),
    "weak_debug_max_batches": int(os.getenv("WEAK_DEBUG_MAX_BATCHES", "2")),
    "weak_debug_max_samples": int(os.getenv("WEAK_DEBUG_MAX_SAMPLES", "2")),
    "us_metric_thresholds": _float_list_env("US_METRIC_THRESHOLDS", "0.30,0.35,0.40,0.50"),
    # Metrics / inference
    "threshold": float(os.getenv("THRESHOLD", "0.5")),
    "iou_threshold": float(os.getenv("IOU_THRESHOLD", "0.1")),
}
