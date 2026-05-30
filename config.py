"""Central project configuration and filesystem paths.

This file is intentionally kept at the repository root because the training,
preprocessing and visualization entry points live under ``scripts/``.
All default paths are derived from ``Path(__file__)`` so the project works from
any current working directory on Windows and Linux.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

DATA_DIR = PROJECT_ROOT / "data"
MRI_READY_DIR = PROJECT_ROOT / "data_ready_RM"
US_READY_DIR = PROJECT_ROOT / "data_ready_US"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _path_env(key: str, default: Path) -> str:
    """Return an env path or a project-relative default as a string.

    Relative paths from .env, such as LOGS_PATH=logs, are resolved against the
    repository root instead of the current shell folder. This keeps training,
    visualization and checkpoint paths consistent even when scripts are launched
    from scripts/training or scripts/visualization.
    """
    value = os.getenv(key)
    if not value:
        return str(default)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


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
    "val_size": float(os.getenv("VAL_SIZE", "0.2")),
    "random_state": int(os.getenv("RANDOM_STATE", "42")),
    "max_patients": None,
    "max_slices_per_patient": None,
    # Target adaptation stability
    # 0 means one target epoch == one pass over the US dataloader.
    "target_steps_per_epoch": int(os.getenv("TARGET_STEPS_PER_EPOCH", "0")),
    "dann_warmup_epochs": int(os.getenv("DANN_WARMUP_EPOCHS", "10")),
    # DANN remains available but is disabled for this experimental stage. The
    # visual/debug logs showed adversarial alignment quickly suppresses the US
    # anatomical activation once alpha becomes nonzero; weak bbox supervision is
    # currently the more stable signal, so localization is consolidated first.
    "lambda_domain_base": float(os.getenv("LAMBDA_DOMAIN_BASE", "0.0")),
    "target_early_stopping_patience": int(os.getenv("TARGET_EARLY_STOPPING_PATIENCE", "10")),
    "target_early_stopping_min_delta": float(os.getenv("TARGET_EARLY_STOPPING_MIN_DELTA", "1e-4")),
    "target_mri_guard_weight": float(os.getenv("TARGET_MRI_GUARD_WEIGHT", "0.05")),
    "target_min_val_dice": float(os.getenv("TARGET_MIN_VAL_DICE", "0.0")),
    # Weak US bbox supervision
    "weak_loss_type": os.getenv("WEAK_LOSS_TYPE", "soft_bbox"),
    "weak_bbox_margin_px": int(os.getenv("WEAK_BBOX_MARGIN_PX", "8")),
    "weak_bbox_expand_ratio": float(os.getenv("WEAK_BBOX_EXPAND_RATIO", "0.10")),
    "weak_outside_weight": float(os.getenv("WEAK_OUTSIDE_WEIGHT", "1.0")),
    "weak_inside_weight": float(os.getenv("WEAK_INSIDE_WEIGHT", "2.0")),
    "weak_area_weight": float(os.getenv("WEAK_AREA_WEIGHT", "1.0")),
    "weak_min_inside_activation": float(os.getenv("WEAK_MIN_INSIDE_ACTIVATION", "0.35")),
    "weak_min_area_ratio": float(os.getenv("WEAK_MIN_AREA_RATIO", "0.20")),
    "weak_max_area_ratio": float(os.getenv("WEAK_MAX_AREA_RATIO", "0.6")),
    "weak_under_area_weight": float(os.getenv("WEAK_UNDER_AREA_WEIGHT", "2.0")),
    "weak_over_area_weight": float(os.getenv("WEAK_OVER_AREA_WEIGHT", "1.0")),
    "weak_inside_fraction": float(os.getenv("WEAK_INSIDE_FRACTION", "0.25")),
    "us_metric_thresholds": os.getenv("US_METRIC_THRESHOLDS", "0.30,0.35,0.40,0.50"),
    "weak_debug_dir": os.getenv("WEAK_DEBUG_DIR", str(OUTPUTS_DIR / "debug_weak_supervision")),
    "weak_debug_every": int(os.getenv("WEAK_DEBUG_EVERY", "1")),
    "weak_debug_max_batches": int(os.getenv("WEAK_DEBUG_MAX_BATCHES", "2")),
    "weak_debug_max_samples": int(os.getenv("WEAK_DEBUG_MAX_SAMPLES", "4")),
    # Metrics / inference
    "threshold": float(os.getenv("THRESHOLD", "0.5")),
    "iou_threshold": float(os.getenv("IOU_THRESHOLD", "0.1")),
}
