"""
config.py
─────────────────────────────────────────────────────────────────────────────
Configuración central del proyecto. Lee rutas desde .env y expone CONFIG.

Todos los scripts importan desde aquí:
    from config import CONFIG

Instalación: pip install python-dotenv
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Buscar el .env en la raíz del proyecto (un nivel arriba de este archivo
# si config.py está en la raíz, o donde corresponda)
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


def _require(key: str) -> str:
    """Lee una variable de entorno y lanza error claro si no existe."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"\n\n  Variable '{key}' no encontrada.\n"
            f"  Verificá que existe el archivo .env en:\n"
            f"  {_ENV_PATH}\n"
            f"  y que contiene:  {key}=<valor>\n"
        )
    return val


CONFIG = {
    # ── Rutas (vienen del .env) ───────────────────────────────────────────
    "base_path":         _require("DATA_PATH"),
    "nifti_root":        _require("NIFTI_ROOT"),
    "nifti_img_suffix":  os.getenv("NIFTI_IMG_SUFFIX", "_t2"),
    "model_path":        os.getenv("MODEL_PATH", "best_model_sagital.pth"),
    "logs_path":         os.getenv("LOGS_PATH", "logs"),

    # ── Arquitectura ──────────────────────────────────────────────────────
    "in_channels":  1,
    "num_classes":  1,

    # ── Entrenamiento ─────────────────────────────────────────────────────
    "batch_size":   8,
    "epochs":       30,
    "lr":           1e-4,

    # ── Dataset ───────────────────────────────────────────────────────────
    "val_size":     0.2,      # fracción de PACIENTES para validación
    "random_state": 42,
    # Limitar pacientes para PCs con poca RAM. None = todos.
    "max_patients": None,
    # Máximo de slices por paciente.
    # None = sin límite. Usar 20 si los .npy son del procesador corregido.
    "max_slices_per_patient": None,

    # ── Métricas / inferencia ─────────────────────────────────────────────
    "threshold":     0.5,
    "iou_threshold": 0.1,
}
