"""
Launcher para abrir el visualizador US con checkpoints weak supervision.

Uso:
    python scripts/visualization/run_visualizar_us_modelo_weak.py
"""

from __future__ import annotations

import site
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENV_SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"
TARGET_APP = ROOT / "scripts" / "visualization" / "visualizar_us_modelo.py"


def main() -> None:
    import pathlib  # Keep stdlib pathlib loaded before site-packages are added.

    _ = pathlib
    if VENV_SITE_PACKAGES.exists():
        site.addsitedir(str(VENV_SITE_PACKAGES))

    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(TARGET_APP),
        "--server.port",
        "8502",
        "--server.address",
        "localhost",
    ]
    stcli.main()


if __name__ == "__main__":
    main()
