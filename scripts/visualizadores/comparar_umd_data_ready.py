"""
Compara un corte original de UMD contra su version procesada en data_ready_RM.

Uso rapido:
    python scripts/visualizadores/comparar_umd_data_ready.py

Ejemplos:
    python scripts/visualizadores/comparar_umd_data_ready.py --split train
    python scripts/visualizadores/comparar_umd_data_ready.py --patient UMD_221129_001 --slice 296
    python scripts/visualizadores/comparar_umd_data_ready.py --original-aspect native
    python scripts/visualizadores/comparar_umd_data_ready.py --save-preview outputs/preview.png --no-show
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


LOCAL_ENV = read_env_file(PROJECT_ROOT / ".env")

DEFAULT_UMD_ROOT = Path(
    os.getenv("NIFTI_ROOT", LOCAL_ENV.get("NIFTI_ROOT", str(PROJECT_ROOT / "data" / "UMD")))
)
DEFAULT_READY_ROOT = Path(
    os.getenv("MRI_OUTPUT_PATH", LOCAL_ENV.get("MRI_OUTPUT_PATH", str(PROJECT_ROOT / "data_ready_RM")))
)
DEFAULT_IMG_SUFFIX = os.getenv("NIFTI_IMG_SUFFIX", LOCAL_ENV.get("NIFTI_IMG_SUFFIX", "_t2"))

SAG_PATTERN = re.compile(r"^(.+)_sag_(\d+)(?:_.+)?$")


def normalize_display(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)

    lo, hi = np.percentile(img[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(img[finite])), float(np.max(img[finite]))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return out.astype(np.float32)


def parse_processed_name(path: Path) -> tuple[str, int]:
    stem = path.stem
    match = SAG_PATTERN.match(stem)
    if not match:
        raise ValueError(
            f"Nombre no compatible: {path.name}. "
            "Se esperaba algo como UMD_221129_001_sag_296.npy."
        )
    return match.group(1), int(match.group(2))


def find_processed_files(ready_root: Path, split: str | None) -> list[Path]:
    patterns: list[str] = []
    if split and split != "all":
        patterns.extend([
            str(ready_root / split / "images" / "*.npy"),
            str(ready_root / split / "images_npy" / "*.npy"),
        ])
    else:
        patterns.extend([
            str(ready_root / "*" / "images" / "*.npy"),
            str(ready_root / "*" / "images_npy" / "*.npy"),
            str(ready_root / "images" / "*.npy"),
            str(ready_root / "images_npy" / "*.npy"),
            str(ready_root / "images_npy" / "*" / "*.npy"),
        ])

    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob(pattern))
    return sorted(set(files))


def find_processed_file(
    ready_root: Path,
    patient_id: str,
    slice_idx: int,
    split: str | None,
) -> Path:
    target = f"{patient_id}_sag_{slice_idx}"
    candidates = [
        path for path in find_processed_files(ready_root, split)
        if path.stem == target or path.stem.startswith(f"{target}_")
    ]
    if not candidates:
        split_msg = f" en split '{split}'" if split and split != "all" else ""
        raise FileNotFoundError(
            f"No encontre un .npy procesado para {target}{split_msg}."
        )
    return candidates[0]


def choose_processed_file(args: argparse.Namespace) -> Path:
    if args.processed:
        return Path(args.processed)
    if args.patient and args.slice is not None:
        return find_processed_file(args.ready_root, args.patient, args.slice, args.split)

    candidates = find_processed_files(args.ready_root, args.split)
    if not candidates:
        raise FileNotFoundError(
            f"No encontre imagenes .npy en {args.ready_root}. "
            "Esperaba data_ready_RM/<split>/images/*.npy."
        )
    return random.choice(candidates)


def find_nifti(umd_root: Path, patient_id: str, suffix: str) -> Path:
    patient_dir = umd_root / patient_id
    matches = sorted(patient_dir.glob(f"*{suffix}.nii*"))
    if not matches:
        raise FileNotFoundError(
            f"No encontre NIfTI con sufijo '{suffix}' para {patient_id} en {patient_dir}."
        )
    return matches[0]


def load_original_slice(nifti_path: Path, slice_idx: int) -> tuple[np.ndarray, str, tuple[float, float]]:
    nii = nib.load(str(nifti_path))
    canonical = nib.as_closest_canonical(nii)
    vol = canonical.get_fdata(dtype=np.float32)

    if vol.ndim != 3:
        raise ValueError(f"El NIfTI no es 3D: {nifti_path} -> shape={vol.shape}")
    if slice_idx < 0 or slice_idx >= vol.shape[0]:
        raise IndexError(
            f"Slice sag_{slice_idx} fuera de rango para {nifti_path.name}; "
            f"el eje 0 tiene {vol.shape[0]} cortes."
        )

    slice_img = vol[slice_idx, :, :]
    zooms = canonical.header.get_zooms()[:3]
    spacing_row, spacing_col = float(zooms[1]), float(zooms[2])
    info = (
        f"shape original={vol.shape} | corte eje 0={slice_idx} | "
        f"spacing=({zooms[0]:.3f}, {zooms[1]:.3f}, {zooms[2]:.3f}) mm"
    )
    return slice_img, info, (spacing_row, spacing_col)


def image_stats(img: np.ndarray) -> str:
    img = np.asarray(img)
    return (
        f"shape={img.shape} | min={np.min(img):.3f} | max={np.max(img):.3f} | "
        f"mean={np.mean(img):.3f} | p99={np.percentile(img, 99):.3f}"
    )


def load_pair(args: argparse.Namespace, processed_path: Path | None = None) -> dict:
    processed_path = processed_path or choose_processed_file(args)
    patient_id, slice_idx = parse_processed_name(processed_path)
    nifti_path = find_nifti(args.umd_root, patient_id, args.img_suffix)

    original_raw, original_info, original_spacing = load_original_slice(nifti_path, slice_idx)
    processed = np.load(processed_path).astype(np.float32)

    return {
        "patient_id": patient_id,
        "slice_idx": slice_idx,
        "nifti_path": nifti_path,
        "processed_path": processed_path,
        "original_raw": original_raw,
        "original_display": normalize_display(original_raw),
        "processed": processed,
        "processed_display": normalize_display(processed),
        "original_info": original_info,
        "original_spacing": original_spacing,
    }


def draw_pair(fig, axes, sample: dict, original_aspect_mode: str = "physical") -> None:
    original_ax, processed_ax = axes
    for ax in axes:
        ax.clear()
        ax.axis("off")

    if original_aspect_mode == "physical":
        spacing_row, spacing_col = sample["original_spacing"]
        original_aspect = spacing_row / spacing_col if spacing_col > 0 else "equal"
        original_title = "UMD original NIfTI (proporcion anatomica)"
    else:
        original_aspect = "equal"
        original_title = "UMD original NIfTI (pixeles nativos)"

    original_ax.imshow(
        sample["original_display"],
        cmap="gray",
        vmin=0,
        vmax=1,
        aspect=original_aspect,
        interpolation="nearest",
    )
    original_ax.set_title(original_title, fontsize=11)
    original_ax.text(
        0.5,
        -0.08,
        f"{sample['nifti_path'].name}\n"
        f"{sample['original_info']}\n"
        f"{image_stats(sample['original_raw'])}",
        ha="center",
        va="top",
        transform=original_ax.transAxes,
        fontsize=8,
        wrap=True,
    )

    processed_ax.imshow(
        sample["processed_display"],
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    processed_ax.set_title("data_ready_RM procesada", fontsize=11)
    processed_ax.text(
        0.5,
        -0.08,
        f"{sample['processed_path'].name}\n"
        f"{image_stats(sample['processed'])}",
        ha="center",
        va="top",
        transform=processed_ax.transAxes,
        fontsize=8,
        wrap=True,
    )

    fig.suptitle(
        f"{sample['patient_id']} | sag_{sample['slice_idx']}",
        fontsize=13,
    )
    fig.canvas.draw_idle()


def print_sample(sample: dict) -> None:
    print(f"Paciente: {sample['patient_id']}")
    print(f"Slice: sag_{sample['slice_idx']}")
    print(f"Original: {sample['nifti_path']}")
    print(f"Procesada: {sample['processed_path']}")
    print(f"Original stats: {image_stats(sample['original_raw'])}")
    print(f"Procesada stats: {image_stats(sample['processed'])}")
    print("-" * 80)


def show_comparison(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    sample = load_pair(args)
    width_ratio = max(1.0, sample["original_raw"].shape[1] / sample["processed"].shape[1])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.8),
        gridspec_kw={"width_ratios": [width_ratio, 1]},
    )
    fig.subplots_adjust(bottom=0.24, top=0.86, wspace=0.08)
    draw_pair(fig, axes, sample, args.original_aspect)
    print_sample(sample)

    if args.save_preview:
        out_path = Path(args.save_preview)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Preview guardado en: {out_path}")

    if not args.no_show:
        button_ax = fig.add_axes([0.39, 0.04, 0.22, 0.07])
        next_button = Button(button_ax, "Nueva muestra")

        def refresh(_event=None):
            next_sample = load_pair(args)
            draw_pair(fig, axes, next_sample, args.original_aspect)
            print_sample(next_sample)

        next_button.on_clicked(refresh)
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara UMD original NIfTI contra data_ready_RM procesado."
    )
    parser.add_argument("--umd-root", type=Path, default=DEFAULT_UMD_ROOT)
    parser.add_argument("--ready-root", type=Path, default=DEFAULT_READY_ROOT)
    parser.add_argument("--img-suffix", default=DEFAULT_IMG_SUFFIX)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--processed", default=None, help="Ruta directa a un .npy procesado.")
    parser.add_argument("--patient", default=None, help="ID de paciente, ej: UMD_221129_001.")
    parser.add_argument("--slice", type=int, default=None, help="Indice sagital, ej: 296.")
    parser.add_argument(
        "--original-aspect",
        choices=["native", "physical"],
        default="physical",
        help="physical corrige por spacing para no aplastar la anatomia; native muestra pixeles crudos.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-preview", default=None, help="Guarda una imagen PNG del panel.")
    parser.add_argument("--no-show", action="store_true", help="No abre ventana interactiva.")
    return parser.parse_args()


if __name__ == "__main__":
    show_comparison(parse_args())
