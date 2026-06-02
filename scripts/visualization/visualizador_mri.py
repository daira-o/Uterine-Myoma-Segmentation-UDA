"""Compara cortes MRI originales contra los tiles generados por el pipeline."""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

try:
    import nibabel as nib

    NIBABEL_OK = True
except ImportError:
    NIBABEL_OK = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import CONFIG
except ModuleNotFoundError:
    CONFIG = {
        "nifti_root": str(PROJECT_ROOT / "data" / "UMD"),
        "nifti_img_suffix": os.getenv("NIFTI_IMG_SUFFIX", "_t2"),
        "nifti_mask_suffix": os.getenv("NIFTI_MASK_SUFFIX", "_seg"),
    }

DEFAULT_BASE = PROJECT_ROOT / "data_ready_RM"
DEFAULT_PREVIEW_PATH = PROJECT_ROOT / "outputs" / "mri_preview.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualizador rapido de cortes MRI originales y procesados.",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=str(DEFAULT_BASE),
        help="Carpeta base procesada, ej: data_ready_RM.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Split a visualizar.",
    )
    parser.add_argument(
        "--original-root",
        type=str,
        default=CONFIG["nifti_root"],
        help="Carpeta raiz con subcarpetas de pacientes NIfTI.",
    )
    parser.add_argument(
        "--img-suffix",
        type=str,
        default=CONFIG["nifti_img_suffix"],
        help="Sufijo del NIfTI de imagen, ej: _t2.",
    )
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default=CONFIG["nifti_mask_suffix"],
        help="Sufijo del NIfTI de mascara, ej: _seg.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Ruta directa al split. Mantiene compatibilidad con el visualizador anterior.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        help="Cantidad de muestras a guardar con --no-show si no se usa --index.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Indice fijo dentro de la lista ordenada. Si no se indica, usa aleatorio.",
    )
    parser.add_argument(
        "--save-preview",
        type=str,
        default=None,
        help="Guarda una previsualizacion PNG.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abre ventana interactiva. Util con --save-preview.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def split_dir_from_args(args: argparse.Namespace) -> Path | None:
    if args.path:
        return resolve_path(args.path)
    if args.split == "all":
        return None
    return resolve_path(Path(args.base) / args.split)


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)

    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.clip(arr, 0.0, 1.0)


def can_show_interactively() -> bool:
    backend = plt.get_backend().lower()
    return "agg" not in backend


def list_pairs_from_split(split_dir: Path) -> list[tuple[Path, Path]]:
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"No se encontraron las carpetas 'images' y 'masks' en: {split_dir.resolve()}"
        )

    return pair_images_with_masks(sorted(img_dir.glob("*.npy")))


def pair_images_with_masks(img_paths: list[Path]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img_path in img_paths:
        if img_path.parent.name == "images":
            mask_path = img_path.parent.parent / "masks" / img_path.name
        else:
            mask_path = img_path.parent / "masks" / img_path.name

        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"[WARN] Falta la mascara para la imagen: {img_path.name}. Saltando...")
    return pairs


def list_pairs(base: Path, split: str, direct_split_dir: Path | None) -> list[tuple[Path, Path]]:
    if direct_split_dir is not None:
        return list_pairs_from_split(direct_split_dir)

    if split == "all":
        patterns = [
            str(base / "*" / "images" / "*.npy"),
            str(base / "images" / "*.npy"),
        ]
    else:
        patterns = [str(base / split / "images" / "*.npy")]

    img_paths = sorted({Path(p) for pattern in patterns for p in glob(pattern)})
    return pair_images_with_masks(img_paths)


def parse_mri_name(npy_path: Path) -> tuple[str | None, int | None]:
    match = re.match(r"^(.+)_sag_(\d+)$", npy_path.stem)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def find_nifti(original_root: Path, patient_id: str | None, suffix: str) -> Path | None:
    if patient_id is None:
        return None

    patient_dir = original_root / patient_id
    patterns = [
        str(patient_dir / f"*{suffix}.nii"),
        str(patient_dir / f"*{suffix}.nii.gz"),
        str(original_root / "**" / patient_id / f"*{suffix}.nii"),
        str(original_root / "**" / patient_id / f"*{suffix}.nii.gz"),
    ]
    hits = sorted({Path(p) for pattern in patterns for p in glob(pattern, recursive=True)})
    return hits[0] if hits else None


def load_nifti_slice(path: Path | None, slice_idx: int | None) -> np.ndarray | None:
    if path is None or slice_idx is None or not NIBABEL_OK:
        return None

    try:
        nii = nib.as_closest_canonical(nib.load(str(path)))
        vol = nii.get_fdata(dtype=np.float32)
    except Exception as exc:
        print(f"[WARN] No se pudo cargar NIfTI {path}: {exc}")
        return None

    if slice_idx < 0 or slice_idx >= vol.shape[0]:
        print(f"[WARN] Corte sag_{slice_idx} fuera de rango para {path.name} (n={vol.shape[0]})")
        return None

    # Mantiene el mismo eje que mri_pipeline.py para que el antes/despues sea comparable.
    slc = vol[slice_idx, :, :].astype(np.float32)
    if slc.max() > slc.min():
        slc = (slc - slc.min()) / (slc.max() - slc.min())
    return np.clip(slc, 0.0, 1.0)


def describe_image(arr: np.ndarray) -> str:
    return f"shape={arr.shape} | rango=[{arr.min():.3f}, {arr.max():.3f}]"


def draw_overlay(ax, img: np.ndarray, mask: np.ndarray, title: str) -> None:
    mask_binary = mask > 0.5
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    masked_overlay = np.ma.masked_where(~mask_binary, mask_binary)
    ax.imshow(masked_overlay, cmap="autumn", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)


def draw_sample(
    axes: np.ndarray,
    img_path: Path,
    mask_path: Path,
    original_root: Path,
    img_suffix: str,
    mask_suffix: str,
) -> None:
    processed = load_npy(img_path)
    processed_mask = load_npy(mask_path)
    processed_mask_binary = processed_mask > 0.5

    patient_id, slice_idx = parse_mri_name(img_path)
    original_path = find_nifti(original_root, patient_id, img_suffix)
    original_mask_path = find_nifti(original_root, patient_id, mask_suffix)
    original = load_nifti_slice(original_path, slice_idx)
    original_mask = load_nifti_slice(original_mask_path, slice_idx)

    for ax in axes:
        ax.clear()
        ax.axis("off")

    if original is not None:
        if original_mask is not None:
            draw_overlay(
                axes[0],
                original,
                original_mask,
                f"Original NIfTI + mascara\n{patient_id} sag_{slice_idx} | {describe_image(original)}",
            )
        else:
            axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title(
                f"Original NIfTI\n{patient_id} sag_{slice_idx} | {describe_image(original)}",
                fontsize=9,
            )
    else:
        axes[0].text(
            0.5,
            0.5,
            "Original NIfTI no encontrado",
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[0].set_title("Original NIfTI", fontsize=9)

    axes[1].imshow(processed_mask_binary, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title(
        f"Mascara procesada\narea: {int(processed_mask_binary.sum())} px",
        fontsize=9,
    )

    draw_overlay(
        axes[2],
        processed,
        processed_mask,
        f"Procesada 256x256 + mascara\n{img_path.name} | {describe_image(processed)}",
    )


def save_preview(fig, save_preview: str, idx: int, total: int) -> None:
    out = Path(save_preview)
    if total > 1:
        out = out.with_name(f"{out.stem}_{idx:02d}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Preview guardada en: {out.resolve()}")


def main() -> None:
    args = parse_args()
    base = resolve_path(args.base)
    split_dir = split_dir_from_args(args)
    original_root = resolve_path(args.original_root)

    try:
        pairs = list_pairs(base, args.split, split_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    source_label = split_dir.resolve() if split_dir else base.resolve()
    if not pairs:
        print(f"[ERROR] No hay pares imagen/mascara .npy en {source_label}")
        return

    if not NIBABEL_OK:
        print("[WARN] nibabel no esta instalado. Se mostrara solo el .npy procesado.")

    if args.index is not None:
        selected_indices = [max(0, min(args.index, len(pairs) - 1))]
    elif args.no_show:
        num_samples = max(1, min(args.num, len(pairs)))
        selected_indices = random.sample(range(len(pairs)), num_samples)
    else:
        selected_indices = [random.randrange(len(pairs))]

    print(f"Mostrando {len(selected_indices)} muestra(s) de: {source_label}")

    interactive = can_show_interactively()
    save_path = args.save_preview
    if not args.no_show and not interactive and save_path is None:
        save_path = str(DEFAULT_PREVIEW_PATH)
        print(
            "[WARN] Matplotlib esta usando un backend no interactivo; "
            f"se guardara una preview en: {DEFAULT_PREVIEW_PATH.resolve()}"
        )

    if args.no_show or not interactive:
        for out_idx, pair_idx in enumerate(selected_indices, start=1):
            img_path, mask_path = pairs[pair_idx]
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            draw_sample(axes, img_path, mask_path, original_root, args.img_suffix, args.mask_suffix)
            fig.suptitle(
                f"MRI antes/despues | split={args.split} | {pair_idx + 1}/{len(pairs)}",
                fontsize=12,
                fontweight="bold",
            )
            fig.tight_layout()

            if save_path:
                save_preview(fig, save_path, out_idx, len(selected_indices))
            plt.close(fig)
        return

    current_idx = selected_indices[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.18, wspace=0.08)

    def show_index(idx: int) -> None:
        nonlocal current_idx
        current_idx = idx % len(pairs)
        img_path, mask_path = pairs[current_idx]
        draw_sample(axes, img_path, mask_path, original_root, args.img_suffix, args.mask_suffix)
        fig.suptitle(
            f"MRI antes/despues | split={args.split} | {current_idx + 1}/{len(pairs)}",
            fontsize=12,
            fontweight="bold",
        )
        fig.canvas.draw_idle()

    show_index(current_idx)

    if save_path:
        save_preview(fig, save_path, 1, 1)

    ax_prev = plt.axes([0.28, 0.04, 0.16, 0.07])
    ax_next = plt.axes([0.56, 0.04, 0.16, 0.07])
    btn_prev = Button(ax_prev, "Anterior")
    btn_next = Button(ax_next, "Siguiente")
    btn_prev.on_clicked(lambda _event: show_index(current_idx - 1))
    btn_next.on_clicked(lambda _event: show_index(current_idx + 1))

    plt.show()


if __name__ == "__main__":
    main()
