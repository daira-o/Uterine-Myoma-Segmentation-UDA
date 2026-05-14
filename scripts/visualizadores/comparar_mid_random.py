import argparse
import os
import random
import sys
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from us_nosano_processor import _crop_fixed, _inpaint_annotations, _normalize


DEFAULT_MRI_READY = PROJECT_ROOT / "data_ready_RM"
DEFAULT_US_READY = PROJECT_ROOT / "data_ready_US"
DEFAULT_US_RAW = PROJECT_ROOT / "data" / "Ultrasound"
DEFAULT_MRI_ZOOM = 1.6


def normalize_display(img):
    img = np.asarray(img, dtype=np.float32)
    diff = float(np.max(img) - np.min(img))
    if diff == 0:
        return np.zeros_like(img, dtype=np.float32)
    return (img - np.min(img)) / diff


def load_npy_image(path):
    return normalize_display(np.load(path).astype(np.float32))


def center_zoom(img, zoom=DEFAULT_MRI_ZOOM):
    if zoom <= 1:
        return img

    h, w = img.shape[:2]
    crop_h = max(1, int(round(h / zoom)))
    crop_w = max(1, int(round(w / zoom)))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)


def load_us_processed_or_raw(path, source):
    if source == "npy":
        return load_npy_image(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen US: {path}")

    img = _crop_fixed(img)
    img = _inpaint_annotations(img)
    img = _normalize(img)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def choose_random_mri_mid(mri_ready):
    candidates = sorted(glob(os.path.join(mri_ready, "images_npy", "1_Anchor", "*_mid.npy")))
    if not candidates:
        candidates = sorted(glob(os.path.join(mri_ready, "images_npy", "*_mid.npy")))
    if not candidates:
        raise FileNotFoundError(
            "No encontre imagenes MRI mid. Ejecuta primero scripts/procesador_imagenes.py "
            "para generar data_ready_RM/images_npy/1_Anchor/*_mid.npy."
        )
    return random.choice(candidates)


def choose_random_us_f28(us_ready, us_raw):
    processed = sorted(set(glob(os.path.join(us_ready, "**", "*F28*.npy"), recursive=True)))
    processed += sorted(set(glob(os.path.join(us_ready, "**", "*f28*.npy"), recursive=True)))
    processed = sorted(set(processed))
    if processed:
        return random.choice(processed), "npy"

    raw_patterns = [
        os.path.join(us_raw, "**", "*F28*.jpg"),
        os.path.join(us_raw, "**", "*F28*.jpeg"),
        os.path.join(us_raw, "**", "*F28*.png"),
        os.path.join(us_raw, "**", "*f28*.jpg"),
        os.path.join(us_raw, "**", "*f28*.jpeg"),
        os.path.join(us_raw, "**", "*f28*.png"),
    ]
    raw = []
    for pattern in raw_patterns:
        raw.extend(glob(pattern, recursive=True))
    raw = sorted(set(raw))
    if raw:
        return random.choice(raw), "raw"

    raise FileNotFoundError("No encontre imagenes US F28 en data_ready_US ni en data/Ultrasound.")


def image_stats(img):
    return (
        f"mean={np.mean(img):.3f} | std={np.std(img):.3f} | "
        f"p05={np.percentile(img, 5):.3f} | p95={np.percentile(img, 95):.3f}"
    )


def sample_pair(args):
    mri_path = choose_random_mri_mid(args.mri_ready)
    us_path, us_source = choose_random_us_f28(args.us_ready, args.us_raw)
    return {
        "mri_path": mri_path,
        "us_path": us_path,
        "us_source": us_source,
        "mri_img": center_zoom(load_npy_image(mri_path), args.mri_zoom),
        "us_img": load_us_processed_or_raw(us_path, us_source),
    }


def clear_axis(ax):
    ax.clear()
    ax.axis("off")


def draw_pair(fig, axes, sample):
    mri_ax, us_ax = axes
    clear_axis(mri_ax)
    clear_axis(us_ax)

    mri_ax.imshow(sample["mri_img"], cmap="gray", vmin=0, vmax=1)
    mri_ax.set_title("MRI mid (_mid) zoom", fontsize=11)
    mri_ax.text(
        0.5,
        -0.08,
        f"{os.path.basename(sample['mri_path'])}\n{image_stats(sample['mri_img'])}",
        ha="center",
        va="top",
        transform=mri_ax.transAxes,
        fontsize=8,
        wrap=True,
    )

    us_ax.imshow(sample["us_img"], cmap="gray", vmin=0, vmax=1)
    us_ax.set_title("US F28", fontsize=11)
    us_ax.text(
        0.5,
        -0.08,
        f"{os.path.basename(sample['us_path'])} ({sample['us_source']})\n"
        f"{image_stats(sample['us_img'])}",
        ha="center",
        va="top",
        transform=us_ax.transAxes,
        fontsize=8,
        wrap=True,
    )

    fig.canvas.draw_idle()


def show_random_mid_images(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.4))
    fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.08)
    fig.suptitle("Random mid-plane images: MRI vs US F28", fontsize=13)

    button_ax = fig.add_axes([0.38, 0.04, 0.24, 0.07])
    next_button = Button(button_ax, "Nueva comparacion")

    def refresh(_event=None):
        sample = sample_pair(args)
        draw_pair(fig, axes, sample)
        print(f"MRI mid: {os.path.basename(sample['mri_path'])}")
        print(f"US F28 ({sample['us_source']}): {os.path.basename(sample['us_path'])}")
        print("-" * 80)

    next_button.on_clicked(refresh)
    refresh()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara imagenes random sin mascaras: MRI mid vs US F28."
    )
    parser.add_argument("--mri-ready", default=str(DEFAULT_MRI_READY))
    parser.add_argument("--us-ready", default=str(DEFAULT_US_READY))
    parser.add_argument("--us-raw", default=str(DEFAULT_US_RAW))
    parser.add_argument("--mri-zoom", type=float, default=DEFAULT_MRI_ZOOM)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    show_random_mid_images(parse_args())
