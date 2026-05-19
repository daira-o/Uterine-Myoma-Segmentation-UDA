"""
scripts/infer_us_production.py
Fase 3: inferencia de produccion sobre ultrasonido estandarizado.

Este script descarta GRL y DomainDiscriminator. Carga solamente el segmentador
adaptado desde un checkpoint DANN y predice mascaras sobre imagenes US 256x256
normalizadas, idealmente a 0.8 mm/px como las generadas por us_pipeline.py.

Uso:
    python scripts/infer_us_production.py --input data_ready_US/test/images
    python scripts/infer_us_production.py --input path/a/imagen.npy --threshold 0.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.attention_unet import AttentionUNet

try:
    from config import CONFIG
except Exception as exc:
    CONFIG = {
        "logs_path": "logs",
        "threshold": 0.5,
        "in_channels": 1,
        "num_classes": 1,
        "base_filters": 64,
    }
    logging.getLogger(__name__).warning(
        "No se pudo cargar config.py (%s). Se usan defaults de inferencia.",
        exc,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


DEFAULT_CHECKPOINT = os.path.join(
    CONFIG["logs_path"],
    "checkpoints_dann",
    "best_model_dann.pth",
)
DEFAULT_OUTPUT_DIR = os.path.join("outputs", "phase3_inference")
SUPPORTED_IMAGE_EXTS = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia Fase 3: segmentacion US con segmentador DANN adaptado."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Archivo .npy/.png/.jpg o carpeta con imagenes US estandarizadas 256x256.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        type=str,
        help="Checkpoint DANN. Default: logs/checkpoints_dann/best_model_dann.pth",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=str,
        help="Carpeta donde guardar probabilidades, mascaras y PNGs.",
    )
    parser.add_argument(
        "--threshold",
        default=CONFIG.get("threshold", 0.5),
        type=float,
        help="Umbral para binarizar la probabilidad.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Dispositivo de inferencia.",
    )
    parser.add_argument(
        "--save-prob",
        action="store_true",
        help="Guardar tambien el mapa de probabilidades .npy.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="No guardar imagenes PNG de mascara/overlay.",
    )
    return parser.parse_args()


def iter_input_paths(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            raise ValueError(f"Extension no soportada: {path.suffix}")
        return [path]

    if path.is_dir():
        paths = [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
        if not paths:
            raise FileNotFoundError(f"No se encontraron imagenes soportadas en {path}")
        return paths

    raise FileNotFoundError(f"No existe input: {input_path}")


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)

    if arr.ndim != 2:
        raise ValueError(f"Se esperaba imagen 2D o canal unico, shape recibido: {arr.shape}")

    if arr.shape != (256, 256):
        raise ValueError(
            f"La Fase 3 espera US estandarizado 256x256 a 0.8 mm/px; "
            f"shape recibido: {arr.shape}. Ejecuta primero scripts/us_pipeline.py."
        )

    if arr.max() > 1.0 or arr.min() < 0.0:
        vmin = float(arr.min())
        vmax = float(arr.max())
        arr = (arr - vmin) / (vmax - vmin + 1e-8)

    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return normalize_image(np.load(path))

    if not PIL_OK:
        raise ImportError("Pillow no esta instalado; usa .npy o instala pillow.")
    image = Image.open(path).convert("L")
    return normalize_image(np.asarray(image, dtype=np.float32) / 255.0)


def extract_segmenter_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """
    Extrae pesos de segmentacion desde checkpoints DANN o state_dicts puros.

    Se ignoran explicitamente GRL y DomainDiscriminator.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint

    if "segmenter_state_dict" in checkpoint:
        state_dict = checkpoint["segmenter_state_dict"]
    elif "model_state_dict" in checkpoint:
        full_state = checkpoint["model_state_dict"]
        state_dict = {
            key[len("segmenter.") :]: value
            for key, value in full_state.items()
            if key.startswith("segmenter.")
        }
        if not state_dict:
            raise ValueError("El checkpoint no contiene pesos con prefijo 'segmenter.'.")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        if clean_key.startswith("segmenter."):
            clean_key = clean_key[len("segmenter.") :]
        if clean_key.startswith(("domain_discriminator.", "grl.")):
            continue
        cleaned[clean_key] = value
    return cleaned


def load_segmenter(checkpoint_path: str, device: torch.device) -> AttentionUNet:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_segmenter_state_dict(checkpoint)

    model = AttentionUNet(
        in_channels=CONFIG["in_channels"],
        num_classes=CONFIG["num_classes"],
        base_filters=CONFIG.get("base_filters", 64),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Pesos faltantes al cargar segmenter: %s", list(missing))
    if unexpected:
        log.warning("Pesos inesperados ignorados: %s", list(unexpected))

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(
    model: AttentionUNet,
    image_np: np.ndarray,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(tensor)
    prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    mask = (prob >= threshold).astype(np.uint8)
    return prob, mask


def save_png_outputs(
    image_np: np.ndarray,
    prob: np.ndarray,
    mask: np.ndarray,
    stem: str,
    output_dir: Path,
) -> None:
    if not PIL_OK:
        log.warning("Pillow no esta instalado; se omiten PNGs.")
        return

    image_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    prob_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    mask_u8 = (mask * 255).astype(np.uint8)

    Image.fromarray(prob_u8).save(output_dir / f"{stem}_prob.png")
    Image.fromarray(mask_u8).save(output_dir / f"{stem}_mask.png")

    rgb = np.stack([image_u8, image_u8, image_u8], axis=-1)
    overlay = rgb.copy()
    overlay[mask.astype(bool), 0] = 255
    overlay[mask.astype(bool), 1] = (0.35 * overlay[mask.astype(bool), 1]).astype(np.uint8)
    overlay[mask.astype(bool), 2] = (0.35 * overlay[mask.astype(bool), 2]).astype(np.uint8)
    Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")


def run_inference(
    model: AttentionUNet,
    input_paths: Iterable[Path],
    output_dir: Path,
    device: torch.device,
    threshold: float,
    save_prob: bool,
    save_png: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    prob_dir = output_dir / "probabilities"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if save_prob:
        prob_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(input_paths, start=1):
        image_np = load_image(path)
        prob, mask = predict_mask(model, image_np, device, threshold)
        stem = path.stem

        np.save(mask_dir / f"{stem}_mask.npy", mask.astype(np.uint8))
        if save_prob:
            np.save(prob_dir / f"{stem}_prob.npy", prob.astype(np.float32))
        if save_png:
            save_png_outputs(image_np, prob, mask, stem, output_dir)

        log.info(
            "[%04d] %s -> area=%d px | prob_max=%.4f | threshold=%.2f",
            idx,
            path.name,
            int(mask.sum()),
            float(prob.max()),
            threshold,
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    input_paths = iter_input_paths(args.input)

    log.info("=" * 60)
    log.info("Fase 3 - Inferencia US de produccion")
    log.info("Checkpoint: %s", args.checkpoint)
    log.info("Input:      %s (%d archivo/s)", args.input, len(input_paths))
    log.info("Output:     %s", args.output_dir)
    log.info("Device:     %s", device)
    log.info("Threshold:  %.2f", args.threshold)
    log.info("GRL/DD:     descartados; se carga solo AttentionUNet segmenter")
    log.info("=" * 60)

    model = load_segmenter(args.checkpoint, device)
    run_inference(
        model=model,
        input_paths=input_paths,
        output_dir=Path(args.output_dir),
        device=device,
        threshold=args.threshold,
        save_prob=args.save_prob,
        save_png=not args.no_png,
    )
    log.info("Inferencia finalizada.")


if __name__ == "__main__":
    main()
