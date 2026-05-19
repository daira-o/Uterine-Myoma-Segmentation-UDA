"""
Build fresh train/val/test splits after visual curation.

Default workflow:
    python scripts/build_us_splits_from_clean.py

Example:
    python scripts/build_us_splits_from_clean.py --ratios 0.70 0.15 0.15 --random-state 123
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_ROOT = PROJECT_ROOT / "data_ready_US_clean"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data_ready_US_curated"
DEFAULT_REVIEW_JSON = PROJECT_ROOT / "data_ready_US_review_progress.json"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_ratios(ratios: list[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("--ratios debe recibir exactamente tres valores: train val test")
    if any(r < 0 for r in ratios):
        raise ValueError("Los ratios no pueden ser negativos.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("La suma de ratios debe ser mayor que cero.")
    return ratios[0] / total, ratios[1] / total, ratios[2] / total


def discover_clean_files(clean_root: Path) -> list[Path]:
    files = sorted(clean_root.rglob("*.npy"), key=lambda p: str(p).lower())
    if not files:
        raise FileNotFoundError(f"No se encontraron .npy limpios en {clean_root}")
    return files


def patient_id_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem
    stem = re.sub(r"^\d+(?:[.,]\d+)?cm[_\-\s]*", "", stem, flags=re.IGNORECASE)
    numbers = re.findall(r"\d+", stem)
    if len(numbers) >= 2:
        return ".".join(numbers[:-1])
    return None


def split_counts(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    train_ratio, val_ratio, _ = ratios
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    train_count = min(max(train_count, 0), total)
    val_count = min(max(val_count, 0), total - train_count)
    test_count = total - train_count - val_count
    return {"train": train_count, "val": val_count, "test": test_count}


def assign_units(
    units: list[tuple[str, list[Path]]],
    ratios: tuple[float, float, float],
    random_state: int,
) -> dict[str, list[Path]]:
    rng = Random(random_state)
    shuffled = list(units)
    rng.shuffle(shuffled)

    total_files = sum(len(paths) for _, paths in shuffled)
    targets = split_counts(total_files, ratios)
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for _, paths in shuffled:
        split_name = min(
            ("train", "val", "test"),
            key=lambda name: (
                len(splits[name]) / targets[name] if targets[name] else float("inf"),
                len(splits[name]),
            ),
        )
        splits[split_name].extend(paths)

    for paths in splits.values():
        paths.sort(key=lambda p: p.name.lower())
    return splits


def build_units(files: list[Path], split_by_patient: bool) -> tuple[list[tuple[str, list[Path]]], str]:
    if split_by_patient:
        groups: dict[str, list[Path]] = defaultdict(list)
        missing_patient = False
        for path in files:
            patient_id = patient_id_from_filename(path.name)
            if not patient_id:
                missing_patient = True
                break
            groups[patient_id].append(path)
        if not missing_patient and groups:
            return sorted(groups.items(), key=lambda item: item[0]), "patient"

    return [(path.name, [path]) for path in files], "file"


def unique_destination(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        candidate = dest_dir / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_split_files(
    splits: dict[str, list[Path]],
    clean_root: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, list[dict[str, str]]]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    elif output_root.exists() and any(output_root.rglob("*.npy")):
        raise FileExistsError(
            f"{output_root} ya contiene .npy. Usa --overwrite para recrearlo."
        )

    copied: dict[str, list[dict[str, str]]] = {}
    for split_name, files in splits.items():
        dest_dir = output_root / split_name / "images"
        copied[split_name] = []
        for src in files:
            dest = unique_destination(dest_dir, src.name)
            shutil.copy2(src, dest)
            copied[split_name].append(
                {
                    "filename": dest.name,
                    "source_path": str(src.resolve()).replace("\\", "/"),
                    "source_relative_path": str(src.relative_to(clean_root)).replace("\\", "/"),
                    "curated_path": str(dest.resolve()).replace("\\", "/"),
                    "patient_id": patient_id_from_filename(src.name) or "",
                }
            )
    return copied


def load_discarded_from_review(review_json: Path) -> list[dict[str, Any]]:
    if not review_json.exists():
        return []
    with review_json.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    decisions = payload.get("decisions", {}) if isinstance(payload, dict) else {}
    return [
        row
        for row in decisions.values()
        if row.get("decision") in {"DISCARD", "UNSURE"}
    ]


def save_manifest(
    output_root: Path,
    clean_root: Path,
    copied: dict[str, list[dict[str, str]]],
    discarded: list[dict[str, Any]],
    split_unit: str,
    ratios: tuple[float, float, float],
    random_state: int,
) -> None:
    manifest = {
        "created_at": now_iso(),
        "clean_root": str(clean_root.resolve()).replace("\\", "/"),
        "output_root": str(output_root.resolve()).replace("\\", "/"),
        "split_unit": split_unit,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "random_state": random_state,
        "total": sum(len(rows) for rows in copied.values()),
        "counts": {name: len(rows) for name, rows in copied.items()},
        "files_used": copied,
        "files_discarded": discarded,
    }
    manifest_path = output_root / "us_curated_splits_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera nuevos splits train/val/test desde data_ready_US_clean."
    )
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.80, 0.10, 0.10])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split-by",
        choices=("auto", "patient", "file"),
        default="auto",
        help="auto intenta paciente y cae a archivo si no puede inferirlo.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recrear data_ready_US_curated si ya existe.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_root = args.clean_root.resolve()
    output_root = args.output_root.resolve()
    ratios = normalize_ratios(args.ratios)
    files = discover_clean_files(clean_root)

    split_by_patient = args.split_by in {"auto", "patient"}
    units, split_unit = build_units(files, split_by_patient=split_by_patient)
    if args.split_by == "patient" and split_unit != "patient":
        raise ValueError("No se pudo inferir paciente para todos los archivos.")

    splits = assign_units(units, ratios, args.random_state)
    copied = copy_split_files(splits, clean_root, output_root, args.overwrite)
    discarded = load_discarded_from_review(args.review_json.resolve())
    save_manifest(
        output_root=output_root,
        clean_root=clean_root,
        copied=copied,
        discarded=discarded,
        split_unit=split_unit,
        ratios=ratios,
        random_state=args.random_state,
    )

    print("Splits curados generados.")
    print(f"Entrada limpia : {clean_root}")
    print(f"Salida         : {output_root}")
    print(f"Unidad split   : {split_unit}")
    print(f"Total          : {sum(len(v) for v in copied.values())}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name:5s}: {len(copied[split_name])}")
    print(f"Manifest       : {output_root / 'us_curated_splits_manifest.json'}")


if __name__ == "__main__":
    main()
