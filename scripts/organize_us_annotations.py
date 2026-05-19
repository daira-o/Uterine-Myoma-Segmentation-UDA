"""
Organize ultrasound XML annotations next to their matching images.

Matches are made by exact base filename, ignoring only the file extension.
The existing image folder structure is preserved.

Default:
    python scripts/organize_us_annotations.py

Dry run:
    python scripts/organize_us_annotations.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "Ultrasound"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_inside_root(path: Path, root: Path) -> None:
    path.resolve().relative_to(root.resolve())


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_files(dataset_root: Path, unused_dir: Path) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    xmls: list[Path] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or is_inside(path, unused_dir):
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            images.append(path)
        elif suffix == ".xml":
            xmls.append(path)
    return sorted(images), sorted(xmls)


def index_by_stem(paths: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[path.stem].append(path)
    return dict(index)


def relative_folder(path: Path, dataset_root: Path) -> str:
    rel_parent = path.parent.relative_to(dataset_root)
    return "." if str(rel_parent) == "." else str(rel_parent).replace("\\", "/")


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    counter = 2
    while True:
        candidate = dest.with_name(f"{stem}__{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def transfer_file(src: Path, dest: Path, mode: str, dry_run: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    final_dest = dest if src.resolve() == dest.resolve() else unique_dest(dest)
    if dry_run or src.resolve() == final_dest.resolve():
        return final_dest
    if mode == "copy":
        shutil.copy2(src, final_dest)
    else:
        shutil.move(str(src), str(final_dest))
    return final_dest


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def organize_annotations(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    unused_dir = args.unused_dir.resolve()
    ensure_inside_root(dataset_root, PROJECT_ROOT)
    ensure_inside_root(unused_dir, PROJECT_ROOT)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"No existe dataset_root: {dataset_root}")

    images, xmls = discover_files(dataset_root, unused_dir)
    xmls_by_stem = index_by_stem(xmls)

    valid_xml_rows: list[dict[str, Any]] = []
    unused_xml_rows: list[dict[str, Any]] = []
    missing_xml_rows: list[dict[str, Any]] = []
    folder_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "images": 0,
            "valid_xml": 0,
            "images_without_xml": 0,
            "unused_xml": 0,
        }
    )

    valid_xml_paths: set[Path] = set()
    used_source_xml_paths: set[Path] = set()

    for image in images:
        folder = relative_folder(image, dataset_root)
        folder_stats[folder]["images"] += 1
        matching_xmls = xmls_by_stem.get(image.stem, [])
        if not matching_xmls:
            folder_stats[folder]["images_without_xml"] += 1
            missing_xml_rows.append(
                {
                    "image_filename": image.name,
                    "image_path": str(image.resolve()).replace("\\", "/"),
                    "folder": folder,
                    "expected_xml": f"{image.stem}.xml",
                }
            )
            continue

        target = image.with_suffix(".xml")
        same_folder_xmls = [
            xml for xml in matching_xmls
            if xml.parent.resolve() == image.parent.resolve()
        ]
        source_xml = target if target in same_folder_xmls else same_folder_xmls[0] if same_folder_xmls else matching_xmls[0]
        dest = transfer_file(source_xml, target, args.mode, args.dry_run)
        valid_xml_paths.add(dest.resolve())
        used_source_xml_paths.add(source_xml.resolve())
        folder_stats[folder]["valid_xml"] += 1
        valid_xml_rows.append(
            {
                "image_filename": image.name,
                "image_path": str(image.resolve()).replace("\\", "/"),
                "xml_filename": source_xml.name,
                "xml_original_path": str(source_xml.resolve()).replace("\\", "/"),
                "xml_new_path": str(dest.resolve()).replace("\\", "/"),
                "folder": folder,
                "action": "none" if source_xml.resolve() == dest.resolve() else args.mode,
                "dry_run": args.dry_run,
            }
        )

    for xml in xmls:
        if xml.resolve() in valid_xml_paths or xml.resolve() in used_source_xml_paths:
            continue
        folder = relative_folder(xml, dataset_root)
        target = unused_dir / xml.name
        dest = transfer_file(xml, target, args.unused_mode, args.dry_run)
        folder_stats[folder]["unused_xml"] += 1
        unused_xml_rows.append(
            {
                "xml_filename": xml.name,
                "xml_original_path": str(xml.resolve()).replace("\\", "/"),
                "xml_new_path": str(dest.resolve()).replace("\\", "/"),
                "folder": folder,
                "action": args.unused_mode,
                "dry_run": args.dry_run,
            }
        )

    report_dir = args.report_dir.resolve()
    ensure_inside_root(report_dir, PROJECT_ROOT)
    summary = {
        "created_at": now_iso(),
        "dry_run": args.dry_run,
        "dataset_root": str(dataset_root).replace("\\", "/"),
        "mode": args.mode,
        "unused_mode": args.unused_mode,
        "total_images": len(images),
        "total_xml_found": len(xmls),
        "total_valid_xml": len(valid_xml_rows),
        "total_unique_valid_xml": len(valid_xml_paths),
        "images_without_annotation": len(missing_xml_rows),
        "unused_annotations": len(unused_xml_rows),
        "per_folder": dict(sorted(folder_stats.items())),
        "reports": {
            "matched_xml": str((report_dir / "annotation_matched_xml.csv").resolve()).replace("\\", "/"),
            "images_without_xml": str((report_dir / "annotation_images_without_xml.csv").resolve()).replace("\\", "/"),
            "unused_xml": str((report_dir / "annotation_unused_xml.csv").resolve()).replace("\\", "/"),
            "summary": str((report_dir / "annotation_alignment_summary.json").resolve()).replace("\\", "/"),
        },
    }

    write_csv(
        report_dir / "annotation_matched_xml.csv",
        valid_xml_rows,
        [
            "image_filename",
            "image_path",
            "xml_filename",
            "xml_original_path",
            "xml_new_path",
            "folder",
            "action",
            "dry_run",
        ],
    )
    write_csv(
        report_dir / "annotation_images_without_xml.csv",
        missing_xml_rows,
        ["image_filename", "image_path", "folder", "expected_xml"],
    )
    write_csv(
        report_dir / "annotation_unused_xml.csv",
        unused_xml_rows,
        ["xml_filename", "xml_original_path", "xml_new_path", "folder", "action", "dry_run"],
    )
    with (report_dir / "annotation_alignment_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mueve/copia XML de US junto a su imagen correspondiente por nombre base."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--unused-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "unused_annotations",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument("--mode", choices=("move", "copy"), default="move")
    parser.add_argument("--unused-mode", choices=("move", "copy"), default="move")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = organize_annotations(parse_args())
    print("Resumen de alineacion XML/imagenes")
    print(f"Dataset              : {summary['dataset_root']}")
    print(f"Dry run              : {summary['dry_run']}")
    print(f"Total imagenes       : {summary['total_images']}")
    print(f"XML encontrados      : {summary['total_xml_found']}")
    print(f"XML validos          : {summary['total_valid_xml']}")
    print(f"XML validos unicos   : {summary['total_unique_valid_xml']}")
    print(f"Imagenes sin XML     : {summary['images_without_annotation']}")
    print(f"XML no utilizados    : {summary['unused_annotations']}")
    print("Por carpeta/FOV:")
    for folder, stats in summary["per_folder"].items():
        print(
            f"  {folder:18s} images={stats['images']:4d} "
            f"valid_xml={stats['valid_xml']:4d} "
            f"sin_xml={stats['images_without_xml']:4d} "
            f"unused_xml={stats['unused_xml']:4d}"
        )
    print("Reportes:")
    for path in summary["reports"].values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
