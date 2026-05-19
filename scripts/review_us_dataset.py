"""
Interactive visual curation for processed ultrasound .npy files.

Default workflow:
    python scripts/review_us_dataset.py

Useful options:
    python scripts/review_us_dataset.py --review-all
    python scripts/review_us_dataset.py --source-root data_ready_US --show-hist

Keyboard shortcuts inside the matplotlib window:
    k = keep      d = discard   u = unsure
    n = next      p = previous  i/g = go to index
    m = note      z = undo      q = quit
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data_ready_US"
DEFAULT_PROGRESS_JSON = PROJECT_ROOT / "data_ready_US_review_progress.json"
DEFAULT_PROGRESS_CSV = PROJECT_ROOT / "data_ready_US_review_progress.csv"
DEFAULT_KEEP_ROOT = PROJECT_ROOT / "data_ready_US_clean"
DEFAULT_DISCARD_ROOT = PROJECT_ROOT / "data_ready_US_discarded"
DEFAULT_UNSURE_ROOT = PROJECT_ROOT / "data_ready_US_unsure"

DECISIONS = {"KEEP", "DISCARD", "UNSURE"}
KEY_TO_DECISION = {"k": "KEEP", "d": "DISCARD", "u": "UNSURE"}


@dataclass
class ReviewItem:
    path: Path
    relative_path: Path
    filename: str


@dataclass
class ReviewState:
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_key(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def load_state(progress_json: Path) -> ReviewState:
    if not progress_json.exists():
        return ReviewState()

    with progress_json.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, list):
        decisions = {
            str(row.get("path", "")).replace("\\", "/"): row
            for row in raw
            if row.get("path")
        }
        return ReviewState(decisions=decisions, history=[])

    return ReviewState(
        decisions=raw.get("decisions", {}),
        history=raw.get("history", []),
    )


def save_state(state: ReviewState, progress_json: Path, progress_csv: Path) -> None:
    progress_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now_iso(),
        "decisions": state.decisions,
        "history": state.history[-500:],
    }
    tmp_json = progress_json.with_suffix(progress_json.suffix + ".tmp")
    with tmp_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    tmp_json.replace(progress_json)

    rows = sorted(
        state.decisions.values(),
        key=lambda row: (row.get("decision", ""), row.get("relative_path", "")),
    )
    with progress_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "filename",
                "path",
                "relative_path",
                "decision",
                "timestamp",
                "notes",
                "copied_path",
                "source_file",
                "shape",
                "min",
                "max",
                "black_background_percent",
                "fov",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def discover_npy_files(source_root: Path) -> list[ReviewItem]:
    excluded_names = {
        DEFAULT_KEEP_ROOT.name,
        DEFAULT_DISCARD_ROOT.name,
        DEFAULT_UNSURE_ROOT.name,
        "data_ready_US_curated",
    }
    paths = []
    for path in source_root.rglob("*.npy"):
        if any(part in excluded_names for part in path.parts):
            continue
        paths.append(path)

    items = [
        ReviewItem(path=p, relative_path=p.relative_to(source_root), filename=p.name)
        for p in sorted(paths, key=lambda p: str(p).lower())
    ]
    if not items:
        raise FileNotFoundError(f"No se encontraron archivos .npy en {source_root}")
    return items


def load_split_manifest(source_root: Path) -> dict[str, str]:
    manifest_path = source_root / "us_splits.json"
    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    npy_to_source: dict[str, str] = {}
    for split_payload in manifest.values():
        npy_files = split_payload.get("npy_files", [])
        source_files = split_payload.get("source_files", [])
        for npy_name, source_name in zip(npy_files, source_files):
            npy_to_source[npy_name] = source_name
    return npy_to_source


def extract_fov(filename: str) -> str:
    match = re.search(r"(?i)(\d+(?:[.,]\d+)?)\s*cm", filename)
    return match.group(1).replace(",", ".") + "cm" if match else ""


def patient_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^\d+(?:[.,]\d+)?cm[_\-\s]*", "", stem, flags=re.IGNORECASE)
    numbers = re.findall(r"\d+", stem)
    if len(numbers) >= 2:
        return ".".join(numbers[:-1])
    return stem


def image_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"shape": list(array.shape), "min": None, "max": None, "black": None}

    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    black_threshold = 1e-6 if max_value <= 1.5 else 5.0
    black_percent = float(np.mean(finite <= black_threshold) * 100.0)
    return {
        "shape": list(array.shape),
        "min": round(min_value, 6),
        "max": round(max_value, 6),
        "black": round(black_percent, 2),
    }


def display_image(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def unique_destination(dest_root: Path, filename: str) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    candidate = dest_root / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while True:
        candidate = dest_root / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def remove_if_exists(path_text: str | None) -> None:
    if not path_text:
        return
    path = Path(path_text)
    if path.exists() and path.is_file():
        path.unlink()


def undo_transfer(record: dict[str, Any]) -> None:
    copied_path = record.get("copied_path")
    if not copied_path:
        return
    path = Path(copied_path)
    if not path.exists():
        return
    if record.get("transfer") == "move":
        original_path = Path(record["path"])
        original_path.parent.mkdir(parents=True, exist_ok=True)
        if not original_path.exists():
            shutil.move(str(path), str(original_path))
            return
    path.unlink()


def copy_or_move(
    src: Path,
    decision: str,
    roots: dict[str, Path],
    transfer: str,
) -> Path | None:
    decision_root = roots[decision] / "images"
    dest = unique_destination(decision_root, src.name)
    if transfer == "move":
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(src, dest)
    return dest


class DatasetReviewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.source_root = args.source_root.resolve()
        self.progress_json = args.progress_json.resolve()
        self.progress_csv = args.progress_csv.resolve()
        self.review_all = args.review_all
        self.transfer = args.transfer
        self.show_hist = args.show_hist
        self.manifest = load_split_manifest(self.source_root)
        self.items = discover_npy_files(self.source_root)
        self.state = load_state(self.progress_json)
        self.roots = {
            "KEEP": args.keep_root.resolve(),
            "DISCARD": args.discard_root.resolve(),
            "UNSURE": args.unsure_root.resolve(),
        }
        self.current = self._initial_index(args.start_index)
        self.fig: Any = None
        self.axes: list[Any] = []
        self.image_artist: Any = None
        self.hist_axis: Any = None

    def _initial_index(self, start_index: int | None) -> int:
        if start_index is not None:
            return max(0, min(start_index, len(self.items) - 1))
        if self.review_all:
            return 0
        for idx, item in enumerate(self.items):
            if normalize_key(item.path) not in self.state.decisions:
                return idx
        return 0

    def run(self) -> None:
        print("\nCurador visual US")
        print(f"Fuente: {self.source_root}")
        print(f"Total .npy: {len(self.items)}")
        print(f"Ya clasificados: {len(self.state.decisions)}")
        print("Atajos: k keep | d discard | u unsure | n next | p previous | i indice | m nota | z undo | q salir\n")

        ncols = 2 if self.show_hist else 1
        self.fig, axs = plt.subplots(1, ncols, figsize=(10 if self.show_hist else 7, 6))
        self.axes = list(np.atleast_1d(axs))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.refresh()
        plt.show()

    def current_item(self) -> ReviewItem:
        return self.items[self.current]

    def current_record(self) -> dict[str, Any] | None:
        return self.state.decisions.get(normalize_key(self.current_item().path))

    def refresh(self) -> None:
        item = self.current_item()
        try:
            array = np.load(item.path)
        except FileNotFoundError:
            self.axes[0].clear()
            self.axes[0].text(0.5, 0.5, "Archivo no encontrado", ha="center", va="center")
            self.fig.canvas.draw_idle()
            return

        arr = display_image(array)
        stats = image_stats(array)
        fov = extract_fov(item.filename)
        source_file = self.manifest.get(item.filename, "")
        record = self.current_record()
        decision = record["decision"] if record else "SIN CLASIFICAR"
        notes = record.get("notes", "") if record else ""

        self.axes[0].clear()
        self.axes[0].imshow(arr, cmap="gray", vmin=np.nanmin(arr), vmax=np.nanmax(arr))
        self.axes[0].axis("off")
        title = (
            f"{self.current + 1}/{len(self.items)} | {decision}\n"
            f"{item.filename}\n"
            f"shape={tuple(stats['shape'])}  min={stats['min']}  max={stats['max']}  "
            f"fondo negro={stats['black']}%  FOV={fov or 'NA'}"
        )
        if source_file:
            title += f"\noriginal={source_file}"
        if notes:
            title += f"\nnota={notes}"
        self.axes[0].set_title(title, fontsize=9)

        if self.show_hist:
            self.axes[1].clear()
            finite = arr[np.isfinite(arr)]
            self.axes[1].hist(finite.ravel(), bins=64, color="tab:blue", alpha=0.85)
            self.axes[1].set_title("Histograma", fontsize=9)

        self.fig.canvas.manager.set_window_title(f"US review - {item.filename}")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def go_next_unreviewed_or_next(self) -> None:
        if self.review_all:
            self.next()
            return
        for idx in range(self.current + 1, len(self.items)):
            if normalize_key(self.items[idx].path) not in self.state.decisions:
                self.current = idx
                self.refresh()
                return
        self.next()

    def next(self) -> None:
        self.current = min(self.current + 1, len(self.items) - 1)
        self.refresh()

    def previous(self) -> None:
        self.current = max(self.current - 1, 0)
        self.refresh()

    def mark(self, decision: str) -> None:
        item = self.current_item()
        key = normalize_key(item.path)
        previous = self.state.decisions.get(key)
        if previous:
            undo_transfer(previous)

        array = np.load(item.path)
        stats = image_stats(array)
        dest = copy_or_move(item.path, decision, self.roots, self.transfer)
        source_file = self.manifest.get(item.filename, "")

        record = {
            "filename": item.filename,
            "path": key,
            "relative_path": str(item.relative_path).replace("\\", "/"),
            "decision": decision,
            "timestamp": now_iso(),
            "notes": previous.get("notes", "") if previous else "",
            "copied_path": str(dest.resolve()).replace("\\", "/") if dest else "",
            "transfer": self.transfer,
            "source_file": source_file,
            "patient_id": patient_id_from_filename(item.filename),
            "shape": "x".join(str(x) for x in stats["shape"]),
            "min": stats["min"],
            "max": stats["max"],
            "black_background_percent": stats["black"],
            "fov": extract_fov(item.filename),
        }
        self.state.decisions[key] = record
        self.state.history.append({"path": key, "previous": previous, "new": record})
        save_state(self.state, self.progress_json, self.progress_csv)
        print(f"{decision}: {item.filename}")
        self.go_next_unreviewed_or_next()

    def undo(self) -> None:
        if not self.state.history:
            print("No hay acciones para deshacer.")
            return
        last = self.state.history.pop()
        key = last["path"]
        new_record = last.get("new") or {}
        previous = last.get("previous")
        undo_transfer(new_record)
        if previous:
            self.state.decisions[key] = previous
        else:
            self.state.decisions.pop(key, None)
        save_state(self.state, self.progress_json, self.progress_csv)
        for idx, item in enumerate(self.items):
            if normalize_key(item.path) == key:
                self.current = idx
                break
        print("Ultima accion deshecha.")
        self.refresh()

    def edit_note(self) -> None:
        item = self.current_item()
        key = normalize_key(item.path)
        record = self.state.decisions.get(key)
        if not record:
            print("Primero clasifica la imagen; luego se puede agregar nota.")
            return
        note = input("Nota para esta imagen (Enter para limpiar): ").strip()
        previous = dict(record)
        record["notes"] = note
        record["timestamp"] = now_iso()
        self.state.history.append({"path": key, "previous": previous, "new": dict(record)})
        save_state(self.state, self.progress_json, self.progress_csv)
        self.refresh()

    def goto_index(self) -> None:
        raw = input(f"Ir a indice 1-{len(self.items)}: ").strip()
        if not raw:
            return
        try:
            value = int(raw)
        except ValueError:
            print("Indice invalido.")
            return
        self.current = max(0, min(value - 1, len(self.items) - 1))
        self.refresh()

    def on_key(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key in KEY_TO_DECISION:
            self.mark(KEY_TO_DECISION[key])
        elif key == "n":
            self.next()
        elif key == "p":
            self.previous()
        elif key == "z":
            self.undo()
        elif key in {"i", "g"}:
            self.goto_index()
        elif key == "m":
            self.edit_note()
        elif key == "q":
            save_state(self.state, self.progress_json, self.progress_csv)
            plt.close(self.fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curacion visual interactiva de imagenes US procesadas en .npy."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--progress-json", type=Path, default=DEFAULT_PROGRESS_JSON)
    parser.add_argument("--progress-csv", type=Path, default=DEFAULT_PROGRESS_CSV)
    parser.add_argument("--keep-root", type=Path, default=DEFAULT_KEEP_ROOT)
    parser.add_argument("--discard-root", type=Path, default=DEFAULT_DISCARD_ROOT)
    parser.add_argument("--unsure-root", type=Path, default=DEFAULT_UNSURE_ROOT)
    parser.add_argument("--review-all", action="store_true", help="No saltear imagenes ya clasificadas.")
    parser.add_argument("--show-hist", action="store_true", help="Mostrar histograma junto a la imagen.")
    parser.add_argument("--start-index", type=int, default=None, help="Indice inicial base 1.")
    parser.add_argument(
        "--transfer",
        choices=("copy", "move"),
        default="copy",
        help="Copiar o mover a carpetas KEEP/DISCARD/UNSURE. Por defecto copia.",
    )
    args = parser.parse_args()
    if args.start_index is not None:
        args.start_index -= 1
    return args


def main() -> None:
    args = parse_args()
    reviewer = DatasetReviewer(args)
    reviewer.run()


if __name__ == "__main__":
    main()
