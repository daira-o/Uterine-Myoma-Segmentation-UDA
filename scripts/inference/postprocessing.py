"""Postprocessing robusto para mascaras US.

El bbox se usa solo como guia anatomica: nunca se aplica como mascara dura ni
se recorta la prediccion al rectangulo.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage import measure, morphology


@dataclass(frozen=True)
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = "lesion"

    def clipped(self, width: int = 256, height: int = 256) -> "BBox | None":
        xmin = min(max(float(self.xmin), 0.0), float(width))
        xmax = min(max(float(self.xmax), 0.0), float(width))
        ymin = min(max(float(self.ymin), 0.0), float(height))
        ymax = min(max(float(self.ymax), 0.0), float(height))
        if xmax <= xmin or ymax <= ymin:
            return None
        return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=self.label)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)


@dataclass
class ComponentInfo:
    label: int
    area: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    bbox_overlap_px: int
    expanded_bbox_overlap_px: int
    centroid_inside_bbox: bool | None
    distance_to_bbox_center_px: float | None


@dataclass
class PostprocessResult:
    mask_before: np.ndarray
    mask_after_small_objects: np.ndarray
    mask_final: np.ndarray
    expanded_bboxes: list[BBox]
    components_before: list[ComponentInfo]
    components_after_small_objects: list[ComponentInfo]
    selected_component: ComponentInfo | None
    selected_reason: str
    stats: dict[str, Any]


def bbox_json_path(image_path: Path) -> Path:
    if image_path.parent.name == "images":
        return image_path.parent.parent / "bboxes" / f"{image_path.stem}.json"
    return image_path.parent / "bboxes" / f"{image_path.stem}.json"


def rows_to_bboxes(rows: Any, shape: tuple[int, int] = (256, 256)) -> list[BBox]:
    boxes: list[BBox] = []
    if not isinstance(rows, list):
        return boxes
    height, width = shape
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            box = BBox(
                xmin=float(row["xmin"]),
                ymin=float(row["ymin"]),
                xmax=float(row["xmax"]),
                ymax=float(row["ymax"]),
                label=str(row.get("label", "lesion")),
            ).clipped(width=width, height=height)
        except (KeyError, TypeError, ValueError):
            continue
        if box is not None:
            boxes.append(box)
    return boxes


def load_bboxes_for_image(image_path: Path, shape: tuple[int, int] = (256, 256)) -> list[BBox]:
    path = bbox_json_path(image_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return rows_to_bboxes(payload.get("bbox_256", []), shape=shape)[:1]


def expand_bboxes(
    boxes: list[BBox],
    margin_px: int,
    shape: tuple[int, int],
) -> list[BBox]:
    height, width = shape
    expanded: list[BBox] = []
    for box in boxes:
        new_box = BBox(
            xmin=box.xmin - margin_px,
            ymin=box.ymin - margin_px,
            xmax=box.xmax + margin_px,
            ymax=box.ymax + margin_px,
            label=box.label,
        ).clipped(width=width, height=height)
        if new_box is not None:
            expanded.append(new_box)
    return expanded


def boxes_to_mask(boxes: list[BBox], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for box in boxes:
        xmin = min(max(int(math.floor(box.xmin)), 0), width)
        xmax = min(max(int(math.ceil(box.xmax)), 0), width)
        ymin = min(max(int(math.floor(box.ymin)), 0), height)
        ymax = min(max(int(math.ceil(box.ymax)), 0), height)
        if xmax > xmin and ymax > ymin:
            mask[ymin:ymax, xmin:xmax] = True
    return mask


def bbox_center(boxes: list[BBox], shape: tuple[int, int]) -> tuple[float, float] | None:
    if not boxes:
        return None
    guide = boxes_to_mask(boxes, shape)
    points = np.argwhere(guide)
    if len(points) == 0:
        return None
    cy, cx = points.mean(axis=0)
    return float(cx), float(cy)


def component_infos(
    mask: np.ndarray,
    boxes: list[BBox],
    expanded_boxes: list[BBox],
) -> list[ComponentInfo]:
    binary = mask.astype(bool)
    labeled = measure.label(binary, connectivity=2)
    props = measure.regionprops(labeled)
    bbox_mask = boxes_to_mask(boxes, binary.shape) if boxes else np.zeros(binary.shape, dtype=bool)
    expanded_mask = (
        boxes_to_mask(expanded_boxes, binary.shape)
        if expanded_boxes
        else np.zeros(binary.shape, dtype=bool)
    )
    center = bbox_center(boxes, binary.shape)

    infos: list[ComponentInfo] = []
    for prop in props:
        comp = labeled == prop.label
        cy, cx = prop.centroid
        if boxes:
            y = int(np.clip(round(cy), 0, binary.shape[0] - 1))
            x = int(np.clip(round(cx), 0, binary.shape[1] - 1))
            centroid_inside = bool(bbox_mask[y, x])
        else:
            centroid_inside = None
        if center is None:
            distance = None
        else:
            distance = float(math.hypot(cx - center[0], cy - center[1]))
        infos.append(
            ComponentInfo(
                label=int(prop.label),
                area=int(prop.area),
                centroid=(float(cx), float(cy)),
                bbox=tuple(int(v) for v in prop.bbox),
                bbox_overlap_px=int((comp & bbox_mask).sum()),
                expanded_bbox_overlap_px=int((comp & expanded_mask).sum()),
                centroid_inside_bbox=centroid_inside,
                distance_to_bbox_center_px=distance,
            )
        )
    return infos


def remove_small_objects_guided(
    mask: np.ndarray,
    min_area_px: int,
    boxes: list[BBox],
    expanded_boxes: list[BBox],
) -> np.ndarray:
    binary = mask.astype(bool)
    if min_area_px <= 1 or not binary.any():
        return binary.astype(np.uint8)

    labeled = measure.label(binary, connectivity=2)
    props = measure.regionprops(labeled)
    expanded_mask = (
        boxes_to_mask(expanded_boxes, binary.shape)
        if expanded_boxes
        else np.zeros(binary.shape, dtype=bool)
    )
    keep_labels = []
    for prop in props:
        comp = labeled == prop.label
        has_bbox_support = bool(boxes and (comp & expanded_mask).any())
        if prop.area >= min_area_px or has_bbox_support:
            keep_labels.append(prop.label)
    cleaned = np.isin(labeled, keep_labels)
    return cleaned.astype(np.uint8)


def select_component(
    mask: np.ndarray,
    boxes: list[BBox],
    expanded_boxes: list[BBox],
) -> tuple[np.ndarray, ComponentInfo | None, str, list[int]]:
    binary = mask.astype(bool)
    labeled = measure.label(binary, connectivity=2)
    infos = component_infos(binary, boxes, expanded_boxes)
    if not infos:
        return np.zeros_like(mask, dtype=np.uint8), None, "empty_mask", []

    if not boxes:
        selected = max(infos, key=lambda info: info.area)
        reason = "no_bbox_keep_largest"
    else:
        intersecting = [info for info in infos if info.expanded_bbox_overlap_px > 0]
        if intersecting:
            selected = max(
                intersecting,
                key=lambda info: (info.expanded_bbox_overlap_px, info.bbox_overlap_px, info.area),
            )
            reason = "max_expanded_bbox_overlap"
        else:
            selected = min(
                infos,
                key=lambda info: (
                    float("inf")
                    if info.distance_to_bbox_center_px is None
                    else info.distance_to_bbox_center_px,
                    -info.area,
                ),
            )
            reason = "closest_to_bbox_center"

    selected_mask = (labeled == selected.label).astype(np.uint8)
    discarded = [info.label for info in infos if info.label != selected.label]
    return selected_mask, selected, reason, discarded


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    if not binary.any():
        return binary.astype(np.uint8)
    labeled = measure.label(binary, connectivity=2)
    props = measure.regionprops(labeled)
    if not props:
        return np.zeros_like(mask, dtype=np.uint8)
    largest = max(props, key=lambda prop: prop.area)
    return (labeled == largest.label).astype(np.uint8)


def morphological_close(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    binary = mask.astype(bool)
    if kernel_px <= 1 or not binary.any():
        return binary.astype(np.uint8)
    if kernel_px % 2 == 0:
        kernel_px += 1
    radius = max(1, kernel_px // 2)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    footprint = ((xx / radius) ** 2 + (yy / radius) ** 2) <= 1.0
    closed = morphology.closing(binary, footprint=footprint)
    return closed.astype(np.uint8)


def bbox_in_ratio(mask: np.ndarray, boxes: list[BBox]) -> float | None:
    pred = mask.astype(bool)
    area = int(pred.sum())
    if not boxes:
        return None
    if area == 0:
        return 0.0
    bbox_mask = boxes_to_mask(boxes, pred.shape)
    return float((pred & bbox_mask).sum() / area)


def postprocess_prediction(
    prob: np.ndarray,
    boxes: list[BBox] | None = None,
    threshold: float = 0.5,
    min_area_px: int = 50,
    bbox_margin_px: int = 8,
    closing_kernel_px: int = 3,
) -> PostprocessResult:
    boxes = (boxes or [])[:1]
    mask_before = (prob >= threshold).astype(np.uint8)
    original_area = int(mask_before.sum())
    expanded_boxes = expand_bboxes(boxes, bbox_margin_px, mask_before.shape)
    components_before = component_infos(mask_before, boxes, expanded_boxes)

    after_small = remove_small_objects_guided(
        mask_before,
        min_area_px,
        boxes,
        expanded_boxes,
    )
    components_after_small = component_infos(after_small, boxes, expanded_boxes)
    if boxes and not components_after_small and components_before:
        # Si min_area_px fue demasiado agresivo, se permite elegir desde la
        # mascara original usando la bbox como guia, no el area como criterio.
        selection_mask = mask_before
        selection_components = components_before
    else:
        selection_mask = after_small
        selection_components = components_after_small

    selected_mask, selected_component, selected_reason, discarded_labels = select_component(
        selection_mask,
        boxes,
        expanded_boxes,
    )
    closed = morphological_close(selected_mask, closing_kernel_px)
    filled = ndi.binary_fill_holes(closed.astype(bool)).astype(np.uint8)
    filled = keep_largest_component(filled)

    final_components = component_infos(filled, boxes, expanded_boxes)
    component_overlaps = [
        {
            "label": info.label,
            "area": info.area,
            "overlap_expanded_bbox_px": info.expanded_bbox_overlap_px,
            "overlap_bbox_px": info.bbox_overlap_px,
            "centroid": info.centroid,
            "distance_to_bbox_center_px": info.distance_to_bbox_center_px,
        }
        for info in selection_components
    ]
    component_overlaps_before = [
        {
            "label": info.label,
            "area": info.area,
            "overlap_expanded_bbox_px": info.expanded_bbox_overlap_px,
            "overlap_bbox_px": info.bbox_overlap_px,
            "centroid": info.centroid,
            "distance_to_bbox_center_px": info.distance_to_bbox_center_px,
        }
        for info in components_before
    ]
    selected_label = selected_component.label if selected_component else None
    removed_by_small = max(0, len(components_before) - len(components_after_small))
    stats = {
        "threshold": float(threshold),
        "min_area_px": int(min_area_px),
        "bbox_margin_px": int(bbox_margin_px),
        "closing_kernel_px": int(closing_kernel_px),
        "objects_before": len(components_before),
        "objects_after_small_objects": len(components_after_small),
        "objects_for_selection": len(selection_components),
        "objects_final": len(final_components),
        "area_before": original_area,
        "area_after_small_objects": int(after_small.sum()),
        "area_final": int(filled.sum()),
        "selected_component_label": selected_label,
        "selected_component_area": selected_component.area if selected_component else 0,
        "selected_reason": selected_reason,
        "component_overlaps_before": component_overlaps_before,
        "component_overlaps": component_overlaps,
        "discarded_component_labels": discarded_labels,
        "discarded_components_count": len(discarded_labels),
        "removed_by_small_objects_count": removed_by_small,
        "selected_overlap_bbox_px": (
            selected_component.bbox_overlap_px if selected_component else 0
        ),
        "selected_overlap_expanded_bbox_px": (
            selected_component.expanded_bbox_overlap_px if selected_component else 0
        ),
        "selected_centroid": selected_component.centroid if selected_component else None,
        "centroid_inside_bbox": (
            selected_component.centroid_inside_bbox if selected_component else None
        ),
        "distance_to_bbox_center_px": (
            selected_component.distance_to_bbox_center_px if selected_component else None
        ),
        "bbox_in_ratio": bbox_in_ratio(filled, boxes),
    }
    return PostprocessResult(
        mask_before=mask_before,
        mask_after_small_objects=after_small,
        mask_final=filled,
        expanded_bboxes=expanded_boxes,
        components_before=components_before,
        components_after_small_objects=components_after_small,
        selected_component=selected_component,
        selected_reason=selected_reason,
        stats=stats,
    )


def safe_stem(value: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).stem)


def draw_bboxes(ax: plt.Axes, boxes: list[BBox], color: str, linewidth: float = 1.8) -> None:
    for box in boxes:
        ax.add_patch(
            Rectangle(
                (box.xmin, box.ymin),
                box.xmax - box.xmin,
                box.ymax - box.ymin,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )


def save_postprocessing_debug_overlay(
    image_np: np.ndarray,
    result: PostprocessResult,
    boxes: list[BBox],
    output_dir: Path,
    stem: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_stem(stem)}_postprocessing_debug.png"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="white", constrained_layout=True)
    panels = [
        ("US + bbox", None, None),
        ("Antes postprocessing", result.mask_before, "autumn"),
        ("Mascara final", result.mask_final, "cool"),
    ]
    for ax, (title, overlay_mask, cmap) in zip(axes, panels):
        ax.imshow(image_np, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if overlay_mask is not None:
            masked = np.ma.masked_where(overlay_mask == 0, overlay_mask)
            ax.imshow(masked, cmap=cmap, alpha=0.42, interpolation="nearest")
        draw_bboxes(ax, boxes, color="#00d1b2", linewidth=1.6)
        draw_bboxes(ax, result.expanded_bboxes, color="#ffb000", linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        (
            f"objects {result.stats['objects_before']} -> {result.stats['objects_final']} | "
            f"area {result.stats['area_before']} -> {result.stats['area_final']} | "
            f"selected={result.stats['selected_component_label']} "
            f"({result.stats['selected_reason']})"
        ),
        fontsize=10,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
