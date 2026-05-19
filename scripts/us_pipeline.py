"""
us_pipeline.py
==============
Pipeline de preparacion de datos del dominio destino para entrenamiento DANN.

Procesa imagenes de ultrasonido (.png / .jpg) organizadas por campo visual
(FOV) y las transforma en arrays .npy normalizados, isometricos (0.8 mm/px)
y de tamano fijo (256x256), compatibles con el dominio MRI generado por
mri_pipeline.py.
"""

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

US_DATA_PATH: str = os.getenv(
    "US_DATA_PATH", str(PROJECT_ROOT / "data" / "Ultrasound")
)
US_OUTPUT_PATH: str = os.getenv(
    "US_OUTPUT_PATH", str(PROJECT_ROOT / "data_ready_US")
)
IMAGE_SIZE: int = int(os.getenv("PROCESSOR_IMAGE_SIZE", "256"))

TARGET_SPACING_MM: float = 0.8

FOV_MAP: dict[str, float] = {
    "11cm": 110.0,
    "12cm": 120.0,
    "15cm": 150.0,
    "16cm": 160.0,
}

IMG_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")
INPAINT_RADIUS: int = int(os.getenv("US_INPAINT_RADIUS", "5"))

RANDOM_STATE: int = 42
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class BBox:
    """Caja Pascal VOC en coordenadas xyxy."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = "lesion"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "label": self.label,
            "xmin": round(float(self.xmin), 3),
            "ymin": round(float(self.ymin), 3),
            "xmax": round(float(self.xmax), 3),
            "ymax": round(float(self.ymax), 3),
        }

    def clipped(self, width: int, height: int) -> Optional["BBox"]:
        xmin = min(max(self.xmin, 0.0), float(width))
        xmax = min(max(self.xmax, 0.0), float(width))
        ymin = min(max(self.ymin, 0.0), float(height))
        ymax = min(max(self.ymax, 0.0), float(height))
        if xmax <= xmin or ymax <= ymin:
            return None
        return BBox(xmin, ymin, xmax, ymax, self.label)


@dataclass
class ProcessedUSSample:
    image: np.ndarray
    original_bboxes: list[BBox]
    transformed_bboxes: list[BBox]
    debug_steps: list[dict[str, object]]
    xml_path: Optional[str]


class PascalVOCReader:
    """Lee bounding boxes desde XML Pascal VOC asociado al nombre de la imagen."""

    @staticmethod
    def xml_path_for_image(img_path: str) -> Optional[Path]:
        xml_path = Path(img_path).with_suffix(".xml")
        return xml_path if xml_path.exists() else None

    @staticmethod
    def read(img_path: str) -> tuple[list[BBox], Optional[str]]:
        xml_path = PascalVOCReader.xml_path_for_image(img_path)
        if xml_path is None:
            return [], None

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            log.warning("XML invalido '%s': %s", xml_path, exc)
            return [], str(xml_path)

        boxes: list[BBox] = []
        for obj in root.findall("object"):
            label = obj.findtext("name", default="lesion")
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                boxes.append(
                    BBox(
                        xmin=float(bnd.findtext("xmin", "0")),
                        ymin=float(bnd.findtext("ymin", "0")),
                        xmax=float(bnd.findtext("xmax", "0")),
                        ymax=float(bnd.findtext("ymax", "0")),
                        label=label,
                    )
                )
            except ValueError:
                log.warning("BBox invalida en XML: %s", xml_path)
        return boxes, str(xml_path)


class BBoxTransformer:
    """Aplica a bboxes la misma geometria usada por el pipeline visual."""

    @staticmethod
    def crop(boxes: list[BBox], x: int, y: int, width: int, height: int) -> list[BBox]:
        result: list[BBox] = []
        for box in boxes:
            shifted = BBox(
                box.xmin - x,
                box.ymin - y,
                box.xmax - x,
                box.ymax - y,
                box.label,
            ).clipped(width, height)
            if shifted is not None:
                result.append(shifted)
        return result

    @staticmethod
    def scale(boxes: list[BBox], scale_x: float, scale_y: float) -> list[BBox]:
        return [
            BBox(
                box.xmin * scale_x,
                box.ymin * scale_y,
                box.xmax * scale_x,
                box.ymax * scale_y,
                box.label,
            )
            for box in boxes
        ]

    @staticmethod
    def tile(
        boxes: list[BBox],
        src_c0: int,
        src_r0: int,
        dst_c0: int,
        dst_r0: int,
        size: int,
    ) -> list[BBox]:
        result: list[BBox] = []
        for box in boxes:
            shifted = BBox(
                box.xmin - src_c0 + dst_c0,
                box.ymin - src_r0 + dst_r0,
                box.xmax - src_c0 + dst_c0,
                box.ymax - src_r0 + dst_r0,
                box.label,
            ).clipped(size, size)
            if shifted is not None:
                result.append(shifted)
        return result

    @staticmethod
    def transpose_after_final_orientation(boxes: list[BBox], size: int) -> list[BBox]:
        result: list[BBox] = []
        for box in boxes:
            transformed = BBox(
                xmin=box.ymin,
                ymin=box.xmin,
                xmax=box.ymax,
                ymax=box.xmax,
                label=box.label,
            ).clipped(size, size)
            if transformed is not None:
                result.append(transformed)
        return result


class ImageLoader:
    """Responsabilidad unica: cargar una imagen desde disco en formato BGR."""

    @staticmethod
    def load(path: str) -> Optional[np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("No se pudo cargar: %s", path)
        return img


class CalibrationMarksRemover:
    """
    Detecta y elimina textos perifericos, marcas de la interfaz y las cruces de
    medicion (calibradores) usando operadores morfologicos dirigidos.
    """

    def __init__(self, inpaint_radius: int = INPAINT_RADIUS) -> None:
        self.inpaint_radius = inpaint_radius

    def remove(self, bgr: np.ndarray) -> np.ndarray:
        h, w, _ = bgr.shape
        gray_temp = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        mask_local = cv2.adaptiveThreshold(
            gray_temp,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            -15,
        )

        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        mask_crosses = cv2.morphologyEx(gray_temp, cv2.MORPH_TOPHAT, kernel_cross)
        _, mask_crosses_bin = cv2.threshold(mask_crosses, 40, 255, cv2.THRESH_BINARY)

        _, mask_white = cv2.threshold(gray_temp, 220, 255, cv2.THRESH_BINARY)
        mask_final = cv2.bitwise_or(mask_local, mask_white)
        mask_final = cv2.bitwise_or(mask_final, mask_crosses_bin)

        periphery_mask = np.ones_like(mask_final) * 255
        r0, r1 = int(h * 0.15), int(h * 0.85)
        c0, c1 = int(w * 0.15), int(w * 0.85)

        center_elements = cv2.bitwise_and(
            mask_final[r0:r1, c0:c1],
            mask_crosses_bin[r0:r1, c0:c1],
        )
        periphery_mask[r0:r1, c0:c1] = 0

        mask_final = cv2.bitwise_and(mask_final, periphery_mask)
        mask_final[r0:r1, c0:c1] = cv2.bitwise_or(
            mask_final[r0:r1, c0:c1],
            center_elements,
        )

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_final = cv2.dilate(mask_final, kernel_dilate, iterations=1)

        if mask_final.any():
            bgr = cv2.inpaint(bgr, mask_final, self.inpaint_radius, cv2.INPAINT_TELEA)

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


class ROICropper:
    """
    Aisla el cono acustico ajustando analiticamente un sector circular real.
    Mantiene intacta la curvatura del arco inferior profundo.
    """

    def __init__(self) -> None:
        pass

    def crop_with_params(self, gray: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        h, w = gray.shape
        identity = {"x": 0, "y": 0, "width": w, "height": h}

        _, binary = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
        binary[0:int(h * 0.11), :] = 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=4,
        )
        if num_labels <= 1:
            return gray, identity

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        component_mask = np.zeros_like(gray, dtype=np.uint8)
        component_mask[labels == largest_label] = 255

        pts = cv2.findNonZero(component_mask)
        if pts is None:
            return gray, identity

        center_x = w // 2
        y_indices, x_indices = np.where(component_mask > 0)
        center_y = -int(h * 0.15)

        y_grid, x_grid = np.ogrid[:h, :w]
        dist_matrix = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

        valid_distances = dist_matrix[component_mask > 0]
        r_min = np.percentile(valid_distances, 1)
        r_max = np.percentile(valid_distances, 99)

        sector_mask = np.zeros_like(gray, dtype=np.uint8)
        sector_mask[(dist_matrix >= r_min) & (dist_matrix <= r_max)] = 255

        angles = np.arctan2(y_indices - center_y, x_indices - center_x)
        theta_min = np.percentile(angles, 0.5)
        theta_max = np.percentile(angles, 99.5)

        angle_matrix = np.arctan2(y_grid - center_y, x_grid - center_x)
        sector_mask[(angle_matrix < theta_min) | (angle_matrix > theta_max)] = 0

        gray_cleaned = cv2.bitwise_and(gray, sector_mask)
        active_pts = cv2.findNonZero(sector_mask)
        if active_pts is None:
            return gray, identity

        x, y, w_box, h_box = cv2.boundingRect(active_pts)
        if w_box < 10 or h_box < 10:
            return gray, identity

        return gray_cleaned[y:y + h_box, x:x + w_box], {
            "x": int(x),
            "y": int(y),
            "width": int(w_box),
            "height": int(h_box),
        }

    def crop(self, gray: np.ndarray) -> np.ndarray:
        cropped, _ = self.crop_with_params(gray)
        return cropped


class PhysicalResampler:
    """Redimensiona el cono usando un factor de escala basado en el FOV vertical."""

    def __init__(self, target_spacing: float = TARGET_SPACING_MM) -> None:
        self.target_spacing = target_spacing

    def resample_with_params(self, gray: np.ndarray, fov_mm: float) -> tuple[np.ndarray, dict[str, float | int]]:
        h_px, w_px = gray.shape
        new_h = max(1, round(fov_mm / self.target_spacing))
        scale_factor = new_h / h_px
        new_w = max(1, round(w_px * scale_factor))

        resampled = cv2.resize(
            gray.astype(np.float32),
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )
        return resampled, {
            "input_width": int(w_px),
            "input_height": int(h_px),
            "output_width": int(new_w),
            "output_height": int(new_h),
            "scale_x": float(new_w / w_px),
            "scale_y": float(new_h / h_px),
        }

    def resample(self, gray: np.ndarray, fov_mm: float) -> np.ndarray:
        resampled, _ = self.resample_with_params(gray, fov_mm)
        return resampled


class TileStandardizer:
    """
    Aplica normalizacion robusta por percentiles excluyendo ceros
    y centra la imagen en una matriz 256x256.
    """

    def __init__(self, size: int = IMAGE_SIZE) -> None:
        self.size = size

    def apply_with_params(self, img: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        valid_pixels = img[img > 0]
        if len(valid_pixels) > 0:
            vmin = np.percentile(valid_pixels, 1)
            vmax = np.percentile(valid_pixels, 99)

            img_norm = np.clip(img, vmin, vmax)
            if (vmax - vmin) > 0:
                img_norm = (img_norm - vmin) / (vmax - vmin)
            else:
                img_norm = np.zeros_like(img, dtype=np.float32)
        else:
            img_norm = np.zeros_like(img, dtype=np.float32)

        size = self.size
        canvas = np.zeros((size, size), dtype=np.float32)
        h, w = img_norm.shape

        if h <= size:
            pad_top = (size - h) // 2
            src_r0, src_r1 = 0, h
            dst_r0, dst_r1 = pad_top, pad_top + h
        else:
            crop_top = (h - size) // 2
            src_r0, src_r1 = crop_top, crop_top + size
            dst_r0, dst_r1 = 0, size

        if w <= size:
            pad_left = (size - w) // 2
            src_c0, src_c1 = 0, w
            dst_c0, dst_c1 = pad_left, pad_left + w
        else:
            crop_left = (w - size) // 2
            src_c0, src_c1 = crop_left, crop_left + size
            dst_c0, dst_c1 = 0, size

        canvas[dst_r0:dst_r1, dst_c0:dst_c1] = img_norm[src_r0:src_r1, src_c0:src_c1]

        return canvas, {
            "src_r0": int(src_r0),
            "src_r1": int(src_r1),
            "src_c0": int(src_c0),
            "src_c1": int(src_c1),
            "dst_r0": int(dst_r0),
            "dst_r1": int(dst_r1),
            "dst_c0": int(dst_c0),
            "dst_c1": int(dst_c1),
            "size": int(size),
        }

    def apply(self, img: np.ndarray) -> np.ndarray:
        canvas, _ = self.apply_with_params(img)
        return canvas


class OrientationModifier:
    """Rota y espeja la matriz final para alinear su orientacion anatomica con MRI."""

    @staticmethod
    def align_with_mri(img: np.ndarray) -> np.ndarray:
        rotated = np.rot90(img, k=-1)
        aligned = np.fliplr(rotated)
        return aligned


class BBoxDebugWriter:
    """Guarda metadata JSON e imagen de control con bbox transformada."""

    @staticmethod
    def draw_debug(image: np.ndarray, boxes: list[BBox], out_path: str) -> None:
        canvas = np.clip(image, 0.0, 1.0)
        canvas_u8 = (canvas * 255).astype(np.uint8)
        bgr = cv2.cvtColor(canvas_u8, cv2.COLOR_GRAY2BGR)
        for box in boxes:
            pt1 = (int(round(box.xmin)), int(round(box.ymin)))
            pt2 = (int(round(box.xmax)), int(round(box.ymax)))
            cv2.rectangle(bgr, pt1, pt2, (0, 255, 255), 2)
            cv2.putText(
                bgr,
                box.label,
                (pt1[0], max(0, pt1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, bgr)

    @staticmethod
    def save_json(payload: dict[str, object], out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)


class DataSplitter:
    """Divide los datos en subconjuntos y exporta el manifiesto de control."""

    def __init__(
        self,
        output_path: str,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.output_path = output_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split(self, paths: list[str]) -> dict[str, list[str]]:
        holdout_ratio = self.val_ratio + self.test_ratio
        train_paths, holdout_paths = train_test_split(
            paths,
            test_size=holdout_ratio,
            random_state=self.random_state,
            shuffle=True,
        )
        relative_test = self.test_ratio / holdout_ratio
        val_paths, test_paths = train_test_split(
            holdout_paths,
            test_size=relative_test,
            random_state=self.random_state,
            shuffle=True,
        )
        splits = {"train": train_paths, "val": val_paths, "test": test_paths}
        log.info(
            "Division: train=%d | val=%d | test=%d",
            len(train_paths),
            len(val_paths),
            len(test_paths),
        )
        return splits

    def build_dirs(self) -> dict[str, str]:
        dirs: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            img_dir = os.path.join(self.output_path, split_name, "images")
            bbox_dir = os.path.join(self.output_path, split_name, "bboxes")
            debug_dir = os.path.join(self.output_path, split_name, "debug_bboxes")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(bbox_dir, exist_ok=True)
            os.makedirs(debug_dir, exist_ok=True)
            dirs[split_name] = img_dir
            dirs[f"{split_name}_bboxes"] = bbox_dir
            dirs[f"{split_name}_debug_bboxes"] = debug_dir
        return dirs

    def save_manifest(
        self,
        splits: dict[str, list[str]],
        saved_names: dict[str, list[str]],
    ) -> None:
        manifest = {
            split_name: {
                "source_files": [os.path.basename(p) for p in paths],
                "npy_files": saved_names.get(split_name, []),
                "bbox_json_files": saved_names.get(f"{split_name}_bboxes", []),
                "bbox_debug_files": saved_names.get(f"{split_name}_debug_bboxes", []),
                "count": len(paths),
            }
            for split_name, paths in splits.items()
        }
        manifest_path = os.path.join(self.output_path, "us_splits.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        log.info("Manifiesto de splits guardado -> %s", manifest_path)


class UltrasoundPipeline:
    """Orquestador central del pipeline modular de ultrasonido."""

    def __init__(
        self,
        base_path: str = US_DATA_PATH,
        output_path: str = US_OUTPUT_PATH,
        image_size: int = IMAGE_SIZE,
        target_spacing: float = TARGET_SPACING_MM,
        inpaint_radius: int = INPAINT_RADIUS,
    ) -> None:
        self.base_path = base_path
        self.output_path = output_path

        self._loader = ImageLoader()
        self._remover = CalibrationMarksRemover(inpaint_radius)
        self._cropper = ROICropper()
        self._resampler = PhysicalResampler(target_spacing)
        self._tiler = TileStandardizer(image_size)
        self._rotator = OrientationModifier()
        self._splitter = DataSplitter(output_path)
        self._voc_reader = PascalVOCReader()
        self._bbox_writer = BBoxDebugWriter()

    def _discover_images(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for fov_name in FOV_MAP:
            fov_dir = os.path.join(self.base_path, fov_name)
            if not os.path.isdir(fov_dir):
                log.warning("Carpeta de FOV no encontrada: %s", fov_dir)
                continue
            paths = [
                p for ext in IMG_EXTENSIONS
                for p in glob(os.path.join(fov_dir, f"*{ext}"))
            ]
            if not paths:
                log.warning("No se encontraron imagenes en: %s", fov_dir)
                continue
            result[fov_name] = sorted(paths)
            log.info("FOV %-5s -> %d imagenes encontradas.", fov_name, len(paths))

        if not result:
            raise FileNotFoundError(
                f"No se encontraron imagenes en subcarpetas de: {self.base_path}"
            )
        return result

    def _process_single(self, img_path: str, fov_mm: float) -> Optional[ProcessedUSSample]:
        bgr = self._loader.load(img_path)
        if bgr is None:
            return None
        original_shape = bgr.shape[:2]
        original_bboxes, xml_path = self._voc_reader.read(img_path)
        boxes = list(original_bboxes)
        debug_steps: list[dict[str, object]] = [
            {
                "step": "original",
                "shape": [int(original_shape[0]), int(original_shape[1])],
                "bbox_count": len(boxes),
            }
        ]

        try:
            gray = self._remover.remove(bgr)
        except Exception as exc:
            log.warning("Error en inpainting '%s': %s. Saltando.", img_path, exc)
            return None

        try:
            cropped, crop_params = self._cropper.crop_with_params(gray)
            boxes = BBoxTransformer.crop(
                boxes,
                x=int(crop_params["x"]),
                y=int(crop_params["y"]),
                width=int(crop_params["width"]),
                height=int(crop_params["height"]),
            )
            debug_steps.append(
                {
                    "step": "roi_crop",
                    "params": crop_params,
                    "shape": [int(cropped.shape[0]), int(cropped.shape[1])],
                    "bbox_count": len(boxes),
                }
            )
        except Exception as exc:
            log.warning("Error en crop '%s': %s. Saltando.", img_path, exc)
            return None

        try:
            resampled, resample_params = self._resampler.resample_with_params(cropped, fov_mm)
            boxes = BBoxTransformer.scale(
                boxes,
                scale_x=float(resample_params["scale_x"]),
                scale_y=float(resample_params["scale_y"]),
            )
            debug_steps.append(
                {
                    "step": "physical_resample",
                    "params": resample_params,
                    "shape": [int(resampled.shape[0]), int(resampled.shape[1])],
                    "bbox_count": len(boxes),
                }
            )
        except Exception as exc:
            log.warning("Error en resampling '%s': %s. Saltando.", img_path, exc)
            return None

        tile, tile_params = self._tiler.apply_with_params(resampled)
        boxes = BBoxTransformer.tile(
            boxes,
            src_c0=int(tile_params["src_c0"]),
            src_r0=int(tile_params["src_r0"]),
            dst_c0=int(tile_params["dst_c0"]),
            dst_r0=int(tile_params["dst_r0"]),
            size=int(tile_params["size"]),
        )
        debug_steps.append(
            {
                "step": "tile_256",
                "params": tile_params,
                "shape": [int(tile.shape[0]), int(tile.shape[1])],
                "bbox_count": len(boxes),
            }
        )
        tile = self._rotator.align_with_mri(tile)
        boxes = BBoxTransformer.transpose_after_final_orientation(boxes, self._tiler.size)
        debug_steps.append(
            {
                "step": "final_orientation_rot90cw_fliplr",
                "equivalent": "transpose",
                "shape": [int(tile.shape[0]), int(tile.shape[1])],
                "bbox_count": len(boxes),
            }
        )

        return ProcessedUSSample(
            image=tile,
            original_bboxes=original_bboxes,
            transformed_bboxes=boxes,
            debug_steps=debug_steps,
            xml_path=xml_path,
        )

    def _save_array(self, array: np.ndarray, source_path: str, out_dir: str) -> str:
        fov_name = Path(source_path).parent.name
        stem = Path(source_path).stem
        filename = f"{fov_name}_{stem}.npy"
        np.save(os.path.join(out_dir, filename), array)
        return filename

    def _save_bbox_outputs(
        self,
        sample: ProcessedUSSample,
        source_path: str,
        npy_name: str,
        bbox_dir: str,
        debug_dir: str,
    ) -> tuple[str, str]:
        stem = Path(npy_name).stem
        json_name = f"{stem}.json"
        debug_name = f"{stem}.png"
        json_path = os.path.join(bbox_dir, json_name)
        debug_path = os.path.join(debug_dir, debug_name)
        payload = {
            "source_image": source_path,
            "source_xml": sample.xml_path,
            "processed_npy": npy_name,
            "image_size": [int(sample.image.shape[0]), int(sample.image.shape[1])],
            "bbox_original": [box.as_dict() for box in sample.original_bboxes],
            "bbox_256": [box.as_dict() for box in sample.transformed_bboxes],
            "transforms": sample.debug_steps,
        }
        self._bbox_writer.save_json(payload, json_path)
        self._bbox_writer.draw_debug(sample.image, sample.transformed_bboxes, debug_path)
        return json_name, debug_name

    def run(self) -> None:
        log.info("=" * 60)
        log.info("  US Pipeline - Inicio de procesamiento")
        log.info("  Base path : %s", self.base_path)
        log.info("  Output    : %s", self.output_path)
        log.info("  Spacing   : %.1f mm/px  |  Tile: %d px", TARGET_SPACING_MM, IMAGE_SIZE)
        log.info("=" * 60)

        images_by_fov = self._discover_images()

        all_paths: list[str] = []
        path_to_fov: dict[str, float] = {}
        for fov_name, paths in images_by_fov.items():
            fov_mm = FOV_MAP[fov_name]
            for p in paths:
                all_paths.append(p)
                path_to_fov[p] = fov_mm

        log.info("Total de imagenes descubiertas: %d", len(all_paths))

        splits = self._splitter.split(all_paths)
        out_dirs = self._splitter.build_dirs()

        total_stats: dict[str, int] = {}
        saved_names: dict[str, list[str]] = {s: [] for s in splits}
        for split_name in splits:
            saved_names[f"{split_name}_bboxes"] = []
            saved_names[f"{split_name}_debug_bboxes"] = []
        skipped_total = 0

        for split_name, split_paths in splits.items():
            saved_count = 0
            out_dir = out_dirs[split_name]
            bbox_dir = out_dirs[f"{split_name}_bboxes"]
            debug_dir = out_dirs[f"{split_name}_debug_bboxes"]
            desc = f"[{split_name.upper():5s}] Procesando"

            for img_path in tqdm(split_paths, desc=desc, unit="img"):
                fov_mm = path_to_fov[img_path]
                sample = self._process_single(img_path, fov_mm)

                if sample is None:
                    skipped_total += 1
                    continue

                npy_name = self._save_array(sample.image, img_path, out_dir)
                bbox_json_name, bbox_debug_name = self._save_bbox_outputs(
                    sample,
                    source_path=img_path,
                    npy_name=npy_name,
                    bbox_dir=bbox_dir,
                    debug_dir=debug_dir,
                )
                saved_names[split_name].append(npy_name)
                saved_names[f"{split_name}_bboxes"].append(bbox_json_name)
                saved_names[f"{split_name}_debug_bboxes"].append(bbox_debug_name)
                saved_count += 1

            total_stats[split_name] = saved_count
            log.info("Split %-5s -> %d imagenes guardadas.", split_name, saved_count)

        self._splitter.save_manifest(splits, saved_names)

        log.info("-" * 60)
        log.info("  RESUMEN FINAL")
        for split_name, count in total_stats.items():
            log.info("    %-6s : %d archivos .npy", split_name, count)
        log.info("  Total   : %d archivos .npy", sum(total_stats.values()))
        if skipped_total:
            log.warning("  Saltadas: %d imagenes por errores.", skipped_total)
        log.info("=" * 60)


if __name__ == "__main__":
    pipeline = UltrasoundPipeline(
        base_path=US_DATA_PATH,
        output_path=US_OUTPUT_PATH,
        image_size=IMAGE_SIZE,
        target_spacing=TARGET_SPACING_MM,
        inpaint_radius=INPAINT_RADIUS,
    )
    pipeline.run()
