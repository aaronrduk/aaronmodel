"""
High-performance Multi-Model Ensemble Inference Engine.

Orchestrates:
- Specialized SOTA Segmentation (DeepLabV3+, DLinkNet, etc.)
- YOLOv8 Object Detection (Wells, Transformers, Tanks)
- EfficientNet Roof Classification
- Sequential execution for VRAM optimization
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from rio_tiler.io import Reader
except ImportError:
    warnings.warn("rio-tiler not installed.")

try:
    from ultralytics import YOLO
except ImportError:
    warnings.warn("ultralytics not installed. YOLOv8 features will be disabled.")
    YOLO = None

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _gaussian_kernel_2d(size: int, sigma: float = 0.0) -> np.ndarray:
    """Create a 2D Gaussian weighting kernel for tile blending."""
    if sigma <= 0:
        sigma = size / 4
    x = np.arange(size) - size / 2 + 0.5
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.max()


class TiledPredictor:
    """
    Ensemble inference engine for SVAMITVA V3 Architecture.
    """

    SEGMENTATION_TASKS = {
        "building_mask": "buildings",
        "road_mask": "roads",
        "waterbody_mask": "water",
        "utility_line_mask": "utilities",
        "railway_mask": "railway",
    }

    DETECTION_TASKS = ["waterbody_point_mask", "utility_point_mask"]

    def __init__(
        self,
        model: nn.Module,
        yolo_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        tile_size: int = 512,
        overlap: int = 128,
        threshold: float = 0.5,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold

        self.yolo = None
        if yolo_path and YOLO:
            self.yolo = YOLO(yolo_path)
            self.yolo.to(device)

        self.blend_kernel = _gaussian_kernel_2d(tile_size).astype(np.float32)

    def _get_valid_mask(self, tif_path: Path) -> np.ndarray:
        """Build a coarse valid-data mask from thumbnail."""
        try:
            with Reader(str(tif_path)) as dst:
                thumb = dst.preview(max_size=1024)
                mask = np.any(thumb.data > 0, axis=0)
                return mask
        except Exception as e:
            logger.warning(f"Thumbnail scan failed: {e}")
            return np.ones((1, 1), dtype=bool)

    @torch.no_grad()
    def predict_tif(
        self,
        tif_path: Path,
        selected_masks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run ensemble inference on a full GeoTIFF."""
        with Reader(str(tif_path)) as dst:
            H, W = dst.dataset.height, dst.dataset.width

        logger.info(f"Predicting {tif_path.name} with Ensemble V3: {W}x{H} px")
        masks_to_predict = selected_masks or list(self.SEGMENTATION_TASKS.keys())

        valid_thumb = self._get_valid_mask(tif_path)
        th_h, th_w = valid_thumb.shape
        scale_y, scale_x = th_h / H, th_w / W

        accumulators = {
            k: np.zeros((H, W), dtype=np.float32)
            for k in masks_to_predict
            if k in self.SEGMENTATION_TASKS
        }

        detections = []
        weight_map = np.zeros((H, W), dtype=np.float32)

        stride = self.tile_size - self.overlap
        windows = []
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                tw = min(self.tile_size, W - x)
                th = min(self.tile_size, H - y)
                tx, ty = int(x * scale_x), int(y * scale_y)
                if not valid_thumb[min(ty, th_h - 1), min(tx, th_w - 1)]:
                    continue
                windows.append((x, y, tw, th))

        for x0, y0, tw_act, th_act in tqdm(windows, desc="Ensemble Inference"):
            with Reader(str(tif_path)) as src:
                part_data = src.part(
                    (x0, y0, x0 + self.tile_size, y0 + self.tile_size),
                    width=self.tile_size,
                    height=self.tile_size,
                    padding=True,
                ).data

            tile_img = np.transpose(part_data, (1, 2, 0))
            blend = self.blend_kernel[:th_act, :tw_act]
            weight_map[y0 : y0 + th_act, x0 : x0 + tw_act] += blend

            for mask_key, task_name in self.SEGMENTATION_TASKS.items():
                if mask_key not in masks_to_predict:
                    continue
                preds = self._predict_single_task(tile_img, task=task_name)
                logits = preds.squeeze()[:th_act, :tw_act]
                accumulators[mask_key][y0 : y0 + th_act, x0 : x0 + tw_act] += (
                    logits * blend
                )

            if self.yolo:
                results = self.yolo.predict(tile_img, conf=0.25, verbose=False)
                for res in results:
                    for box in res.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        detections.append(
                            {
                                "box": [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0],
                                "class": int(box.cls[0]),
                                "conf": float(box.conf[0]),
                            }
                        )

        weight_map = np.maximum(weight_map, 1e-8)
        final_results = {}
        for key, accum in accumulators.items():
            final_results[key] = 1.0 / (1.0 + np.exp(-(accum / weight_map)))

        if "building_mask" in final_results and hasattr(self.model, "roof_model"):
            logger.info("Classifying building roof types...")
            final_results["roof_type_mask"] = self._classify_roofs(
                tif_path, final_results["building_mask"]
            )

        final_results["detections"] = detections
        return final_results

    def _predict_single_task(self, tile: np.ndarray, task: str) -> np.ndarray:
        img = tile.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        )
        with torch.no_grad():
            outputs = self.model(tensor, task=task)
            for val in outputs.values():
                return val.cpu().numpy()
        return np.zeros((1, 1, self.tile_size, self.tile_size))

    def _classify_roofs(self, tif_path: Path, building_mask: np.ndarray) -> np.ndarray:
        from skimage.measure import label, regionprops

        binary = (building_mask > self.threshold).astype(np.uint8)
        labels = label(binary)
        props = regionprops(labels)
        roof_output = np.zeros_like(building_mask, dtype=np.uint8)

        for prop in tqdm(props, desc="Roof Classification", leave=False):
            y1, x1, y2, x2 = prop.bbox
            with Reader(str(tif_path)) as src:
                pad = 15
                patch = src.part(
                    (
                        max(0, x1 - pad),
                        max(0, y1 - pad),
                        min(building_mask.shape[1], x2 + pad),
                        min(building_mask.shape[0], y2 + pad),
                    ),
                    width=224,
                    height=224,
                ).data

            p_tensor = (
                torch.from_numpy(patch).float().unsqueeze(0).to(self.device) / 255.0
            )
            p_tensor = (
                p_tensor - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device)
            ) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(self.device)
            with torch.no_grad():
                logits = self.model.roof_model(p_tensor)
                preds = torch.argmax(logits, dim=1)  # (1, 224, 224)
                # Take the majority class in the patch as the roof type
                class_id = torch.mode(preds.view(-1)).values.item()
                roof_output[labels == prop.label] = int(class_id)
        return roof_output


def load_ensemble_pipeline(
    weights_path: str,
    yolo_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> TiledPredictor:
    from models.model import EnsembleSvamitvaModel

    model = EnsembleSvamitvaModel(pretrained=False)
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    return TiledPredictor(model, yolo_path=yolo_path, device=device)
