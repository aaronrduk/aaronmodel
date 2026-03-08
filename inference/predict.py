"""
Tiled inference engine for SVAMITVA outputs.

Backbone segmentation model:
- SAM2 encoder + multi-head decoder for all raster tasks

Detection model:
- YOLOv8 for sparse point objects (wells/transformers/tanks),
  fused into point masks.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_np(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(np.clip(x, -50, 50))
    return ex / np.maximum(ex.sum(axis=axis, keepdims=True), 1e-8)


class TiledPredictor:
    """
    End-to-end tiled predictor for segmentation + point detection.
    """

    BINARY_MODEL_KEYS: List[str] = [
        "building_mask",
        "road_mask",
        "road_centerline_mask",
        "waterbody_mask",
        "waterbody_line_mask",
        "waterbody_point_mask",
        "utility_line_mask",
        "utility_point_mask",
        "bridge_mask",
        "railway_mask",
    ]
    ROOF_KEY = "roof_type_mask"
    POINT_KEYS: Set[str] = {"waterbody_point_mask", "utility_point_mask"}
    ALL_OUTPUT_KEYS: List[str] = BINARY_MODEL_KEYS + [ROOF_KEY]

    # YOLO class IDs -> target point masks
    YOLO_CLASS_TO_MASK = {
        0: "waterbody_point_mask",  # wells
        1: "utility_point_mask",  # transformers
        2: "utility_point_mask",  # tanks
    }

    def __init__(
        self,
        model: nn.Module,
        yolo_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        tile_size: int = 512,
        overlap: int = 128,
        threshold: float = 0.5,
        yolo_conf: float = 0.25,
        point_radius_px: int = 5,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold
        self.yolo_conf = yolo_conf
        self.point_radius_px = point_radius_px

        self.yolo = None
        if yolo_path and YOLO:
            self.yolo = YOLO(yolo_path)
            self.yolo.to(device)

        self.blend_kernel = _gaussian_kernel_2d(tile_size).astype(np.float32)

    def _get_valid_mask(self, tif_path: Path) -> np.ndarray:
        """Build a coarse valid-data mask from a thumbnail."""
        try:
            with Reader(str(tif_path)) as dst:
                thumb = dst.preview(max_size=1024)
                return np.any(thumb.data > 0, axis=0)
        except Exception as e:
            logger.warning("Thumbnail scan failed: %s", e)
            return np.ones((1, 1), dtype=bool)

    def _normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Convert tile to normalized RGB float32 tensor input in ImageNet space.
        Expects tile as HxWxC.
        """
        if tile.ndim != 3:
            raise ValueError(f"Expected HxWxC tile, got shape {tile.shape}")

        if tile.shape[2] == 1:
            tile = np.repeat(tile, 3, axis=2)
        elif tile.shape[2] > 3:
            tile = tile[:, :, :3]
        elif tile.shape[2] == 2:
            tile = np.concatenate([tile, tile[:, :, :1]], axis=2)

        tile = tile.astype(np.float32)
        tmax = float(np.max(tile)) if tile.size > 0 else 1.0
        if tmax > 1.0:
            if tmax <= 255.0:
                tile = tile / 255.0
            elif tmax <= 65535.0:
                tile = tile / 65535.0
            else:
                tmin = float(np.min(tile))
                tile = (tile - tmin) / max(tmax - tmin, 1e-6)

        tile = np.clip(tile, 0.0, 1.0)
        tile = (tile - IMAGENET_MEAN) / IMAGENET_STD
        return tile

    def _predict_tile_model(self, tile_img: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the segmentation model for one tile and return numpy outputs."""
        img = self._normalize_tile(tile_img)
        tensor = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        )
        with torch.no_grad():
            outputs = self.model(tensor, task="all")

        out_np: Dict[str, np.ndarray] = {}
        for k, v in outputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            arr = v.detach().cpu().numpy()
            if arr.ndim >= 3:
                arr = arr[0]  # drop batch dim
            out_np[k] = arr
        return out_np

    @torch.no_grad()
    def predict_tif(
        self,
        tif_path: Path,
        selected_masks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run full tiled inference for all requested outputs."""
        selected = set(selected_masks or self.ALL_OUTPUT_KEYS)
        selected = {k for k in selected if k in set(self.ALL_OUTPUT_KEYS)}
        if not selected:
            selected = set(self.ALL_OUTPUT_KEYS)

        valid_thumb = self._get_valid_mask(tif_path)

        with Reader(str(tif_path)) as src:
            H, W = src.dataset.height, src.dataset.width
            logger.info("Predicting %s (%dx%d)", tif_path.name, W, H)

            th_h, th_w = valid_thumb.shape
            scale_y, scale_x = th_h / H, th_w / W

            model_accum = {
                k: np.zeros((H, W), dtype=np.float32)
                for k in self.BINARY_MODEL_KEYS
                if k in selected
            }
            roof_accum = (
                np.zeros((5, H, W), dtype=np.float32)
                if self.ROOF_KEY in selected
                else None
            )
            det_point_masks = {
                k: np.zeros((H, W), dtype=np.float32)
                for k in self.POINT_KEYS
                if k in selected
            }
            detections: List[Dict[str, Any]] = []
            weight_map = np.zeros((H, W), dtype=np.float32)

            stride = max(1, self.tile_size - self.overlap)
            windows = []
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    tw = min(self.tile_size, W - x)
                    th = min(self.tile_size, H - y)
                    tx, ty = int(x * scale_x), int(y * scale_y)
                    if not valid_thumb[min(ty, th_h - 1), min(tx, th_w - 1)]:
                        continue
                    windows.append((x, y, tw, th))

            for x0, y0, tw_act, th_act in tqdm(windows, desc="Inference", leave=False):
                part = src.part(
                    (x0, y0, x0 + self.tile_size, y0 + self.tile_size),
                    width=self.tile_size,
                    height=self.tile_size,
                    padding=True,
                ).data
                tile_img = np.transpose(part, (1, 2, 0))
                blend = self.blend_kernel[:th_act, :tw_act]
                weight_map[y0 : y0 + th_act, x0 : x0 + tw_act] += blend

                # SAM2 multi-task segmentation
                model_outputs = self._predict_tile_model(tile_img)

                for key in list(model_accum.keys()):
                    if key not in model_outputs:
                        continue
                    logits = model_outputs[key]
                    if logits.ndim == 3 and logits.shape[0] == 1:
                        logits2d = logits[0]
                    elif logits.ndim == 2:
                        logits2d = logits
                    else:
                        continue
                    prob = _sigmoid_np(logits2d[:th_act, :tw_act])
                    model_accum[key][y0 : y0 + th_act, x0 : x0 + tw_act] += prob * blend

                if roof_accum is not None and self.ROOF_KEY in model_outputs:
                    roof_logits = model_outputs[self.ROOF_KEY]
                    if roof_logits.ndim == 3 and roof_logits.shape[0] >= 2:
                        roof_probs = _softmax_np(
                            roof_logits[:, :th_act, :tw_act], axis=0
                        )
                        roof_accum[:, y0 : y0 + th_act, x0 : x0 + tw_act] += (
                            roof_probs * blend[None]
                        )

                # YOLO point detection (wells / transformers / tanks)
                if self.yolo is not None and det_point_masks:
                    tile_u8 = np.clip(tile_img, 0, 255).astype(np.uint8)
                    yolo_results = self.yolo.predict(
                        tile_u8, conf=self.yolo_conf, verbose=False
                    )
                    for res in yolo_results:
                        for box in res.boxes:
                            cls_id = int(box.cls[0])
                            mask_key = self.YOLO_CLASS_TO_MASK.get(cls_id)
                            if mask_key not in det_point_masks:
                                continue

                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                            gx1, gy1 = bx1 + x0, by1 + y0
                            gx2, gy2 = bx2 + x0, by2 + y0
                            cx = int((gx1 + gx2) / 2.0)
                            cy = int((gy1 + gy2) / 2.0)
                            radius = max(
                                self.point_radius_px,
                                int(max(gx2 - gx1, gy2 - gy1) * 0.25),
                            )

                            try:
                                import cv2

                                cv2.circle(
                                    det_point_masks[mask_key],
                                    (cx, cy),
                                    radius,
                                    color=1.0,
                                    thickness=-1,
                                )
                            except Exception:
                                if 0 <= cy < H and 0 <= cx < W:
                                    det_point_masks[mask_key][cy, cx] = 1.0

                            detections.append(
                                {
                                    "box": [float(gx1), float(gy1), float(gx2), float(gy2)],
                                    "class": cls_id,
                                    "conf": float(box.conf[0]),
                                    "mask_key": mask_key,
                                }
                            )

        weight_map = np.maximum(weight_map, 1e-8)
        final_results: Dict[str, Any] = {}

        # Finalize binary probability masks
        for key, accum in model_accum.items():
            final_results[key] = accum / weight_map

        # Finalize roof class map
        if roof_accum is not None:
            roof_probs = roof_accum / weight_map[None]
            roof_mask = np.argmax(roof_probs, axis=0).astype(np.uint8)
            if "building_mask" in final_results:
                roof_mask[final_results["building_mask"] <= self.threshold] = 0
            final_results[self.ROOF_KEY] = roof_mask

        # Fuse YOLO points with segmentation point masks
        for point_key in self.POINT_KEYS:
            if point_key not in selected:
                continue
            seg_prob = final_results.get(point_key)
            det_prob = det_point_masks.get(point_key)
            if det_prob is not None and seg_prob is not None:
                final_results[point_key] = np.maximum(seg_prob, det_prob)
            elif det_prob is not None:
                final_results[point_key] = det_prob
            elif seg_prob is None:
                final_results[point_key] = np.zeros((H, W), dtype=np.float32)

        # Fill any missing requested output key with zeros
        for key in selected:
            if key not in final_results:
                dtype = np.uint8 if key == self.ROOF_KEY else np.float32
                final_results[key] = np.zeros((H, W), dtype=dtype)

        final_results["detections"] = detections
        return final_results


def _resolve_weights_path(weights_path: str) -> Optional[Path]:
    p = Path(weights_path)
    if p.exists():
        return p
    fallback_candidates = [
        Path("checkpoints/best.pt"),
        Path("checkpoints/best_latest.pt"),
    ]
    for cand in fallback_candidates:
        if cand.exists():
            logger.warning("Weights not found at %s; using fallback %s", p, cand)
            return cand
    logger.warning("No ensemble weights found at %s or fallback paths.", p)
    return None


def load_ensemble_pipeline(
    weights_path: str,
    yolo_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> TiledPredictor:
    from models.model import EnsembleSvamitvaModel

    model = EnsembleSvamitvaModel(pretrained=True)
    resolved_weights = _resolve_weights_path(weights_path)
    if resolved_weights is not None:
        ckpt = torch.load(resolved_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        logger.info("Loaded ensemble weights from %s", resolved_weights)

    return TiledPredictor(model, yolo_path=yolo_path, device=device)
