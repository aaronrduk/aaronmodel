"""
Tiled inference engine for large GeoTIFF orthophotos.

Runs the model on overlapping tiles and blends results with
Gaussian weighting to produce seamless full-image predictions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    compute_tile_windows,
    read_geotiff_meta,
    read_tile,
    compute_global_stretch,
)

logger = logging.getLogger(__name__)


# ── Gaussian Blending Weights ─────────────────────────────────────────────────


def _gaussian_kernel_2d(size: int, sigma: float = 0.0) -> np.ndarray:
    """Create a 2D Gaussian weighting kernel for tile blending."""
    if sigma <= 0:
        sigma = size / 4
    x = np.arange(size) - size / 2 + 0.5
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.max()


# ── Predictor ─────────────────────────────────────────────────────────────────


class TiledPredictor:
    """
    Run inference on full-size orthophotos using overlapping tiles.

    Args:
        model: trained SvamitvaModel
        device: cuda/mps/cpu
        tile_size: tile dimension (must match training)
        overlap: tile overlap in pixels
        threshold: sigmoid threshold for binary masks
    """

    BINARY_MASKS = [
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

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        tile_size: int = 512,
        overlap: int = 64,
        threshold: float = 0.5,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold

        # Pre-compute blending kernel
        self.blend_kernel = _gaussian_kernel_2d(tile_size).astype(np.float32)

    @torch.no_grad()
    def predict_tif(
        self,
        tif_path: Path,
        selected_masks: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run tiled inference on a full GeoTIFF.

        Args:
            tif_path: path to input GeoTIFF
            selected_masks: optional subset of mask keys to predict

        Returns:
            Dict of mask_key → (H, W) float32 probability maps
        """
        meta = read_geotiff_meta(tif_path)
        H, W = meta["height"], meta["width"]
        logger.info(f"Predicting {tif_path.name}: {W}×{H} px")

        # Compute global stretch params for this map
        stretch = (
            compute_global_stretch(tif_path) if meta["dtype"] != np.uint8 else None
        )

        # Nodata mask to zero-out predictions on black areas
        nodata_map = np.zeros((H, W), dtype=bool)

        # Initialize logit accumulators (initialize with high negative for logit-space sum)
        # Binary tasks: logit(p/(1-p)) -> we add weighted logits
        logit_accum = {k: np.zeros((H, W), dtype=np.float32) for k in self.BINARY_MASKS}

        # Multi-class (roof type): we'll store (C, H, W) logits
        roof_logit_accum = (
            np.zeros((5, H, W), dtype=np.float32)
            if "roof_type_mask" in selected_masks
            else None
        )

        weight_map = np.zeros((H, W), dtype=np.float32)

        # Compute tile windows
        windows = compute_tile_windows(W, H, self.tile_size, self.overlap)
        logger.info(f"  Processing {len(windows)} tiles...")

        for win in tqdm(windows, desc="Inference", dynamic_ncols=True):
            tile_img, _ = read_tile(tif_path, win, stretch_params=stretch)
            th, tw = tile_img.shape[:2]

            # Update nodata map
            y, x = int(win.row_off), int(win.col_off)
            nodata_map[y : y + th, x : x + tw] |= tile_img.sum(axis=-1) == 0

            # Pad if needed
            if th < self.tile_size or tw < self.tile_size:
                padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile_img
                tile_img = padded

            # Run model - returns RAW LOGITS (no sigmoid/softmax)
            preds = self._predict_single_tile(tile_img, return_logits=True)

            # Blending
            blend = self.blend_kernel[:th, :tw]

            # Accumulate logits
            for key in selected_masks:
                if key == "roof_type_mask" and roof_logit_accum is not None:
                    logits = preds["roof_type_mask"][:, :th, :tw]
                    roof_logit_accum[:, y : y + th, x : x + tw] += (
                        logits * blend[np.newaxis]
                    )
                elif key in preds:
                    logits = preds[key][:th, :tw]
                    logit_accum[key][y : y + th, x : x + tw] += logits * blend

            weight_map[y : y + th, x : x + tw] += blend

        # Final normalization and activation
        weight_map = np.maximum(weight_map, 1e-8)
        results = {}

        for key in selected_masks:
            if key == "roof_type_mask" and roof_logit_accum is not None:
                for c in range(5):
                    roof_logit_accum[c] /= weight_map
                # Set nodata area to background (class 0) by boosting background logit
                roof_logit_accum[0][nodata_map] = 100.0
                results["roof_type_mask"] = roof_logit_accum.argmax(axis=0).astype(
                    np.uint8
                )
            elif key in logit_accum:
                avg_logits = logit_accum[key] / weight_map
                prob = 1.0 / (1.0 + np.exp(-avg_logits))
                prob[nodata_map] = 0
                results[key] = prob

        return results

    def _predict_single_tile(
        self, tile: np.ndarray, return_logits: bool = False
    ) -> Dict[str, np.ndarray]:
        """Predict on a single tile (H, W, 3) uint8."""
        # Normalize
        img = tile.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        # To tensor
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        # Forward
        with torch.no_grad():
            outputs = self.model(tensor)

        # Convert to numpy
        result = {}
        # Tasks we expect results for
        all_tasks = self.BINARY_MASKS + ["roof_type_mask"]

        for key in all_tasks:
            # Handle naming inconsistency between head output keys and FEATURES keys
            mask_key = f"{key}_mask" if key in self.BINARY_MASKS else key
            if mask_key in outputs:
                logits = outputs[mask_key].squeeze().cpu().numpy()
                if return_logits:
                    result[key] = logits
                else:
                    if key == "roof_type_mask":
                        # Softmax for probabilities
                        exp_l = np.exp(logits - np.max(logits, axis=0))
                        result[key] = exp_l / np.sum(exp_l, axis=0)
                    else:
                        # Sigmoid for binary
                        result[key] = 1.0 / (1.0 + np.exp(-logits))
        return result

    def predict_image(
        self,
        image: np.ndarray,
        selected_masks: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on an in-memory image (H, W, 3) uint8.
        Uses tiling for large images, single-pass for small ones.
        """
        H, W = image.shape[:2]

        if H <= self.tile_size and W <= self.tile_size:
            preds = self._predict_single_tile(image)
            results = {}
            for key in selected_masks or self.BINARY_MASKS:
                if key in preds:
                    if key == "roof_type_mask":
                        results[key] = preds[key].argmax(axis=0).astype(np.uint8)
                    else:
                        results[key] = preds[key]
            return results

        # Tile large images
        if selected_masks is None:
            selected_masks = self.BINARY_MASKS + ["roof_type_mask"]

        accum = {k: np.zeros((H, W), dtype=np.float32) for k in selected_masks}
        weight_map = np.zeros((H, W), dtype=np.float32)
        roof_accum = None
        if "roof_type_mask" in selected_masks:
            roof_accum = np.zeros((5, H, W), dtype=np.float32)

        step = self.tile_size - self.overlap
        for y in range(0, H, step):
            for x in range(0, W, step):
                h = min(self.tile_size, H - y)
                w = min(self.tile_size, W - x)
                if h < self.tile_size // 4 or w < self.tile_size // 4:
                    continue

                tile = image[y : y + h, x : x + w]
                if h < self.tile_size or w < self.tile_size:
                    padded = np.zeros(
                        (self.tile_size, self.tile_size, 3), dtype=np.uint8
                    )
                    padded[:h, :w] = tile
                    tile = padded

                preds = self._predict_single_tile(tile)
                blend = self.blend_kernel[:h, :w]

                for key in selected_masks:
                    if key == "roof_type_mask" and roof_accum is not None:
                        roof_accum[:, y : y + h, x : x + w] += (
                            preds.get("roof_type_mask", np.zeros((5, h, w)))[:, :h, :w]
                            * blend[np.newaxis]
                        )
                    elif key in preds:
                        accum[key][y : y + h, x : x + w] += preds[key][:h, :w] * blend
                weight_map[y : y + h, x : x + w] += blend

        weight_map = np.maximum(weight_map, 1e-8)
        results = {}
        for key in selected_masks:
            if key == "roof_type_mask" and roof_accum is not None:
                for c in range(5):
                    roof_accum[c] /= weight_map
                results["roof_type_mask"] = roof_accum.argmax(axis=0).astype(np.uint8)
            elif key in accum:
                results[key] = accum[key] / weight_map

        return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def load_model_for_inference(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a trained model from checkpoint."""
    from models.model import SvamitvaModel

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Integrity check
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict) or len(state) == 0:
        raise ValueError("Invalid checkpoint: empty state dict")

    # Detect backbone from weights
    backbone = "sam2" if any("encoder.encoder" in k for k in state) else "resnet50"

    model = SvamitvaModel(backbone=backbone, pretrained=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    logger.info(f"Loaded model ({backbone}) from {ckpt_path}")
    return model
