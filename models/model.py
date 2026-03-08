"""
Unified SVAMITVA segmentation model (SAM2 encoder + multi-head decoder).

This model predicts all raster outputs directly:
- building_mask, roof_type_mask
- road_mask, road_centerline_mask
- waterbody_mask, waterbody_line_mask, waterbody_point_mask
- utility_line_mask, utility_point_mask
- bridge_mask, railway_mask

YOLO-based point detections are integrated at inference-time in `inference/predict.py`
and fused with point masks.
"""

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam2_encoder import SAM2Encoder
from .decoder import FPNDecoder
from .heads import create_all_heads

logger = logging.getLogger(__name__)

DEFAULT_SAM2_CKPT_CANDIDATES = [
    Path("checkpoints/sam2.1_hiera_base_plus.pt"),
    Path("checkpoints/sam2_hiera_base_plus.pt"),
]


class EnsembleSvamitvaModel(nn.Module):
    """
    Unified Production Architecture for SVAMITVA Feature Extraction.

    Integrates:
    - SAM2 Hiera Backbone (Multi-scale features)
    - FPN Decoder with CBAM Attention
    - Specialized Task Heads (Building, Line, Detection)
    """

    def __init__(
        self,
        num_roof_classes: int = 5,
        pretrained: bool = True,
        checkpoint_path: str = "",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
        dropout: float = 0.1,
    ):
        super().__init__()

        resolved_ckpt = self._resolve_sam2_checkpoint(checkpoint_path, pretrained)

        # 1. Backbone (SAM2)
        self.encoder = SAM2Encoder(
            checkpoint_path=resolved_ckpt, model_cfg=model_cfg, freeze=False
        )

        # 2. Decoder (FPN + CBAM)
        self.decoder = FPNDecoder(
            in_channels=self.encoder.feature_channels, out_channels=256
        )

        # 3. Heads (11 tasks)
        self.heads = create_all_heads(
            in_channels=256, num_roof_classes=num_roof_classes, dropout=dropout
        )

    def _resolve_sam2_checkpoint(self, checkpoint_path: str, pretrained: bool) -> str:
        """Resolve which SAM2 checkpoint should initialize the encoder."""
        if checkpoint_path:
            p = Path(checkpoint_path)
            if p.exists():
                return str(p)
            logger.warning("Specified SAM2 checkpoint not found: %s", p)

        if pretrained:
            for cand in DEFAULT_SAM2_CKPT_CANDIDATES:
                if cand.exists():
                    logger.info("Using SAM2 checkpoint: %s", cand)
                    return str(cand)
            logger.warning(
                "No SAM2 checkpoint file found in default locations. "
                "Encoder will initialize without local checkpoint."
            )

        return ""

    def forward(self, x: torch.Tensor, task: str = "all") -> Dict[str, torch.Tensor]:
        """
        Forward pass through the unified pipeline.

        Args:
            x: Input image tensor (B, 3, H, W)
            task: Task filter (e.g., "buildings", "roads", or "all")
        """
        # Feature extraction
        backbone_feats = self.encoder(x)

        # Multi-scale fusion
        fused_feat = self.decoder(backbone_feats)
        target_size = x.shape[2:]

        outputs = {}

        task_norm = task.lower().strip()
        run_all = task_norm in {"all", "*", "full"}

        # Buildings + roof
        if run_all or task_norm in {
            "buildings",
            "building",
            "building_mask",
            "roof",
            "roof_type",
            "roof_type_mask",
        }:
            mask, roof = self.heads["building"](fused_feat)
            outputs["building_mask"] = F.interpolate(
                mask, size=target_size, mode="bilinear", align_corners=False
            )
            outputs["roof_type_mask"] = F.interpolate(
                roof, size=target_size, mode="bilinear", align_corners=False
            )

        # Roads
        if run_all or task_norm in {"roads", "road", "road_mask"}:
            outputs["road_mask"] = F.interpolate(
                self.heads["road"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {
            "roads",
            "road",
            "road_centerline",
            "road_centerline_mask",
        }:
            outputs["road_centerline_mask"] = F.interpolate(
                self.heads["road_centerline"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Water
        if run_all or task_norm in {"water", "waterbody", "waterbody_mask"}:
            outputs["waterbody_mask"] = F.interpolate(
                self.heads["waterbody"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {
            "water",
            "waterbody",
            "waterbody_line",
            "waterbody_line_mask",
        }:
            outputs["waterbody_line_mask"] = F.interpolate(
                self.heads["waterbody_line"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {
            "water",
            "waterbody",
            "waterbody_point",
            "waterbody_point_mask",
        }:
            outputs["waterbody_point_mask"] = F.interpolate(
                self.heads["waterbody_point"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Utilities
        if run_all or task_norm in {"utilities", "utility", "utility_line_mask"}:
            outputs["utility_line_mask"] = F.interpolate(
                self.heads["utility_line"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {
            "utilities",
            "utility",
            "utility_point",
            "utility_point_mask",
        }:
            outputs["utility_point_mask"] = F.interpolate(
                self.heads["utility_point"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Infrastructure
        if run_all or task_norm in {"railway", "railway_mask"}:
            outputs["railway_mask"] = F.interpolate(
                self.heads["railway"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if run_all or task_norm in {"bridge", "bridge_mask"}:
            outputs["bridge_mask"] = F.interpolate(
                self.heads["bridge"](fused_feat),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return outputs

    def freeze_backbone(self):
        """Freeze SAM2 encoder for head-only training."""
        self.encoder.freeze()

    def unfreeze_backbone(self):
        """Unfreeze SAM2 encoder for full fine-tuning."""
        self.encoder.unfreeze()

    def get_param_groups(self, base_lr: float = 1e-4) -> list:
        """Categorize parameters for LR scaling."""
        backbone_params = list(self.encoder.parameters())
        head_params = list(self.decoder.parameters()) + list(self.heads.parameters())

        return [
            {"params": head_params, "lr": base_lr},
            {"params": backbone_params, "lr": base_lr * 0.1},  # Slower backbone LR
        ]


SvamitvaModel = EnsembleSvamitvaModel
