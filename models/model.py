"""
Complete SVAMITVA multi-task segmentation model.

Architecture:
    Image → SAM2 Encoder → FPN Decoder → Task-Group Refinement → 11 Task Heads

Outputs dict of per-pixel predictions for all geographic features.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam2_encoder import SAM2Encoder
from .decoder import FPNDecoder, TaskGroupRefinement
from .heads import create_all_heads

logger = logging.getLogger(__name__)


# Task groups: related tasks share a refinement block
TASK_GROUPS = {
    "building_group": ["building"],
    "road_group": ["road", "road_centerline"],
    "water_group": ["waterbody", "waterbody_line", "waterbody_point"],
    "utility_group": ["utility_line", "utility_point"],
    "infra_group": ["bridge", "railway"],
}


class SvamitvaModel(nn.Module):
    """
    End-to-end multi-task model for SVAMITVA drone feature extraction.

    Args:
        backbone: 'sam2' or 'resnet50'
        sam2_checkpoint: path to SAM2 .pt file
        sam2_model_cfg: SAM2 config name
        pretrained: use pretrained weights
        freeze_encoder: freeze backbone initially
        num_roof_classes: number of roof type classes (incl. background)
        fpn_channels: channel dimension for FPN and heads
        dropout: dropout rate for heads
    """

    def __init__(
        self,
        backbone: str = "sam2",
        sam2_checkpoint: str = "",
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        num_roof_classes: int = 5,
        fpn_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.fpn_channels = fpn_channels

        # 1. Backbone encoder
        self.encoder = SAM2Encoder(
            checkpoint_path=sam2_checkpoint if backbone == "sam2" else "",
            model_cfg=sam2_model_cfg,
            freeze=freeze_encoder,
        )
        feat_channels = self.encoder.feature_channels
        logger.info(f"Encoder feature channels: {feat_channels}")

        # 2. FPN decoder
        self.decoder = FPNDecoder(feat_channels, fpn_channels)

        # 3. Task-group refinement blocks
        self.task_refinement = nn.ModuleDict()
        for group_name in TASK_GROUPS:
            self.task_refinement[group_name] = TaskGroupRefinement(fpn_channels)

        # 4. Task heads
        self.heads = create_all_heads(fpn_channels, num_roof_classes, dropout)

        # Initialize head weights
        self._init_heads()

        # Log parameter count
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Model params: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable"
        )

    def _init_heads(self):
        """Kaiming initialization for head layers."""
        for module in [self.heads, self.task_refinement]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: (B, 3, H, W) normalized input image

        Returns:
            Dict with keys:
                building_mask, roof_type_mask, road_mask, road_centerline_mask,
                waterbody_mask, waterbody_line_mask, waterbody_point_mask,
                utility_line_mask, utility_point_mask, bridge_mask, railway_mask
            Each value is (B, C, H, W) logits at input resolution.
        """
        input_size = x.shape[2:]

        # Backbone
        features = self.encoder(x)

        # FPN decode
        fpn_out = self.decoder(features)  # (B, fpn_channels, H/4, W/4)

        # Per-group refinement + heads
        outputs = {}

        for group_name, task_keys in TASK_GROUPS.items():
            refined = self.task_refinement[group_name](fpn_out)

            for task_key in task_keys:
                if task_key not in self.heads:
                    continue

                head = self.heads[task_key]

                if task_key == "building":
                    mask_logits, roof_logits = head(refined)
                    # Upsample to input resolution
                    outputs["building_mask"] = F.interpolate(
                        mask_logits,
                        size=input_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    outputs["roof_type_mask"] = F.interpolate(
                        roof_logits,
                        size=input_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    logits = head(refined)
                    outputs[f"{task_key}_mask"] = F.interpolate(
                        logits, size=input_size, mode="bilinear", align_corners=False
                    )

        return outputs

    def freeze_backbone(self):
        """Freeze encoder weights for warm-start head training."""
        self.encoder.freeze()

    def unfreeze_backbone(self):
        """Unfreeze encoder for full fine-tuning."""
        self.encoder.unfreeze()

    def get_param_groups(self, base_lr: float = 3e-4) -> list:
        """
        Get parameter groups with differential learning rates.
        Backbone gets 0.1× the base LR.
        """
        encoder_params = list(self.encoder.parameters())
        encoder_ids = set(id(p) for p in encoder_params)
        other_params = [p for p in self.parameters() if id(p) not in encoder_ids]

        return [
            {"params": encoder_params, "lr": base_lr * 0.1},
            {"params": other_params, "lr": base_lr},
        ]
