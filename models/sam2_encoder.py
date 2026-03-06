"""
SAM2 Image Encoder wrapper for feature extraction.

Loads Meta's SAM2.1 Hiera B+ checkpoint and exposes the image encoder
as a multi-scale feature extractor producing 4 hierarchical feature maps.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default SAM2.1 checkpoint URL
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2"
    "/092824/sam2.1_hiera_base_plus.pt"
)


class SAM2Encoder(nn.Module):
    """
    Wraps the SAM2.1 Hiera image encoder for multi-scale feature extraction.

    Produces a dict of feature maps at 4 scales:
        feat_s4  → stride 4  (H/4 × W/4)
        feat_s8  → stride 8  (H/8 × W/8)
        feat_s16 → stride 16 (H/16 × W/16)
        feat_s32 → stride 32 (H/32 × W/32)

    Args:
        checkpoint_path: path to SAM2 .pt checkpoint
        model_cfg: SAM2 model config name (for build_sam2)
        freeze: whether to freeze encoder weights initially
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
        freeze: bool = False,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self._freeze = freeze

        # Build SAM2 model and extract image encoder
        self.encoder, self.feature_channels = self._build_encoder()

        if freeze:
            self.freeze()

    def _build_encoder(self) -> Tuple[nn.Module, Dict[str, int]]:
        """Load SAM2 and extract the image encoder trunk."""
        try:
            from sam2.build_sam import build_sam2

            ckpt = self.checkpoint_path
            if ckpt and Path(ckpt).exists():
                logger.info(f"Loading SAM2 from checkpoint: {ckpt}")
                sam2_model = build_sam2(
                    self.model_cfg,
                    ckpt,
                    device="cpu",
                )
            else:
                logger.info("Building SAM2 without checkpoint (random init)")
                sam2_model = build_sam2(
                    self.model_cfg,
                    ckpt_path=None,
                    device="cpu",
                )

            # Extract the image encoder
            encoder = sam2_model.image_encoder
            trunk = encoder.trunk

            # Infer feature channels by running a dummy forward
            channels = self._infer_channels(encoder)

            return encoder, channels

        except ImportError:
            logger.warning(
                "sam2 package not installed. Using ResNet50 fallback backbone."
            )
            return self._resnet_fallback()

    def _infer_channels(self, encoder: nn.Module) -> Dict[str, int]:
        """Run a dummy input to determine output channels."""
        encoder.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            try:
                out = encoder(dummy)
                feats = self._extract_features(out)
                return {k: v.shape[1] for k, v in feats.items()}
            except Exception as e:
                logger.warning(f"Channel inference failed: {e}. " "Using defaults.")
                return {
                    "feat_s4": 256,
                    "feat_s8": 256,
                    "feat_s16": 256,
                    "feat_s32": 256,
                }

    def _extract_features(self, out) -> Dict[str, torch.Tensor]:
        """Parse SAM2 encoder output into named feature maps."""
        if isinstance(out, dict) and "backbone_fpn" in out:
            # SAM2 returns:
            #   backbone_fpn: [stride4, stride8, stride16]
            #   vision_features: stride16 (same as last)
            fpn = out["backbone_fpn"]
            result = {}
            # fpn[0]=stride4, fpn[1]=stride8, fpn[2]=stride16
            strides = [4, 8, 16]
            for i, feat in enumerate(fpn):
                result[f"feat_s{strides[i]}"] = feat.contiguous()
            # stride 32 via downsampling stride 16
            s16 = fpn[-1].contiguous()
            s32 = F.avg_pool2d(s16, 2, stride=2)
            result["feat_s32"] = s32
            return result
        elif isinstance(out, dict):
            result = {}
            for i, (k, v) in enumerate(out.items()):
                if isinstance(v, torch.Tensor):
                    result[f"feat_s{4*(2**i)}"] = v.contiguous()
            return result
        elif isinstance(out, (list, tuple)):
            result = {}
            for i, feat in enumerate(out):
                if isinstance(feat, torch.Tensor):
                    result[f"feat_s{4*(2**i)}"] = feat.contiguous()
            return result
        else:
            return {"feat_s16": out.contiguous()}

    def _resnet_fallback(self) -> Tuple[nn.Module, Dict[str, int]]:
        """Fallback to ResNet50 if SAM2 is not available."""
        import torchvision.models as models

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        class ResNetEncoder(nn.Module):
            def __init__(self, resnet):
                super().__init__()
                self.conv1 = resnet.conv1
                self.bn1 = resnet.bn1
                self.relu = resnet.relu
                self.maxpool = resnet.maxpool
                self.layer1 = resnet.layer1
                self.layer2 = resnet.layer2
                self.layer3 = resnet.layer3
                self.layer4 = resnet.layer4

            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                c1 = self.layer1(x)  # stride 4
                c2 = self.layer2(c1)  # stride 8
                c3 = self.layer3(c2)  # stride 16
                c4 = self.layer4(c3)  # stride 32
                return {"feat_s4": c1, "feat_s8": c2, "feat_s16": c3, "feat_s32": c4}

        encoder = ResNetEncoder(resnet)
        channels = {
            "feat_s4": 256,
            "feat_s8": 512,
            "feat_s16": 1024,
            "feat_s32": 2048,
        }
        return encoder, channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            Dict with feat_s4, feat_s8, feat_s16, feat_s32
        """
        out = self.encoder(x)
        return self._extract_features(out)

    def freeze(self):
        """Freeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info("SAM2 encoder frozen")

    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        logger.info("SAM2 encoder unfrozen")
