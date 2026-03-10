"""
SAM2 Image Encoder wrapper for feature extraction.

Loads Meta's SAM2.1 Hiera B+ checkpoint and exposes the image encoder
as a multi-scale feature extractor producing 4 hierarchical feature maps.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Default SAM2.1 checkpoint URL
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2"
    "/092824/sam2.1_hiera_base_plus.pt"
)


def _repo_search_roots() -> List[Path]:
    roots: List[Path] = []
    search_paths = [Path.cwd()]
    try:
        search_paths.append(Path(__file__).resolve().parents[1])
    except NameError:
        pass

    for cand in search_paths:
        if cand.exists() and cand.is_dir() and cand not in roots:
            roots.append(cand)
    return roots


def _discover_local_sam2_roots() -> List[Path]:
    """Discover local SAM2 source trees in the workspace."""
    candidates: List[Path] = []
    for root in _repo_search_roots():
        try:
            children = [root] + [d for d in root.iterdir() if d.is_dir()]
        except Exception:
            children = [root]

        for base in children:
            if (base / "sam2" / "build_sam.py").exists() and base not in candidates:
                candidates.append(base)

            # One extra depth level is enough for typical `sam2-main` extracts.
            try:
                grand_children = [d for d in base.iterdir() if d.is_dir()]
            except Exception:
                grand_children = []
            for child in grand_children:
                if (child / "sam2" / "build_sam.py").exists():
                    if child not in candidates:
                        candidates.append(child)

    def _score(path: Path) -> int:
        name = path.name.lower()
        if "sam2-main" in name:
            return 0
        if "sam2" in name or "segment-anything-2" in name:
            return 1
        return 2

    return sorted(candidates, key=_score)


def _has_sam2_build() -> bool:
    try:
        return importlib.util.find_spec("sam2.build_sam") is not None
    except Exception:
        return False


def _ensure_sam2_importable() -> bool:
    if _has_sam2_build():
        return True

    for candidate in _discover_local_sam2_roots():
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))
        if _has_sam2_build():
            logger.info("Loaded SAM2 package from local source tree: %s", candidate)
            return True
    return False


def _resolve_model_cfg_path(model_cfg: str) -> str:
    """
    Resolve a SAM2 config reference.

    Accepts:
    - Existing file paths
    - Relative paths inside discovered local sam2 source trees
    - Config names installed with the sam2 package (for example sam2.1_hiera_b+.yaml)
    """
    cfg_path = Path(model_cfg)
    if cfg_path.exists():
        return str(cfg_path)

    for root in _discover_local_sam2_roots():
        local_cfg = root / model_cfg
        if local_cfg.exists():
            logger.info("Resolved SAM2 config from local source: %s", local_cfg)
            return str(local_cfg)

    return model_cfg


class SAM2Encoder(nn.Module):
    """
    Wraps the SAM2.1 Hiera image encoder for multi-scale feature extraction.

    Produces a dict of feature maps at 4 scales:
        feat_s4  -> stride 4  (H/4 x W/4)
        feat_s8  -> stride 8  (H/8 x W/8)
        feat_s16 -> stride 16 (H/16 x W/16)
        feat_s32 -> stride 32 (H/32 x W/32)

    Args:
        checkpoint_path: path to SAM2 .pt checkpoint
        model_cfg: SAM2 model config name (for build_sam2)
        freeze: whether to freeze encoder weights initially
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        model_cfg: str = DEFAULT_SAM2_MODEL_CFG,
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
            if not _ensure_sam2_importable():
                raise ImportError("sam2 package is unavailable")

            from sam2.build_sam import build_sam2

            resolved_model_cfg = _resolve_model_cfg_path(self.model_cfg)
            ckpt = self.checkpoint_path
            if ckpt and Path(ckpt).exists():
                logger.info("Loading SAM2 from checkpoint: %s", ckpt)
                sam2_model = build_sam2(
                    resolved_model_cfg,
                    ckpt,
                    device="cpu",
                )
            else:
                logger.info("Building SAM2 without checkpoint (random init)")
                sam2_model = build_sam2(
                    resolved_model_cfg,
                    ckpt_path=None,
                    device="cpu",
                )

            # Extract the image encoder
            encoder = sam2_model.image_encoder

            # Infer feature channels by running a dummy forward
            channels = self._infer_channels(encoder)

            return encoder, channels

        except ImportError:
            logger.warning(
                "SAM2 package not available (pip/local source not found). "
                "Using ResNet50 fallback backbone."
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
                return {key: val.shape[1] for key, val in feats.items()}
            except Exception as e:
                logger.warning("Channel inference failed: %s. Using defaults.", e)
                return {
                    "feat_s4": 256,
                    "feat_s8": 256,
                    "feat_s16": 256,
                    "feat_s32": 256,
                }

    def _extract_features(self, out) -> Dict[str, torch.Tensor]:
        """Parse SAM2 encoder output into named feature maps."""
        # Handle already-formatted dicts (e.g. from fallback)
        if isinstance(out, dict) and any(
            key.startswith("feat_s") for key in out.keys()
        ):
            return {key: val.contiguous() for key, val in out.items()}

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

        if isinstance(out, dict):
            result = {}
            for i, (_, val) in enumerate(out.items()):
                if isinstance(val, torch.Tensor):
                    result[f"feat_s{4*(2**i)}"] = val.contiguous()
            return result

        if isinstance(out, (list, tuple)):
            result = {}
            for i, feat in enumerate(out):
                if isinstance(feat, torch.Tensor):
                    result[f"feat_s{4*(2**i)}"] = feat.contiguous()
            return result

        return {"feat_s16": out.contiguous()}

    def _resnet_fallback(self) -> Tuple[nn.Module, Dict[str, int]]:
        """Fallback to ResNet50 if SAM2 is not available."""
        import torchvision.models as models

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        class ResNetEncoder(nn.Module):
            def __init__(self, resnet_model):
                super().__init__()
                self.conv1 = resnet_model.conv1
                self.bn1 = resnet_model.bn1
                self.relu = resnet_model.relu
                self.maxpool = resnet_model.maxpool
                self.layer1 = resnet_model.layer1
                self.layer2 = resnet_model.layer2
                self.layer3 = resnet_model.layer3
                self.layer4 = resnet_model.layer4

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
