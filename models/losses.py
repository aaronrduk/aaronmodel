"""
Multi-task loss functions for SVAMITVA feature extraction.

Combines:
  - Binary tasks: BCE + Dice + Lovász hinge
  - Roof-type task: Cross-entropy + Dice
  - Per-task weighting for class imbalance
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dice Loss ─────────────────────────────────────────────────────────────────


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) binary targets
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.float().view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


# ── Focal Loss ────────────────────────────────────────────────────────────────


class BinaryFocalLoss(nn.Module):
    """Focal loss for handling class imbalance in binary tasks."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ── Lovász Hinge Loss ────────────────────────────────────────────────────────


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-7)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszHingeLoss(nn.Module):
    """Lovász-Softmax loss for binary segmentation (IoU surrogate)."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        signs = 2.0 * targets - 1.0
        errors = 1.0 - logits * signs
        errors = torch.clamp(errors, min=0)  # stability
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = targets[perm]
        grad = _lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


# ── Boundary Loss ────────────────────────────────────────────────────────────


class BoundaryLoss(nn.Module):
    """Weighted BCE along mask boundaries for sharp edges."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Extract boundary via morphological gradient
        pad = self.kernel_size // 2
        t = targets.float()
        dilated = F.max_pool2d(t, self.kernel_size, stride=1, padding=pad)
        eroded = -F.max_pool2d(-t, self.kernel_size, stride=1, padding=pad)
        boundary = (dilated - eroded).clamp(0, 1)

        # Compute boundary-weighted BCE
        if boundary.sum() < 1:
            return torch.tensor(0.0, device=logits.device)

        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        return (bce * boundary).sum() / boundary.sum()


# ── Multi-class Dice Loss ────────────────────────────────────────────────────


class MultiClassDiceLoss(nn.Module):
    """Dice loss for multi-class segmentation (roof types)."""

    def __init__(self, num_classes: int = 5, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) class indices
        """
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.long(), self.num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ── Combined Multi-Task Loss ─────────────────────────────────────────────────


# Default task weights
DEFAULT_WEIGHTS = {
    "building": 1.0,
    "roof_type": 0.5,
    "road": 1.0,
    "road_centerline": 1.2,
    "waterbody": 1.2,
    "waterbody_line": 1.2,
    "waterbody_point": 1.5,
    "utility_line": 1.2,
    "utility_point": 1.3,
    "bridge": 1.5,
    "railway": 1.3,
}

BINARY_TASKS = [
    "building",
    "road",
    "road_centerline",
    "waterbody",
    "waterbody_line",
    "waterbody_point",
    "utility_line",
    "utility_point",
    "bridge",
    "railway",
]


class MultiTaskLoss(nn.Module):
    """
    Combined loss for all 11 tasks.

    Binary tasks: BCE + Dice + Lovász (equal-weighted sum)
    Roof-type task: CE + MultiClassDice

    Each task loss is multiplied by its weight before summation.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        num_roof_classes: int = 5,
    ):
        super().__init__()
        self.weights = weights or DEFAULT_WEIGHTS

        # Binary loss components
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        self.lovasz = LovaszHingeLoss()

        # Multi-class loss for roof types
        self.ce = nn.CrossEntropyLoss(ignore_index=0)  # ignore background
        self.mc_dice = MultiClassDiceLoss(num_roof_classes)

        self.boundary = BoundaryLoss()

    def _binary_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combined binary loss with boundary emphasis."""
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # Base losses
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        focal = self.focal(logits, targets)

        # Edge-sharpening losses
        lovasz = self.lovasz(logits, targets)
        bdry = self.boundary(logits, targets)

        return bce + dice + 0.5 * focal + 1.0 * lovasz + 0.5 * bdry

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total multi-task loss.

        Args:
            predictions: dict of model outputs per task
            targets: dict of ground truth masks per task

        Returns:
            total_loss: scalar
            breakdown: dict of per-task loss values (for logging)
        """
        device = next(iter(predictions.values())).device
        total = torch.zeros(1, device=device)
        breakdown = {}

        for task in BINARY_TASKS:
            pred_key = f"{task}_mask"
            if pred_key not in predictions or pred_key not in targets:
                continue
            loss = self._binary_loss(predictions[pred_key], targets[pred_key])
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            weighted = loss * self.weights.get(task, 1.0)
            total = total + weighted
            breakdown[task] = loss.item()

        # Roof type loss
        if "roof_type_mask" in predictions and "roof_type_mask" in targets:
            rt_pred = predictions["roof_type_mask"]
            rt_target = targets["roof_type_mask"]
            if rt_target.ndim == 4:
                rt_target = rt_target.squeeze(1)
            # Skip if all pixels are background (class 0)
            # CE with ignore_index=0 returns NaN when
            # there are no non-ignored pixels
            if (rt_target.long() != 0).any():
                rt_loss = self.ce(rt_pred, rt_target.long()) + self.mc_dice(
                    rt_pred, rt_target
                )
                if not (torch.isnan(rt_loss) or torch.isinf(rt_loss)):
                    w = self.weights.get("roof_type", 0.5)
                    total = total + rt_loss * w
                    breakdown["roof_type"] = rt_loss.item()

        return total, breakdown
