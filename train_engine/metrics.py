"""
Per-task evaluation metrics: IoU, Dice/F1, pixel accuracy.
"""

from typing import Dict, Optional

import numpy as np
import torch


class TaskMetrics:
    """
    Accumulates predictions and targets for a single binary task
    and computes IoU, Dice, and pixel accuracy.
    """

    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Update with a batch of predictions.

        Args:
            logits: (B, 1, H, W) raw logits
            targets: (B, 1, H, W) or (B, H, W) binary targets
            mask: (B, H, W) optional valid pixel mask
        """
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > self.threshold).float()
            if preds.ndim == 4:
                preds = preds.squeeze(1)
            if targets.ndim == 4:
                targets = targets.squeeze(1)
            targets = targets.float()

            if mask is not None:
                if mask.ndim == 4:
                    mask = mask.squeeze(1)
                mask = mask > 0.5
                # Apply mask to all components
                t_tp = (preds == 1) & (targets == 1) & mask
                t_fp = (preds == 1) & (targets == 0) & mask
                t_fn = (preds == 0) & (targets == 1) & mask
                t_tn = (preds == 0) & (targets == 0) & mask

                self.tp += t_tp.sum().item()
                self.fp += t_fp.sum().item()
                self.fn += t_fn.sum().item()
                self.tn += t_tn.sum().item()
            else:
                self.tp += ((preds == 1) & (targets == 1)).sum().item()
                self.fp += ((preds == 1) & (targets == 0)).sum().item()
                self.fn += ((preds == 0) & (targets == 1)).sum().item()
                self.tn += ((preds == 0) & (targets == 0)).sum().item()

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / (denom + 1e-8)

    @property
    def dice(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp) / (denom + 1e-8)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / (denom + 1e-8)

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / (denom + 1e-8)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / (total + 1e-8)

    def compute(self) -> Dict[str, float]:
        return {
            f"{self.name}_iou": self.iou,
            f"{self.name}_dice": self.dice,
            f"{self.name}_precision": self.precision,
            f"{self.name}_recall": self.recall,
            f"{self.name}_accuracy": self.accuracy,
        }


class RoofTypeMetrics:
    """Multi-class accuracy for roof type classification."""

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.per_class_correct = np.zeros(self.num_classes)
        self.per_class_total = np.zeros(self.num_classes)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) class indices
        """
        with torch.no_grad():
            preds = logits.argmax(dim=1)  # (B, H, W)
            if targets.ndim == 4:
                targets = targets.squeeze(1)
            # Only evaluate where target is not background (0)
            mask = targets > 0
            if mask.sum() > 0:
                self.correct += (preds[mask] == targets[mask]).sum().item()
                self.total += mask.sum().item()
                for c in range(self.num_classes):
                    c_mask = targets == c
                    if c_mask.sum() > 0:
                        self.per_class_correct[c] += (
                            (preds[c_mask] == c).sum().item()
                        )
                        self.per_class_total[c] += c_mask.sum().item()

    @property
    def accuracy(self) -> float:
        return self.correct / (self.total + 1e-8)

    def compute(self) -> Dict[str, float]:
        result = {"roof_type_accuracy": float(self.accuracy)}
        classes = ["Background", "RCC", "Tiled", "Tin", "Others"]
        for c in range(self.num_classes):
            acc = self.per_class_correct[c] / (self.per_class_total[c] + 1e-8)
            result[f"roof_{classes[c]}_acc"] = float(acc)
        return result


class MetricsTracker:
    """Tracks all task metrics across an epoch."""

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

    def __init__(self, threshold: float = 0.5, num_roof_classes: int = 5):
        self.binary_metrics = {
            task: TaskMetrics(task, threshold) for task in self.BINARY_TASKS
        }
        self.roof_metrics = RoofTypeMetrics(num_roof_classes)

    def reset(self):
        for m in self.binary_metrics.values():
            m.reset()
        self.roof_metrics.reset()

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        valid_mask = targets.get("valid_mask", None)

        for task in self.BINARY_TASKS:
            key = f"{task}_mask"
            if key in predictions and key in targets:
                self.binary_metrics[task].update(
                    predictions[key], targets[key], mask=valid_mask
                )

        if "roof_type_mask" in predictions and "roof_type_mask" in targets:
            # RoofTypeMetrics already has an internal mask check for targets > 0
            # which is essentially "onlyPixelsUnderShapefile".
            # We add valid_mask to ensure NoData is also excluded.
            # However, since RoofType target is 0 for background anyway,
            # Targets > 0 already excludes background and NoData.
            self.roof_metrics.update(
                predictions["roof_type_mask"], targets["roof_type_mask"]
            )

    def compute(self) -> Dict[str, float]:
        result = {}
        ious = []
        dices = []

        for task, m in self.binary_metrics.items():
            metrics = m.compute()
            result.update(metrics)
            ious.append(m.iou)
            dices.append(m.dice)

        result.update(self.roof_metrics.compute())

        # Aggregates
        result["avg_iou"] = float(np.mean(ious)) if ious else 0.0
        result["avg_dice"] = float(np.mean(dices)) if dices else 0.0

        return result
