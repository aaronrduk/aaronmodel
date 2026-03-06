"""
Training configuration dataclass for SVAMITVA SAM2 model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TrainingConfig:
    """All hyperparameters and paths for training."""

    # ── Paths ────────────────────────────────────────────────────────────────
    train_dirs: List[str] = field(
        default_factory=lambda: [
            "data/MAP1",
            "data/MAP2",
        ]
    )
    val_dir: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── Model ────────────────────────────────────────────────────────────────
    backbone: str = "sam2"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    pretrained: bool = True
    freeze_encoder: bool = False
    num_roof_classes: int = 5
    fpn_channels: int = 256
    dropout: float = 0.1

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw | sgd
    scheduler: str = "cosine"  # cosine | step | plateau
    lr_min: float = 1e-6
    warmup_epochs: int = 5
    gradient_clip: float = 0.5
    mixed_precision: bool = True
    freeze_backbone_epochs: int = 3
    seed: int = 42
    force_cpu: bool = False

    # ── Data ─────────────────────────────────────────────────────────────────
    tile_size: int = 512
    tile_overlap: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    val_split: float = 0.2

    # ── Loss Weights ─────────────────────────────────────────────────────────
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
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
    )

    # ── Checkpointing ────────────────────────────────────────────────────────
    save_top_k: int = 3
    metric_for_best: str = "avg_iou"
    early_stopping: bool = True
    patience: int = 25
    eval_every_n_epochs: int = 1

    # ── Logging ──────────────────────────────────────────────────────────────
    log_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "svamitva-sam2"
    experiment_name: str = "baseline"

    def __post_init__(self):
        self.train_dirs = [Path(d) for d in self.train_dirs]
        if self.val_dir:
            self.val_dir = Path(self.val_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def get_quick_test_config() -> TrainingConfig:
    """Small config for quick smoke-testing."""
    return TrainingConfig(
        batch_size=2,
        num_epochs=3,
        tile_size=256,
        num_workers=0,
        mixed_precision=False,
        early_stopping=False,
    )


def get_full_training_config() -> TrainingConfig:
    """Full production training config."""
    return TrainingConfig(
        batch_size=8,
        num_epochs=100,
        tile_size=512,
        mixed_precision=True,
        early_stopping=True,
        patience=25,
    )
