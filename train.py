#!/usr/bin/env python3
"""
CLI entry point for training the SVAMITVA Ensemble SOTA Architecture (V3).

Usage:
    python train.py --train_dirs /path/to/MAP1 --epochs 50 --lr 3e-4
    python train.py --quick_test
"""

import argparse
import logging
import re
from pathlib import Path

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train SVAMITVA Ensemble Feature Extraction Model (V3)"
    )

    # Data
    p.add_argument(
        "--train_dirs",
        nargs="+",
        default=["data/MAP1"],
        help="Directories containing MAP*.tif + shapefiles",
    )
    p.add_argument("--val_dir", default=None, help="Separate validation directory")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--freeze_epochs", type=int, default=5, help="Epochs to freeze heads"
    )
    p.add_argument(
        "--sam2_checkpoint",
        default="checkpoints/sam2.1_hiera_base_plus.pt",
        help="Path to SAM2 backbone checkpoint",
    )
    p.add_argument(
        "--sam2_model_cfg",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM2 model config path/name",
    )

    # Misc
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint .pt")
    p.add_argument("--name", default="ensemble_v3", help="Experiment name")
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument("--quick_test", action="store_true", help="3-epoch smoke test")

    return p.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    for d in args.train_dirs:
        if not Path(d).is_dir():
            logger.error(f"Training directory not found: {d}")
            return

    args.name = re.sub(r"[^a-zA-Z0-9_\-]", "_", args.name)

    # Imports
    from data.dataset import create_dataloaders
    from models.losses import MultiTaskLoss
    from models.model import EnsembleSvamitvaModel
    from training.config import TrainingConfig, get_quick_test_config
    from training.trainer import Trainer

    # Configuration
    if args.quick_test:
        config = get_quick_test_config()
        config.train_dirs = [Path(d) for d in args.train_dirs]
        logger.info("⚡ Quick-test mode enabled")
    else:
        config = TrainingConfig(
            train_dirs=args.train_dirs,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            tile_size=args.tile_size,
            num_workers=args.num_workers,
            seed=args.seed,
            freeze_backbone_epochs=args.freeze_epochs,
            checkpoint_dir=args.checkpoint_dir,
            force_cpu=args.force_cpu,
            experiment_name=args.name,
            dropout=args.dropout,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_cfg=args.sam2_model_cfg,
        )

    # Data
    logger.info("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_dirs=config.train_dirs,
        val_dir=config.val_dir,
        image_size=config.tile_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
    )

    # Model
    logger.info("Building Ensemble model...")
    model = EnsembleSvamitvaModel(
        num_roof_classes=config.num_roof_classes,
        pretrained=config.pretrained,
        checkpoint_path=(
            str(config.sam2_checkpoint) if config.sam2_checkpoint is not None else ""
        ),
        model_cfg=config.sam2_model_cfg,
        dropout=config.dropout,
    )

    # Resume
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(
                checkpoint.get("model_state_dict", checkpoint), strict=True
            )
            # Recover epoch
            start_epoch = checkpoint.get("epoch", 0) + 1
        else:
            logger.warning(f"Checkpoint not found: {resume_path}")

    # Loss
    loss_fn = MultiTaskLoss()

    # Train
    trainer = Trainer(
        model, train_loader, val_loader, loss_fn, config, start_epoch=start_epoch
    )

    # Resume optimizer/scheduler if available
    if args.resume and "checkpoint" in locals():
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("  → Optimizer state restored")
        if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("  → Scheduler state restored")

    trainer.fit()

    if getattr(trainer, "was_interrupted", False):
        logger.warning(
            "Training stopped before completion. Latest checkpoint was saved for resume."
        )
    else:
        logger.info("Project Finalization (V3) — Training Success! 🎉")


if __name__ == "__main__":
    main()
