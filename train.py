#!/usr/bin/env python3
"""
CLI entry point for training the SVAMITVA SAM2 feature extraction model.

Usage:
    # Full training with MAP1 and MAP2
    python train.py --train_dirs /path/to/MAP1 /path/to/MAP2

    # Quick smoke-test (3 epochs, 256px tiles)
    python train.py --quick_test

    # Custom configuration
    python train.py --train_dirs /data/MAP1 /data/MAP2 --epochs 50 --lr 0.0001 --batch_size 4

    # Resume from checkpoint
    python train.py --resume checkpoints/best.pt
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


# Default SAM2 checkpoint URL
SAM2_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2"
    "/092824/sam2.1_hiera_base_plus.pt"
)


def download_sam2_checkpoint(dest: Path) -> Path:
    """Download SAM2 checkpoint if not present."""
    if dest.exists():
        logger.info(f"SAM2 checkpoint already exists: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading SAM2 checkpoint to {dest}...")
    try:
        import urllib.request

        urllib.request.urlretrieve(SAM2_URL, str(dest))
        logger.info(f"Download complete: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    except Exception as e:
        logger.error(f"Failed to download SAM2 checkpoint: {e}")
        logger.info("Training will proceed without pretrained weights")
    return dest


def parse_args():
    p = argparse.ArgumentParser(
        description="Train SVAMITVA SAM2 Feature Extraction Model"
    )

    # Data
    p.add_argument(
        "--train_dirs",
        nargs="+",
        default=["data/MAP1", "data/MAP2"],
        help="Directories containing MAP*.tif + shapefiles",
    )
    p.add_argument("--val_dir", default=None, help="Separate validation directory")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--backbone", choices=["sam2", "resnet50"], default="sam2")
    p.add_argument(
        "--sam2_checkpoint",
        default="checkpoints/sam2.1_hiera_base_plus.pt",
    )
    p.add_argument("--freeze_epochs", type=int, default=3)
    p.add_argument("--fpn_channels", type=int, default=256)

    # Misc
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")
    p.add_argument("--name", default="baseline", help="Experiment name")
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument("--quick_test", action="store_true", help="3-epoch smoke test")
    p.add_argument("--wandb", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    for d in args.train_dirs:
        if not Path(d).is_dir():
            logger.error(f"Training directory not found: {d}")
            return
    # Sanitize experiment name (alphanumeric, hyphens, underscores only)
    args.name = re.sub(r"[^a-zA-Z0-9_\-]", "_", args.name)

    # Import here to avoid import-time errors
    from training.config import TrainingConfig, get_quick_test_config
    from training.trainer import Trainer
    from models.model import SvamitvaModel
    from models.losses import MultiTaskLoss
    from data.dataset import create_dataloaders

    # Configuration
    if args.quick_test:
        config = get_quick_test_config()
        config.train_dirs = [Path(d) for d in args.train_dirs]
        logger.info("⚡ Quick-test mode: 3 epochs, 256px tiles, small batch")
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
            backbone=args.backbone,
            sam2_checkpoint=args.sam2_checkpoint,
            freeze_encoder=False,
            freeze_backbone_epochs=args.freeze_epochs,
            fpn_channels=args.fpn_channels,
            checkpoint_dir=args.checkpoint_dir,
            force_cpu=args.force_cpu,
            experiment_name=args.name,
            use_wandb=args.wandb,
        )

    # Ensure SAM2 checkpoint
    if config.backbone == "sam2":
        ckpt_path = Path(config.sam2_checkpoint)
        download_sam2_checkpoint(ckpt_path)

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
    logger.info(
        f"Data loaded: {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches"
    )

    # Model
    logger.info("Building model...")
    model = SvamitvaModel(
        backbone=config.backbone,
        sam2_checkpoint=str(config.sam2_checkpoint),
        sam2_model_cfg=config.sam2_model_cfg,
        pretrained=config.pretrained,
        freeze_encoder=False,  # Trainer handles staged unfreezing
        num_roof_classes=config.num_roof_classes,
        fpn_channels=config.fpn_channels,
        dropout=config.dropout,
    )

    # Resume from checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning(f"Checkpoint not found: {resume_path}")

    # Loss
    loss_fn = MultiTaskLoss(weights=config.loss_weights)

    # Train
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config)
    trainer.fit()

    logger.info("🎉 Training complete!")


if __name__ == "__main__":
    main()
