#!/usr/bin/env python3
"""
DGX-Optimized Training entry point for SVAMITVA Ensemble (V3).
Automatically selects the GPU with most free memory and saves to check/best.pt.
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dgx_train")


def parse_args():
    p = argparse.ArgumentParser(
        description="DGX Training: SVAMITVA Ensemble Feature Extraction (V3)"
    )

    # Data
    p.add_argument(
        "--train_dirs",
        nargs="+",
        default=["data/MAP1"],
        help="Directories containing MAP*.tif + shapefiles",
    )
    p.add_argument("--val_dir", default=None, help="Separate validation directory")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--split_mode", default="map", choices=["map", "tile"])

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--tile_overlap", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=8)  # Higher for DGX
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freeze_epochs", type=int, default=5)
    p.add_argument(
        "--sam2_checkpoint",
        default="checkpoints/sam2.1_hiera_base_plus.pt",
        help="Path to SAM2 backbone checkpoint",
    )

    # DGX Specifics
    p.add_argument("--checkpoint_dir", default="check", help="Requested 'check' dir")
    p.add_argument("--name", default="dgx_ensemble_v3", help="Experiment name")
    p.add_argument("--quick_test", action="store_true", help="3-epoch smoke test")

    return p.parse_args()


def main():
    args = parse_args()

    # Ensure check directory exists
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Imports (deferred for faster CLI response)
    from data.dataset import create_dataloaders
    from models.losses import MultiTaskLoss
    from models.model import EnsembleSvamitvaModel
    from training.config import TrainingConfig, get_quick_test_config
    from training.trainer import Trainer

    # Configuration
    if args.quick_test:
        config = get_quick_test_config()
        config.train_dirs = [Path(d) for d in args.train_dirs]
        config.checkpoint_dir = Path(args.checkpoint_dir)
        logger.info("⚡ Quick-test mode enabled")
    else:
        config = TrainingConfig(
            train_dirs=[Path(d) for d in args.train_dirs],
            val_dir=Path(args.val_dir) if args.val_dir else None,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            split_mode=args.split_mode,
            val_split=args.val_split,
            num_workers=args.num_workers,
            freeze_backbone_epochs=args.freeze_epochs,
            checkpoint_dir=Path(args.checkpoint_dir),
            experiment_name=args.name,
            dropout=args.dropout,
            sam2_checkpoint=Path(args.sam2_checkpoint),
            mixed_precision=True,
        )

    # Data (Preprocessing happens here: tiling, normalization, etc.)
    logger.info(f"Loading datasets with tile_size={config.tile_size}...")
    train_loader, val_loader = create_dataloaders(
        train_dirs=config.train_dirs,
        val_dir=config.val_dir,
        image_size=config.tile_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        split_mode=config.split_mode,
    )

    # Model
    logger.info("Building model...")
    model = EnsembleSvamitvaModel(
        checkpoint_path=str(config.sam2_checkpoint),
        dropout=config.dropout,
    )

    # Loss
    loss_fn = MultiTaskLoss(
        num_roof_classes=config.num_roof_classes,
    )

    # Train
    # Note: Trainer now automatically selects the best GPU via updated get_device()
    trainer = Trainer(model, train_loader, val_loader, loss_fn, config)

    logger.info("Starting training on optimized GPU...")
    trainer.fit()

    logger.info(f"🎉 Training finished. Checkpoints in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
