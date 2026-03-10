#!/usr/bin/env python3
"""
Map-wise K-fold training for small SVAMITVA datasets (for example 5 villages).

This avoids tile leakage between train/validation by splitting at map-level.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import create_kfold_dataloaders
from models.losses import MultiTaskLoss
from models.model import EnsembleSvamitvaModel
from train_engine.config import TrainingConfig
from train_engine.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-22s │ %(levelname)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_kfold")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map-wise K-fold training for SVAMITVA.")
    p.add_argument(
        "--train_dirs",
        nargs="+",
        default=["data/MAP1"],
        help="MAP directories or parent directories containing MAP folders.",
    )
    p.add_argument("--n_splits", type=int, default=5, help="Number of map-level folds.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--tile_overlap", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freeze_epochs", type=int, default=5)
    p.add_argument("--sam2_checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument(
        "--sam2_model_cfg",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
    )
    p.add_argument("--checkpoint_root", default="checkpoints/kfold")
    p.add_argument("--name", default="svamitva_kfold")
    p.add_argument("--force_cpu", action="store_true")
    return p.parse_args()


def _validate_dirs(train_dirs: List[str]) -> List[Path]:
    resolved: List[Path] = []
    for d in train_dirs:
        p = Path(d)
        if not p.is_dir():
            raise FileNotFoundError(f"Training directory not found: {p}")
        resolved.append(p)
    return resolved


def main() -> None:
    args = parse_args()
    train_dirs = _validate_dirs(args.train_dirs)
    exp_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", args.name)

    fold_loaders = create_kfold_dataloaders(
        train_dirs=train_dirs,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        seed=args.seed,
    )
    if not fold_loaders:
        raise RuntimeError("No folds were created. Check input dataset.")

    all_scores: List[float] = []
    all_results: List[Dict[str, object]] = []
    ckpt_root = Path(args.checkpoint_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_loader, val_loader, val_maps) in enumerate(
        fold_loaders, start=1
    ):
        fold_tag = f"fold_{fold_idx:02d}"
        fold_ckpt = ckpt_root / fold_tag
        fold_logs = fold_ckpt / "logs"
        fold_ckpt.mkdir(parents=True, exist_ok=True)
        fold_logs.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting %s | val maps: %s | train batches: %d | val batches: %d",
            fold_tag,
            ", ".join(val_maps),
            len(train_loader),
            len(val_loader),
        )

        config = TrainingConfig(
            train_dirs=train_dirs,
            val_dir=None,
            checkpoint_dir=fold_ckpt,
            log_dir=fold_logs,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            split_mode="map",
            num_workers=args.num_workers,
            seed=args.seed + fold_idx,
            dropout=args.dropout,
            freeze_backbone_epochs=args.freeze_epochs,
            force_cpu=args.force_cpu,
            experiment_name=f"{exp_name}_{fold_tag}",
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_cfg=args.sam2_model_cfg,
        )

        model = EnsembleSvamitvaModel(
            num_roof_classes=config.num_roof_classes,
            pretrained=config.pretrained,
            checkpoint_path=str(config.sam2_checkpoint or ""),
            model_cfg=config.sam2_model_cfg,
            dropout=config.dropout,
        )
        loss_fn = MultiTaskLoss(
            weights=config.loss_weights,
            num_roof_classes=config.num_roof_classes,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            config=config,
            start_epoch=1,
        )
        trainer.fit()

        score = float(trainer.ckpt_mgr.best_score)
        all_scores.append(score)
        all_results.append(
            {
                "fold": fold_tag,
                "val_maps": val_maps,
                "best_score": score,
                "best_epoch": int(trainer.ckpt_mgr.best_epoch),
                "checkpoint_dir": str(fold_ckpt),
            }
        )
        logger.info("%s done | best_%s=%.5f", fold_tag, config.metric_for_best, score)

    summary = {
        "experiment": exp_name,
        "n_folds": len(all_results),
        "metric": "avg_iou",
        "folds": all_results,
        "cv_mean": float(mean(all_scores)) if all_scores else 0.0,
        "cv_std": float(pstdev(all_scores)) if len(all_scores) > 1 else 0.0,
    }
    out_path = ckpt_root / "kfold_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "K-fold training complete | mean=%.5f std=%.5f | summary=%s",
        summary["cv_mean"],
        summary["cv_std"],
        out_path,
    )


if __name__ == "__main__":
    main()
