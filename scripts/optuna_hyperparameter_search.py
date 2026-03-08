#!/usr/bin/env python3
"""
Optuna hyperparameter search for the SVAMITVA training pipeline.

This script is aligned with the current project APIs:
- data.dataset.create_dataloaders
- models.model.EnsembleSvamitvaModel
- models.losses.MultiTaskLoss
- training.trainer.Trainer
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import optuna

from data.dataset import create_dataloaders
from models.losses import MultiTaskLoss
from models.model import EnsembleSvamitvaModel
from training.config import TrainingConfig
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("optuna_search")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for SVAMITVA training."
    )
    parser.add_argument(
        "--train_dirs",
        nargs="+",
        default=["data/MAP1"],
        help="One or more training map directories.",
    )
    parser.add_argument(
        "--val_dir",
        default=None,
        help="Optional separate validation directory (if omitted, auto-split is used).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("logs/optuna"),
        help="Directory to save Optuna artifacts.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs per trial.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tile size used by dataset/dataloaders.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for trials.",
    )
    parser.add_argument(
        "--study_name",
        default="svamitva_optuna",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL (for example sqlite:///logs/optuna/study.db).",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU training during search.",
    )
    return parser.parse_args()


def _suggest_hparams(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Sample trial hyperparameters using modern Optuna APIs."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "roof_type_weight": trial.suggest_float("roof_type_weight", 0.2, 1.2),
        "patience": trial.suggest_int("patience", 3, 12),
    }


def _build_config(
    args: argparse.Namespace, trial: optuna.trial.Trial, hparams: Dict[str, Any]
) -> TrainingConfig:
    train_dirs = [Path(d) for d in args.train_dirs]
    val_dir: Optional[Path] = Path(args.val_dir) if args.val_dir else None
    trial_dir = args.output_dir / f"trial_{trial.number:04d}"
    trial_log_dir = trial_dir / "logs"
    trial_ckpt_dir = trial_dir / "checkpoints"

    base_weights = TrainingConfig().loss_weights
    tuned_weights = {**base_weights, "roof_type": float(hparams["roof_type_weight"])}

    return TrainingConfig(
        train_dirs=train_dirs,
        val_dir=val_dir,
        checkpoint_dir=trial_ckpt_dir,
        log_dir=trial_log_dir,
        batch_size=int(hparams["batch_size"]),
        num_epochs=args.epochs,
        learning_rate=float(hparams["learning_rate"]),
        weight_decay=float(hparams["weight_decay"]),
        tile_size=args.tile_size,
        num_workers=args.num_workers,
        seed=args.seed + trial.number,
        dropout=float(hparams["dropout"]),
        loss_weights=tuned_weights,
        early_stopping=True,
        patience=int(hparams["patience"]),
        use_wandb=False,
        force_cpu=args.force_cpu,
        experiment_name=f"{args.study_name}_trial_{trial.number:04d}",
    )


def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    """Build data/model/trainer stack for a trial and return best validation score."""
    hparams = _suggest_hparams(trial)
    config = _build_config(args, trial, hparams)

    train_loader, val_loader = create_dataloaders(
        train_dirs=config.train_dirs,
        val_dir=config.val_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.tile_size,
        val_split=config.val_split,
    )

    model = EnsembleSvamitvaModel(
        num_roof_classes=config.num_roof_classes,
        pretrained=config.pretrained,
        dropout=config.dropout,
    )
    loss_fn = MultiTaskLoss(
        weights=config.loss_weights, num_roof_classes=config.num_roof_classes
    )

    trainer = Trainer(model, train_loader, val_loader, loss_fn, config)
    trainer.fit()

    best_score = float(trainer.ckpt_mgr.best_score)
    trial.set_user_attr("best_epoch", int(trainer.ckpt_mgr.best_epoch))
    logger.info(
        "Trial %d done | best_%s=%.5f | params=%s",
        trial.number,
        config.metric_for_best,
        best_score,
        hparams,
    )
    return best_score


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    logger.info(
        "Starting study '%s' | trials=%d | epochs_per_trial=%d",
        args.study_name,
        args.trials,
        args.epochs,
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.trials,
        gc_after_trial=True,
    )

    summary = {
        "study_name": args.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_trial_attrs": study.best_trial.user_attrs,
    }
    summary_path = args.output_dir / "best_trial.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Best trial summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
