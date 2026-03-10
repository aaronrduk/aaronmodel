"""
Training loop for SVAMITVA SAM2 model.

Features:
    - Mixed-precision training (AMP)
    - Cosine annealing with linear warmup
    - Gradient clipping
    - Staged backbone unfreezing
    - Early stopping on avg IoU
    - Top-K checkpoint saving
    - Per-epoch metric logging
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)


# ── Utilities ─────────────────────────────────────────────────────────────────


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_targets(batch: dict, device: torch.device) -> dict:
    """Move all tensor targets to device, skip metadata fields."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def get_best_gpu() -> int:
    """Find the GPU index with the most free memory."""
    if not torch.cuda.is_available():
        return 0

    n_devices = torch.cuda.device_count()
    if n_devices <= 1:
        return 0

    best_idx = 0
    max_free = 0

    for i in range(n_devices):
        try:
            free, total = torch.cuda.mem_get_info(i)
            if free > max_free:
                max_free = free
                best_idx = i
        except Exception:
            continue

    return best_idx


def get_device(config: TrainingConfig) -> torch.device:
    """Determine the best available device."""
    if config.force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        best_gpu = get_best_gpu()
        logger.info(f"Total CUDA GPUs detected: {n_gpus}")
        logger.info(f"Auto-selected primary GPU:{best_gpu} (most free memory)")
        return torch.device(f"cuda:{best_gpu}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Learning Rate Schedule ────────────────────────────────────────────────────


class WarmupCosineScheduler:
    """Linear warmup → cosine annealing LR schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        lr_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            import math

            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(base_lr * scale, self.lr_min)


# ── Checkpoint Manager ────────────────────────────────────────────────────────


class CheckpointManager:
    """Saves top-K checkpoints and supports early stopping."""

    def __init__(
        self,
        save_dir: Path,
        save_top_k: int = 3,
        metric_name: str = "avg_iou",
        patience: int = 25,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.metric_name = metric_name
        self.patience = patience

        self.best_score = -float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.top_k: list = []  # (score, path) sorted ascending

    def save_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        rank: int = 0,
    ):
        """Save a 'best_latest.pt' checkpoint for crash recovery."""
        if rank != 0:
            return

        # Handle DataParallel/DDP state_dict
        if hasattr(model, "module"):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        state = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "metrics": metrics,
        }
        path = self.save_dir / "best_latest.pt"
        torch.save(state, path)
        logger.debug(f"Saved latest checkpoint to {path}")

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        rank: int = 0,
    ) -> bool:
        """
        Save model to best.pt if it's the new best.
        """
        score = metrics.get(self.metric_name, 0)
        is_best = score > self.best_score

        # Always save latest for crash recovery (only rank 0)
        self.save_latest(model, optimizer, scheduler, epoch, metrics, rank)

        if is_best and rank == 0:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_no_improve = 0

            # Handle DataParallel state_dict
            if hasattr(model, "module"):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            state = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "metrics": metrics,
                "best_score": self.best_score,
            }

            best_path = self.save_dir / "best.pt"
            torch.save(state, best_path)
            logger.info(
                f"★ New best model saved to best.pt "
                f"(epoch {epoch}, {self.metric_name}={score:.4f})"
            )
        else:
            self.epochs_no_improve += 1

        return is_best

    @property
    def should_stop(self) -> bool:
        return self.epochs_no_improve >= self.patience


# ── Training Engine ───────────────────────────────────────────────────────────


class Trainer:
    """
    Full training engine for SVAMITVA model.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, loss_fn, config)
        trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        config: TrainingConfig,
        start_epoch: int = 1,
    ):
        self.config = config
        self.start_epoch = start_epoch
        self.device = get_device(config)
        logger.info(f"Using device: {self.device}")

        # Handle Distributed or DataParallel
        raw_model = model.to(self.device)
        self.is_multi_gpu = False
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        if self.is_distributed:
            logger.info(
                f"🚀 Activating DDP on rank {self.rank}/{self.world_size} "
                f"(Device: {self.device})"
            )
            self.model = DDP(
                raw_model,
                device_ids=(
                    [self.device.index] if self.device.type == "cuda" else None
                ),
            )
            self.is_multi_gpu = True
        elif (
            torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and not config.force_cpu
        ):
            n_gpus = torch.cuda.device_count()
            # DataParallel requires model on device_ids[0] = cuda:0
            try:
                self.device = torch.device("cuda:0")
                raw_model = raw_model.to(self.device)
                logger.info(
                    f"🚀 Activating DataParallel on "
                    f"{n_gpus} GPUs (Master: {self.device})"
                )
                self.model = nn.DataParallel(raw_model)
                self.is_multi_gpu = True
            except RuntimeError as dp_err:
                logger.warning(
                    f"⚠️ DataParallel failed: {dp_err}. " f"Falling back to single GPU."
                )
                self.model = raw_model
                self.is_multi_gpu = False
        else:
            self.model = raw_model
            if not config.force_cpu:
                n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if n_gpus <= 1:
                    logger.info(f"Multi-GPU skipped: {n_gpus} GPU(s) visible.")
            else:
                logger.info("Multi-GPU skipped: force_cpu is True.")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)

        # Optimizer
        if hasattr(model, "get_param_groups"):
            param_groups = getattr(model, "get_param_groups")(config.learning_rate)
        else:
            param_groups = model.parameters()

        self.optimizer: torch.optim.Optimizer
        if config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                param_groups,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )

        # Scheduler
        self.scheduler: Any
        if config.optimizer == "adamw":
            # Using OneCycleLR for faster convergence
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                epochs=config.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,  # 30% warmup
                div_factor=25,
                final_div_factor=1e4,
            )
        else:
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                config.warmup_epochs,
                config.num_epochs,
                config.lr_min,
            )
        self._scheduler_step_per_batch = isinstance(
            self.scheduler, torch.optim.lr_scheduler.OneCycleLR
        )

        # AMP
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Metrics
        self.metrics = MetricsTracker(num_roof_classes=config.num_roof_classes)

        # Checkpoint manager
        self.ckpt_mgr = CheckpointManager(
            Path(config.checkpoint_dir),
            config.save_top_k,
            config.metric_for_best,
            config.patience,
        )

        # TensorBoard writer (only on rank 0)
        self.tb_writer = None
        if self.rank == 0:
            try:
                # Check if logging is explicitly disabled to save file handles
                if getattr(config, "enable_tensorboard", True):
                    from torch.utils.tensorboard import SummaryWriter

                    self.tb_writer = SummaryWriter(log_dir=str(config.log_dir))
                    logger.info("TensorBoard logging enabled.")
                else:
                    logger.info("TensorBoard logging disabled by config.")
            except (ImportError, OSError) as e:
                logger.warning(f"TensorBoard logging disabled: {e}")

        # History
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "metrics": [],
            "train_breakdown": [],
            "val_breakdown": [],
        }

    def fit(self):
        """Run full training loop with detailed logging and TensorBoard visualization."""
        set_seed(self.config.seed)
        self.was_interrupted = False
        logger.info(
            f"Starting training for {self.config.num_epochs} epochs "
            f"({len(self.train_loader)} train batches, "
            f"{len(self.val_loader)} val batches)"
        )
        completed_successfully = False

        try:
            for epoch in range(self.start_epoch, self.config.num_epochs + 1):
                # For DDP, we need to set the epoch on the sampler for correct shuffling
                if self.is_distributed and hasattr(self.train_loader, "sampler"):
                    if hasattr(self.train_loader.sampler, "set_epoch"):
                        self.train_loader.sampler.set_epoch(epoch)

                # ... unfreezing logic ...
                model_to_unfreeze = (
                    self.model.module if self.is_multi_gpu else self.model
                )
                if epoch == self.config.freeze_backbone_epochs + 1:
                    logger.info(f"Epoch {epoch}: Unfreezing backbone")
                    model_to_unfreeze.unfreeze_backbone()

                if epoch <= self.config.freeze_backbone_epochs:
                    model_to_unfreeze.freeze_backbone()

                # Train
                train_loss, train_breakdown = self._train_epoch(epoch)
                self.history["train_loss"].append(train_loss)
                self.history["train_breakdown"].append(train_breakdown)

                # Validate
                val_loss, val_metrics = self._validate_epoch(epoch)
                self.history["val_loss"].append(val_loss)
                self.history["metrics"].append(val_metrics)
                self.history["val_breakdown"].append(
                    {k: val_metrics.get(k, 0) for k in train_breakdown.keys()}
                )

                # LR schedule
                if not self._scheduler_step_per_batch:
                    if isinstance(self.scheduler, WarmupCosineScheduler):
                        # WarmupCosineScheduler expects a zero-based epoch index.
                        self.scheduler.step(epoch - 1)
                    else:
                        self.scheduler.step()
                current_lr = self.optimizer.param_groups[-1]["lr"]

                # Log
                avg_iou = val_metrics.get("avg_iou", 0)
                avg_dice = val_metrics.get("avg_dice", 0)
                logger.info(
                    f"Epoch {epoch}/{self.config.num_epochs} │ "
                    f"Train Loss: {train_loss:.4f} │ Val Loss: {val_loss:.4f} │ "
                    f"Avg IoU: {avg_iou:.4f} │ Avg Dice: {avg_dice:.4f} │ "
                    f"LR: {current_lr:.2e}"
                )

                # Save checkpoint (includes best_latest.pt save)
                self.ckpt_mgr.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics,
                    self.rank,
                )

                # Early stopping
                if self.config.early_stopping and self.ckpt_mgr.should_stop:
                    logger.info(
                        f"Early stopping after {self.config.patience} epochs "
                        f"without improvement. Best epoch: {self.ckpt_mgr.best_epoch}"
                    )
                    break
            completed_successfully = True

        except (KeyboardInterrupt, Exception) as e:
            logger.error(f"Training interrupted or crashed: {e}")
            logger.info("Saving emergency checkpoint to best_latest.pt...")
            # Emergency save using current state (only rank 0)
            current_epoch = epoch if "epoch" in locals() else 0
            metrics = {"epoch_interrupted": True}
            self.ckpt_mgr.save_latest(
                self.model,
                self.optimizer,
                self.scheduler,
                current_epoch,
                metrics,
                self.rank,
            )
            if isinstance(e, KeyboardInterrupt):
                self.was_interrupted = True
                logger.info("Keyboard interrupt received. Exiting.")
            else:
                raise e
        finally:
            # Final save if we completed all epochs successfully
            if (
                completed_successfully
                and "epoch" in locals()
                and epoch == self.config.num_epochs
            ):
                # Do not overwrite ckpt manager's best.pt with the last epoch.
                # Save final epoch weights separately for reproducibility/debug.
                final_path = self.ckpt_mgr.save_dir / "final_last_epoch.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": (
                            self.model.module.state_dict()
                            if self.is_multi_gpu
                            else self.model.state_dict()
                        ),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": (
                            self.scheduler.state_dict()
                            if self.scheduler is not None
                            else None
                        ),
                    },
                    final_path,
                )
                logger.info(
                    "Training completed successfully. "
                    f"Last-epoch checkpoint saved to {final_path} "
                    "(best.pt preserved from best validation epoch)."
                )

        # Save training history
        history_path = self.config.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"Training history saved to {history_path}")

        if self.tb_writer:
            self.tb_writer.close()
            logger.info("TensorBoard logs written.")

        logger.info(
            f"Training complete. Best {self.config.metric_for_best}: "
            f"{self.ckpt_mgr.best_score:.4f} at epoch "
            f"{self.ckpt_mgr.best_epoch}"
        )

    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        breakdown_sums: Dict[str, float] = {}
        n_batches = 0

        # For DDP, only show progress bar on rank 0
        train_iter: Any = self.train_loader
        if self.rank == 0:
            train_iter = tqdm(
                self.train_loader,
                desc=f"Train Epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
            )

        for i, batch in enumerate(train_iter):
            batch = move_targets(batch, self.device)
            images = batch["image"]

            self.optimizer.zero_grad()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                predictions = self.model(images)
                loss, breakdown = self.loss_fn(predictions, batch)

            # Skip NaN/Inf loss batches
            if torch.isnan(loss) or torch.isinf(loss):
                if self.rank == 0:
                    train_iter.set_postfix(loss="NaN-skip")
                continue

            # Backward and Step
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if self._scheduler_step_per_batch:
                self.scheduler.step()

            for k, v in breakdown.items():
                breakdown_sums[k] = breakdown_sums.get(k, 0) + v

            if self.rank == 0:
                train_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        avg_breakdown = {k: v / max(n_batches, 1) for k, v in breakdown_sums.items()}
        return avg_loss, avg_breakdown

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run validation and compute metrics."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        n_batches = 0

        # For DDP, only show progress bar on rank 0
        val_iter: Any = self.val_loader
        if self.rank == 0:
            val_iter = tqdm(
                self.val_loader,
                desc=f"Val Epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
            )

        for batch in val_iter:
            batch = move_targets(batch, self.device)
            images = batch["image"]

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                predictions = self.model(images)
                loss, _ = self.loss_fn(predictions, batch)

            total_loss += loss.item()
            n_batches += 1

            self.metrics.update(predictions, batch)

        avg_loss = total_loss / max(n_batches, 1)
        metrics = self.metrics.compute()
        metrics["val_loss"] = avg_loss
        return avg_loss, metrics
