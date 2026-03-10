import os
import sys
import torch
import torch.distributed as dist
import argparse
from pathlib import Path
import logging

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from models.model import EnsembleSvamitvaModel
from models.losses import MultiTaskLoss
from data.dataset import create_dataloaders
from train_engine.trainer import Trainer
from train_engine.config import TrainingConfig
from train_engine.utils import get_best_gpu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DDP-Train")


def setup():
    # Initialize the process group
    dist.init_process_group("nccl")
    # Set the device for this process
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="SVAMITVA DDP Training")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Total batch size across all GPUs"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="check")
    parser.add_argument("--train_dirs", type=str, nargs="+", required=True)
    args = parser.parse_args()

    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Device for this process
    device = torch.device(f"cuda:{local_rank}")

    # Configuration
    # Note: TrainingConfig expect per-gpu batch size if using DDP-ready code
    # but our Trainer handles the world_size logic if we pass it correctly.
    # Actually, we divide the requested total batch size by world_size.
    per_gpu_batch = args.batch_size // world_size

    config = TrainingConfig(
        batch_size=per_gpu_batch,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        checkpoint_dir=Path(args.save_dir),
        force_cpu=False,
    )

    # DataLoaders
    train_dirs = [Path(d) for d in args.train_dirs]
    train_loader, val_loader = create_dataloaders(
        train_dirs=train_dirs,
        batch_size=per_gpu_batch,
        num_workers=args.num_workers,
        distributed=True,
    )

    # Model & Loss
    model = EnsembleSvamitvaModel()
    loss_fn = MultiTaskLoss()

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
    )

    if rank == 0:
        logger.info(f"🚀 Starting DDP Training on {world_size} GPUs")
        logger.info(f"Config: Batch/GPU={per_gpu_batch}, Total={args.batch_size}")

    trainer.fit()

    cleanup()


if __name__ == "__main__":
    main()
