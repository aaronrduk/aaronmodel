"""
Train a custom YOLOv8 model on SVAMITVA point features.

Expects the dataset created by prepare_yolo_dataset.py.

Usage:
    # On DGX (auto-selects best GPU):
    python scripts/train_yolo.py --data yolo_dataset/svamitva_points.yaml

    # Specify GPU and epochs:
    python scripts/train_yolo.py \
        --data yolo_dataset/svamitva_points.yaml \
        --device 0 \
        --epochs 100

    # Resume from a previous run:
    python scripts/train_yolo.py \
        --data yolo_dataset/svamitva_points.yaml \
        --resume
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_best_gpu() -> int:
    """Select GPU with most free memory."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        best, best_free = 0, 0
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best, best_free = i, free
        return best
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on SVAMITVA point features",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="yolo_dataset/svamitva_points.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model (pretrained weights)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="GPU device (auto-selects if not set)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="check/yolo_training",
        help="Output project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="svamitva_points",
        help="Run name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    args = parser.parse_args()

    # Validate dataset
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(
            f"Dataset YAML not found: {data_path}\n"
            f"Run prepare_yolo_dataset.py first."
        )
        sys.exit(1)

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Run:\n" "  pip install ultralytics")
        sys.exit(1)

    # Select device
    device = args.device
    if device is None:
        device = str(get_best_gpu())
    logger.info(f"Using device: {device}")

    # Load model
    model_path = args.model
    # Check common locations
    for candidate in [
        Path(model_path),
        Path("checkpoints") / model_path,
        Path("check") / model_path,
    ]:
        if candidate.exists():
            model_path = str(candidate)
            break

    logger.info(f"Base model: {model_path}")
    model = YOLO(model_path)

    # Train
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting YOLO Training")
    logger.info(f"  Dataset: {args.data}")
    logger.info(f"  Epochs:  {args.epochs}")
    logger.info(f"  ImgSize: {args.imgsz}")
    logger.info(f"  Batch:   {args.batch}")
    logger.info(f"  Device:  {device}")
    logger.info(f"{'='*50}\n")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        # Augmentation suited for drone imagery
        flipud=0.5,  # Vertical flip (drone images)
        fliplr=0.5,  # Horizontal flip
        mosaic=0.5,  # Mosaic augmentation
        scale=0.3,  # Scale augmentation
        hsv_h=0.015,  # Hue shift
        hsv_s=0.4,  # Saturation
        hsv_v=0.3,  # Value/brightness
        # Save settings
        save=True,
        save_period=10,  # Save every 10 epochs
        val=True,
        plots=True,
        verbose=True,
    )

    # Copy best weights to standard location
    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if best_path.exists():
        import shutil

        dest = Path("check") / "yolo_svamitva_best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, dest)
        logger.info(f"\n✅ Best YOLO weights copied to: {dest}")
        logger.info(
            "To use in inference, update yolo_path in "
            "predict.py or app.py to point to this file."
        )

    logger.info("\n🎉 YOLO training complete!")


if __name__ == "__main__":
    main()
