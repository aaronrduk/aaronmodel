#!/usr/bin/env python3
"""
Class balance analysis for SVAMITVA masks.

This script iterates over dataset samples and reports per-mask class
pixel counts and percentages.
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from data.dataset import SvamitvaDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-mask class balance.")
    parser.add_argument(
        "--root_dirs",
        nargs="+",
        default=["data"],
        help="Dataset root directories or MAP directories.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Tile size used by the dataset.",
    )
    parser.add_argument(
        "--mode",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset mode. 'val' avoids train-time resampling behavior.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of task keys (for example: building road waterbody).",
    )
    return parser.parse_args()


def analyze(dataset: SvamitvaDataset) -> Dict[str, Dict[int, int]]:
    class_counts: Dict[str, Dict[int, int]] = {}

    for idx in tqdm(range(len(dataset)), desc="Analyzing class balance"):
        sample = dataset[idx]
        for key, mask in sample.items():
            if not key.endswith("_mask") or key == "valid_mask":
                continue
            if not isinstance(mask, torch.Tensor):
                continue

            mask_np = mask.detach().cpu().numpy()
            unique, counts = np.unique(mask_np, return_counts=True)
            if key not in class_counts:
                class_counts[key] = {}
            for cls_id, count in zip(unique.tolist(), counts.tolist()):
                cls_int = int(cls_id)
                class_counts[key][cls_int] = class_counts[key].get(cls_int, 0) + int(
                    count
                )

    return class_counts


def print_summary(class_counts: Dict[str, Dict[int, int]]) -> None:
    print("\nClass balance summary:")
    for task in sorted(class_counts.keys()):
        counts = class_counts[task]
        total = sum(counts.values())
        print(f"\n{task}:")
        for cls_id in sorted(counts.keys()):
            count = counts[cls_id]
            pct = 100.0 * count / total if total else 0.0
            print(f"  Class {cls_id}: {count} pixels ({pct:.2f}%)")


def main() -> None:
    args = parse_args()
    dataset = SvamitvaDataset(
        root_dirs=[Path(d) for d in args.root_dirs],
        image_size=args.image_size,
        transform=None,
        mode=args.mode,
        tasks=args.tasks,
    )
    class_counts = analyze(dataset)
    print_summary(class_counts)


if __name__ == "__main__":
    main()
