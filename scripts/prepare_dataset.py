#!/usr/bin/env python3
"""
Dataset preparation utilities for diffusion model training.

Handles:
  - Image resizing and center cropping
  - Caption generation (via BLIP or manual)
  - Train/test splitting
  - ControlNet conditioning image generation (canny, depth)
  - Dataset format conversion for diffusers

Usage:
    # Prepare a DreamBooth dataset
    python scripts/prepare_dataset.py --mode dreambooth \
        --input_dir ./raw_images/my_dog \
        --output_dir ./data/instance_images \
        --resolution 512

    # Generate canny edge conditioning images
    python scripts/prepare_dataset.py --mode canny \
        --input_dir ./raw_images \
        --output_dir ./data/conditional_pairs
"""

import argparse
import logging
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def get_image_files(directory: Path) -> list:
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS])


def resize_and_crop(input_path: Path, output_path: Path, resolution: int, center_crop=True):
    """Resize and crop a single image."""
    img = Image.open(input_path).convert("RGB")

    # Resize maintaining aspect ratio, then crop
    aspect = img.width / img.height
    if aspect > 1:
        new_w = int(resolution * aspect)
        new_h = resolution
    else:
        new_w = resolution
        new_h = int(resolution / aspect)

    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    if center_crop:
        left = (new_w - resolution) // 2
        top = (new_h - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))
    else:
        # Random crop
        left = random.randint(0, max(0, new_w - resolution))
        top = random.randint(0, max(0, new_h - resolution))
        img = img.crop((left, top, left + resolution, top + resolution))

    img.save(output_path)


def prepare_dreambooth(input_dir: str, output_dir: str, resolution: int = 512):
    """Prepare images for DreamBooth training."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = get_image_files(input_path)
    logger.info(f"Processing {len(images)} images for DreamBooth")

    for i, img_path in enumerate(images):
        output_file = output_path / f"{i:04d}.png"
        resize_and_crop(img_path, output_file, resolution)
        logger.info(f"  {img_path.name} -> {output_file.name}")

    logger.info(f"Done! {len(images)} images saved to {output_path}")


def generate_canny(input_dir: str, output_dir: str, resolution: int = 512, low_threshold=100, high_threshold=200):
    """Generate canny edge conditioning images."""
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV required for canny generation: pip install opencv-python")
        return

    input_path = Path(input_dir)
    output_dir = Path(output_dir)

    source_dir = output_dir / "source"
    target_dir = output_dir / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    images = get_image_files(input_path)
    logger.info(f"Generating canny edges for {len(images)} images")

    for img_path in images:
        # Resize target
        target_path = target_dir / img_path.name
        resize_and_crop(img_path, target_path, resolution)

        # Generate canny edge
        img = cv2.imread(str(target_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        source_path = source_dir / img_path.name
        Image.fromarray(edges_rgb).save(source_path)

    logger.info(f"Done! Source/target pairs saved to {output_dir}")


def split_dataset(data_dir: str, test_ratio: float = 0.1, seed: int = 42):
    """Split dataset into train/test."""
    data_path = Path(data_dir)
    images = get_image_files(data_path)

    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * (1 - test_ratio))
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    train_dir = data_path / "train"
    test_dir = data_path / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    for img in train_images:
        shutil.copy2(img, train_dir / img.name)
    for img in test_images:
        shutil.copy2(img, test_dir / img.name)

    logger.info(f"Split: {len(train_images)} train, {len(test_images)} test")


def main():
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument("--mode", choices=["dreambooth", "canny", "split"], required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.mode == "dreambooth":
        prepare_dreambooth(args.input_dir, args.output_dir, args.resolution)
    elif args.mode == "canny":
        generate_canny(args.input_dir, args.output_dir, args.resolution)
    elif args.mode == "split":
        split_dataset(args.input_dir, args.test_ratio)


if __name__ == "__main__":
    main()
