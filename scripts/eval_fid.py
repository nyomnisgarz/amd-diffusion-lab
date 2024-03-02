#!/usr/bin/env python3
"""
FID (Fréchet Inception Distance) evaluation for generated images.

Computes FID between a set of generated images and real reference images.
Uses the standard pytorch-fid implementation.

Usage:
    python scripts/eval_fid.py \
        --real_dir ./data/test_images \
        --generated_dir ./output/generated \
        --batch_size 16 \
        --device cuda
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageFolderDataset(Dataset):
    """Simple image folder dataset for FID computation."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.images = sorted([p for p in self.root_dir.iterdir() if p.suffix.lower() in EXTENSIONS])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_inception_model(device="cuda"):
    """Load InceptionV3 for FID computation."""
    from torchvision.models import inception_v3

    model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove classification head
    model.eval()
    model.to(device)
    return model


def compute_statistics(dataloader, model, device="cuda"):
    """Compute mean and covariance of Inception features."""
    features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing features"):
            batch = batch.to(device)
            # Inception expects 299x299
            if batch.shape[-1] != 299:
                batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode="bilinear")

            feat = model(batch)
            feat = feat.cpu().numpy()
            features.append(feat)

    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate FID between two distributions."""
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        msg = f"FID calculation: adding {eps} to diagonal of cov estimates"
        logger.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def main():
    parser = argparse.ArgumentParser(description="FID Evaluation")
    parser.add_argument("--real_dir", type=str, required=True, help="Real/reference images directory")
    parser.add_argument("--generated_dir", type=str, required=True, help="Generated images directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    real_dataset = ImageFolderDataset(args.real_dir, transform=transform)
    gen_dataset = ImageFolderDataset(args.generated_dir, transform=transform)

    logger.info(f"Real images: {len(real_dataset)}")
    logger.info(f"Generated images: {len(gen_dataset)}")

    if len(real_dataset) == 0 or len(gen_dataset) == 0:
        logger.error("No images found in one or both directories!")
        return

    real_loader = DataLoader(real_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False)

    logger.info("Loading InceptionV3...")
    model = get_inception_model(device=args.device)

    logger.info("Computing statistics for real images...")
    mu_real, sigma_real = compute_statistics(real_loader, model, device=args.device)

    logger.info("Computing statistics for generated images...")
    mu_gen, sigma_gen = compute_statistics(gen_loader, model, device=args.device)

    fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    logger.info(f"\n{'='*50}")
    logger.info(f"FID Score: {fid:.4f}")
    logger.info(f"{'='*50}")
    logger.info(f"Lower is better. Typical good range: < 50")
    logger.info(f"Real: {args.real_dir} ({len(real_dataset)} images)")
    logger.info(f"Generated: {args.generated_dir} ({len(gen_dataset)} images)")

    # Save results
    results_path = Path(args.generated_dir) / "fid_results.txt"
    with open(results_path, "w") as f:
        f.write(f"FID Score: {fid:.4f}\n")
        f.write(f"Real images: {len(real_dataset)} ({args.real_dir})\n")
        f.write(f"Generated images: {len(gen_dataset)} ({args.generated_dir})\n")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
