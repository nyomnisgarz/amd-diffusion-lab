#!/usr/bin/env python3
"""
ControlNet training on AMD ROCm — WIP.

Training ControlNet from scratch is VRAM-hungry. On 24GB:
  - SD 1.5 ControlNet: fits with batch size 1-2
  - SDXL ControlNet: barely fits, needs aggressive memory optimization

Current status: getting SD 1.5 ControlNet to work on the 7900 XTX.
SDXL ControlNet is still experimental.

Usage:
    python src/controlnet_trainer.py --dataset_dir ./data/conditional_pairs \
        --model_name runwayml/stable-diffusion-v1-5 \
        --output_dir ./output/controlnet \
        --resolution 512 \
        --max_steps 10000
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ControlNetDataset(Dataset):
    """
    Dataset for ControlNet training.

    Expects paired images:
      - source/ : conditioning images (canny edges, depth maps, etc.)
      - target/ : target images
      - captions.txt : filename|caption pairs

    TODO: support multiple conditioning types (canny, depth, pose, etc.)
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, data_dir, resolution=512, tokenizer=None):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution

        self.source_dir = self.data_dir / "source"
        self.target_dir = self.data_dir / "target"

        self.images = sorted([
            p for p in self.target_dir.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        ])

        # Load captions
        self.captions = {}
        caption_file = self.data_dir / "captions.txt"
        if caption_file.exists():
            with open(caption_file) as f:
                for line in f:
                    if "|" in line:
                        fname, caption = line.strip().split("|", 1)
                        self.captions[fname] = caption

        logger.info(f"Found {len(self.images)} training pairs")

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        source_path = self.source_dir / img_path.name

        image = Image.open(img_path).convert("RGB")
        source = Image.open(source_path).convert("RGB")

        image = self.transform(image)
        source = self.transform(source)

        caption = self.captions.get(img_path.name, "")

        # Tokenize caption
        if self.tokenizer:
            tokens = self.tokenizer(
                caption, padding="max_length", max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).input_ids.squeeze(0)
        else:
            tokens = torch.zeros(77, dtype=torch.long)

        return {
            "pixel_values": image,
            "conditioning_pixel_values": source,
            "input_ids": tokens,
        }


def main():
    parser = argparse.ArgumentParser(description="ControlNet training on AMD ROCm")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="./output/controlnet")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--checkpoint_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16"])
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
    )

    # Load base model components
    logger.info(f"Loading model: {args.model_name}")

    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")

    # Initialize ControlNet from UNet
    controlnet = ControlNetModel.from_unet(unet)

    # Freeze base model
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    dataset = ControlNetDataset(
        data_dir=args.dataset_dir,
        resolution=args.resolution,
        tokenizer=tokenizer,
    )

    def collate_fn(examples):
        return {
            "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
            "conditioning_pixel_values": torch.stack([e["conditioning_pixel_values"] for e in examples]),
            "input_ids": torch.stack([e["input_ids"] for e in examples]),
        }

    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=2,
    )

    num_epochs = math.ceil(args.max_steps / len(dataloader))
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0,
                                  num_training_steps=args.max_steps)

    controlnet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, dataloader, lr_scheduler
    )

    text_encoder.to(accelerator.device, dtype=torch.float16)
    vae.to(accelerator.device, dtype=torch.float16)
    unet.to(accelerator.device, dtype=torch.float16)

    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_steps), desc="ControlNet training")

    for epoch in range(num_epochs):
        controlnet.train()
        for batch in dataloader:
            with accelerator.accumulate(controlnet):
                # Encode images to latents
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device, dtype=torch.long,
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get ControlNet conditioning
                controlnet_image = batch["conditioning_pixel_values"]
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict noise with UNet + ControlNet residuals
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpoint_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(controlnet)
                    unwrapped.save_pretrained(str(ckpt_dir))
                    logger.info(f"Saved checkpoint: {ckpt_dir}")

            if global_step >= args.max_steps:
                break

    # Save final
    logger.info("Training complete. Saving ControlNet...")
    unwrapped = accelerator.unwrap_model(controlnet)
    unwrapped.save_pretrained(str(output_dir))
    logger.info(f"ControlNet saved to {output_dir}")


if __name__ == "__main__":
    main()
