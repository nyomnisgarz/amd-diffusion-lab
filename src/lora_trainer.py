#!/usr/bin/env python3
"""
LoRA adapter training for SD 1.5 and SDXL on AMD ROCm.

Uses Hugging Face PEFT for LoRA injection into the UNet.
Much lower VRAM usage than full fine-tuning — SDXL LoRA fits easily on 24GB.

Usage:
    python src/lora_trainer.py --config configs/lora_sdxl.yaml
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCaptionDataset(Dataset):
    """Dataset that loads images with associated captions from a directory."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, data_dir, resolution=1024, center_crop=True, random_flip=False):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        # Load images — expects caption file or folder of images
        self.images = sorted([p for p in self.data_dir.iterdir() if p.suffix.lower() in self.EXTENSIONS])

        # Try to load captions
        self.captions = {}
        caption_file = self.data_dir / "captions.txt"
        if caption_file.exists():
            with open(caption_file) as f:
                for line in f:
                    if "|" in line:
                        fname, caption = line.strip().split("|", 1)
                        self.captions[fname] = caption
                    elif ":" in line:
                        fname, caption = line.strip().split(":", 1)
                        self.captions[fname] = caption

        logger.info(f"Found {len(self.images)} images")

        transforms_list = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        if center_crop:
            transforms_list.append(transforms.CenterCrop(resolution))
        if random_flip:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption = self.captions.get(img_path.name, "")

        return {"pixel_values": image, "caption": caption}


def main():
    parser = argparse.ArgumentParser(description="LoRA training for SD/SDXL on AMD ROCm")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    is_sdxl = "xl" in model_name.lower()
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        project_config=project_config,
    )

    logger.info(f"Loading {'SDXL' if is_sdxl else 'SD'} model: {model_name}")

    # Load model components
    if is_sdxl:
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)

    vae = pipeline.vae
    unet = pipeline.unet
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    if is_sdxl:
        tokenizer_2 = pipeline.tokenizer_2
        text_encoder_2 = pipeline.text_encoder_2

    # Freeze everything except LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    if is_sdxl:
        text_encoder_2.requires_grad_(False)

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    if config["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    # Optimizer for LoRA params only
    learning_rate = config["training"]["learning_rate"]
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # Dataset
    data_config = config["data"]
    dataset = ImageCaptionDataset(
        data_dir=data_config["dataset_dir"],
        resolution=data_config["resolution"],
        center_crop=data_config.get("center_crop", True),
        random_flip=data_config.get("random_flip", False),
    )

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        captions = [e["caption"] for e in examples]
        return {"pixel_values": pixel_values, "captions": captions}

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # LR scheduler
    num_epochs = math.ceil(config["training"]["max_train_steps"] / len(dataloader))
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"],
        num_training_steps=config["training"]["max_train_steps"],
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    if is_sdxl:
        text_encoder_2.to(accelerator.device)

    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(config["training"]["max_train_steps"]), desc="Training LoRA")

    for epoch in range(num_epochs):
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Encode images to latents
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device, dtype=torch.long,
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                captions = batch["captions"]
                if is_sdxl:
                    # SDXL dual text encoder
                    tokens_1 = tokenizer(captions, padding="max_length",
                                         max_length=tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt").to(accelerator.device)
                    tokens_2 = tokenizer_2(captions, padding="max_length",
                                           max_length=tokenizer_2.model_max_length,
                                           truncation=True, return_tensors="pt").to(accelerator.device)

                    enc1 = text_encoder(tokens_1.input_ids, output_hidden_states=True)
                    enc2 = text_encoder_2(tokens_2.input_ids, output_hidden_states=True)

                    # Concatenate penultimate hidden states
                    encoder_hidden_states = torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1)
                else:
                    tokens = tokenizer(captions, padding="max_length",
                                       max_length=tokenizer.model_max_length,
                                       truncation=True, return_tensors="pt").to(accelerator.device)
                    encoder_hidden_states = text_encoder(tokens.input_ids)[0]

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % config["training"]["checkpointing_steps"] == 0:
                    # Save LoRA weights
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(unet)
                    unwrapped.save_pretrained(str(ckpt_dir))
                    logger.info(f"Saved LoRA checkpoint to {ckpt_dir}")

            if global_step >= config["training"]["max_train_steps"]:
                break

    # Save final LoRA
    logger.info("Training complete. Saving final LoRA weights...")
    unwrapped = accelerator.unwrap_model(unet)
    unwrapped.save_pretrained(str(output_dir))
    logger.info(f"LoRA saved to {output_dir}")


if __name__ == "__main__":
    main()
