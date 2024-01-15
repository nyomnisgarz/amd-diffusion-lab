#!/usr/bin/env python3
"""
DreamBooth fine-tuning for Stable Diffusion on AMD ROCm.

Works on RX 7900 XTX (24GB VRAM) with ROCm 6.x.
FP16 is recommended — BF16 support on ROCm is spotty for some ops.

Usage:
    python src/dreambooth_train.py --config configs/dreambooth.yaml
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


class DreamBoothDataset(Dataset):
    """Simple dataset that loads instance + class images."""

    def __init__(self, instance_dir, class_dir, tokenizer, resolution=512, center_crop=True):
        self.tokenizer = tokenizer
        self.instance_dir = Path(instance_dir)
        self.class_dir = Path(class_dir) if class_dir else None

        self.instance_images = sorted(
            [p for p in self.instance_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        )
        self.class_images = []
        if self.class_dir and self.class_dir.exists():
            self.class_images = sorted(
                [p for p in self.class_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
            )

        self.num_instance = len(self.instance_images)
        self.num_class = len(self.class_images)
        self._length = max(self.num_instance, self.num_class)

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        logger.info(f"Dataset: {self.num_instance} instance images, {self.num_class} class images")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Instance image
        img_path = self.instance_images[idx % self.num_instance]
        instance_image = Image.open(img_path).convert("RGB")
        instance_image = self.transform(instance_image)

        # Class image (for prior preservation)
        if self.class_images:
            class_img_path = self.class_images[idx % self.num_class]
            class_image = Image.open(class_img_path).convert("RGB")
            class_image = self.transform(class_image)
        else:
            class_image = torch.zeros_like(instance_image)

        return {
            "instance_pixel_values": instance_image,
            "class_pixel_values": class_image,
        }


def main():
    parser = argparse.ArgumentParser(description="DreamBooth training on AMD ROCm")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ROCm workaround — sometimes need to set this before any CUDA calls
    gfx_ver = config.get("hardware", {}).get("hsa_override_gfx_version")
    if gfx_ver:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = gfx_ver
        logger.info(f"Set HSA_OVERRIDE_GFX_VERSION={gfx_ver}")

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with="tensorboard",
        project_config=project_config,
    )

    # Load model components
    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

    # Freeze text encoder and VAE
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # Enable gradient checkpointing to save VRAM
    if config["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    # Optimizer — AdamW works fine on ROCm
    # Note: 8-bit Adam via bitsandbytes-rocm can be flaky, disabled by default
    learning_rate = config["training"]["learning_rate"]
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Dataset
    dataset = DreamBoothDataset(
        instance_dir=config["instance"]["data_dir"],
        class_dir=config.get("class_data_dir"),
        tokenizer=tokenizer,
        resolution=config["training"]["resolution"],
    )

    def collate_fn(examples):
        instance_pixels = torch.stack([e["instance_pixel_values"] for e in examples])
        class_pixels = torch.stack([e["class_pixel_values"] for e in examples])
        return {
            "instance_pixel_values": instance_pixels,
            "class_pixel_values": class_pixels,
        }

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

    # Prepare with accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # Move frozen models to device
    text_encoder.to(accelerator.device, dtype=torch.float16)
    vae.to(accelerator.device, dtype=torch.float16)

    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # Tokenize prompts
    instance_prompt = config["instance"]["prompt"]
    class_prompt = config["instance"].get("class_prompt", "")

    instance_tokens = tokenizer(
        instance_prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.squeeze(0)

    if class_prompt:
        class_tokens = tokenizer(
            class_prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(config["training"]["max_train_steps"]), desc="Training")

    for epoch in range(num_epochs):
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Encode instance images
                instance_latents = vae.encode(batch["instance_pixel_values"]).latent_dist.sample()
                instance_latents = instance_latents * vae.config.scaling_factor

                noise = torch.randn_like(instance_latents)
                bsz = instance_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=instance_latents.device)

                noisy_latents = noise_scheduler.add_noise(instance_latents, noise, timesteps)

                # Instance prompt conditioning
                encoder_hidden_states = text_encoder(
                    instance_tokens.expand(bsz, -1).to(accelerator.device)
                )[0]

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Prior preservation loss
                if config["training"]["prior_preservation"] and dataset.num_class > 0:
                    class_latents = vae.encode(batch["class_pixel_values"]).latent_dist.sample()
                    class_latents = class_latents * vae.config.scaling_factor

                    class_noise = torch.randn_like(class_latents)
                    class_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=class_latents.device)
                    class_noisy = noise_scheduler.add_noise(class_latents, class_noise, class_timesteps)

                    class_encoder_hidden = text_encoder(
                        class_tokens.expand(bsz, -1).to(accelerator.device)
                    )[0]

                    class_pred = unet(class_noisy, class_timesteps, class_encoder_hidden).sample
                    prior_loss = torch.nn.functional.mse_loss(class_pred.float(), class_noise.float())
                    loss = loss + config["training"]["prior_loss_weight"] * prior_loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % config["training"]["checkpointing_steps"] == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        model_name,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=text_encoder,
                        vae=vae,
                    )
                    pipeline.save_pretrained(str(checkpoint_dir))
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

            if global_step >= config["training"]["max_train_steps"]:
                break

    # Save final model
    logger.info("Training complete. Saving final model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        unet=accelerator.unwrap_model(unet),
        text_encoder=text_encoder,
        vae=vae,
    )
    pipeline.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
