#!/usr/bin/env python3
"""
Textual Inversion on AMD ROCm.

Learns new concept embeddings without touching the model weights.
Very low VRAM — works on any AMD GPU with ~8GB+ VRAM.

Usage:
    python src/textual_inversion.py \
        --model_name stabilityai/stable-diffusion-2-1 \
        --data_dir ./data/my_concept \
        --placeholder_token "<my-concept>" \
        --initializer_token "object" \
        --output_dir ./output/textual_inversion \
        --max_steps 3000
"""

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
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


class ConceptDataset(Dataset):
    """Simple dataset for textual inversion — just images with optional captions."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, data_dir, resolution=512):
        self.data_dir = Path(data_dir)
        self.images = sorted([p for p in self.data_dir.iterdir() if p.suffix.lower() in self.EXTENSIONS])

        self.captions = {}
        caption_file = self.data_dir / "captions.txt"
        if caption_file.exists():
            with open(caption_file) as f:
                for line in f:
                    parts = line.strip().split("|", 1)
                    if len(parts) == 2:
                        self.captions[parts[0]] = parts[1]

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        logger.info(f"Concept dataset: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption = self.captions.get(img_path.name, "")
        return {"pixel_values": image, "caption": caption}


def main():
    parser = argparse.ArgumentParser(description="Textual Inversion on AMD ROCm")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, required=True, help="e.g. <my-concept>")
    parser.add_argument("--initializer_token", type=str, required=True, help="e.g. 'object', 'dog'")
    parser.add_argument("--output_dir", type=str, default="./output/textual_inversion")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--only_train_embeds", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_config = ProjectConfiguration(project_dir=str(output_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
        project_config=project_config,
    )

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")

    # Freeze everything
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add placeholder token
    placeholder_tokens = [args.placeholder_token]
    num_added = tokenizer.add_tokens(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    logger.info(f"Added {num_added} token(s): {placeholder_tokens}")

    # Get the token IDs
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Get initializer token embedding to initialize our new embedding
    initializer_token_id = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    if len(initializer_token_id) > 1:
        logger.warning(f"Initializer token '{args.initializer_token}' is multiple tokens, using first")

    with torch.no_grad():
        # Initialize placeholder embedding from the initializer
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for pid in placeholder_token_ids:
            token_embeds[pid] = token_embeds[initializer_token_id[0]].clone()

    # Only train the new embedding
    embedding_params = []
    for pid in placeholder_token_ids:
        param = text_encoder.get_input_embeddings().weight[pid]
        param.requires_grad = True
        embedding_params.append(param)

    # Wrap in a parameter list for the optimizer
    learnable_params = torch.nn.ParameterList(embedding_params)

    optimizer = torch.optim.AdamW(
        learnable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # Dataset
    dataset = ConceptDataset(data_dir=args.data_dir, resolution=args.resolution)

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        captions = [e["caption"] for e in examples]
        return {"pixel_values": pixel_values, "captions": captions}

    dataloader = DataLoader(dataset, batch_size=args.train_batch_size,
                             shuffle=True, collate_fn=collate_fn, num_workers=2)

    lr_scheduler = get_scheduler("constant", optimizer=optimizer,
                                  num_warmup_steps=0, num_training_steps=args.max_steps)

    text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, dataloader, lr_scheduler
    )
    vae.to(accelerator.device, dtype=torch.float16)
    unet.to(accelerator.device, dtype=torch.float16)

    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_steps), desc="Textual Inversion")

    for epoch in range(math.ceil(args.max_steps / len(dataloader))):
        text_encoder.train()
        for batch in dataloader:
            with accelerator.accumulate(text_encoder):
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device, dtype=torch.long,
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Build captions with placeholder token
                captions = [c if c else args.placeholder_token for c in batch["captions"]]
                tokens = tokenizer(captions, padding="max_length",
                                    max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt").to(accelerator.device)

                encoder_hidden_states = text_encoder(tokens.input_ids)[0]

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Re-norm embeddings to keep them stable
                with torch.no_grad():
                    token_embeds = text_encoder.get_input_embeddings().weight.data
                    for pid in placeholder_token_ids:
                        token_embeds[pid] = F.normalize(token_embeds[pid], dim=0)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if global_step >= args.max_steps:
                break

    # Save
    logger.info("Training complete. Saving learned embeddings...")
    learned_embeds = {}
    for pid, token in zip(placeholder_token_ids, placeholder_tokens):
        learned_embeds[token] = text_encoder.get_input_embeddings().weight[pid].detach().cpu()

    embed_path = output_dir / "learned_embeds.safetensors"
    from safetensors.torch import save_file
    save_file(learned_embeds, str(embed_path))
    logger.info(f"Embeddings saved to {embed_path}")

    # Also save the full pipeline for convenience
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        text_encoder=accelerator.unwrap_model(text_encoder),
    )
    pipeline.save_pretrained(str(output_dir / "pipeline"))
    logger.info(f"Pipeline saved to {output_dir / 'pipeline'}")


if __name__ == "__main__":
    main()
