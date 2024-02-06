#!/usr/bin/env python3
"""
SDXL training experiments on RX 7900 XTX.

Experiments with different configurations:
  - Full fine-tuning (tight on VRAM at 1024x1024)
  - LoRA fine-tuning (recommended)
  - Text encoder fine-tuning
  - Various scheduler/sampler combos

VRAM notes for SDXL on 24GB:
  - Full fine-tune at 1024x1024: ~22GB with gradient checkpointing, no xformers
  - LoRA at 1024x1024: ~14GB comfortable
  - Batch size 2 with gradient accumulation: works with LoRA
  - FP16 is mandatory for SDXL on 24GB

Usage:
    python src/sdxl_experiments.py --mode train --config configs/lora_sdxl.yaml
    python src/sdxl_experiments.py --mode generate --prompt "..." --model_path ./output/lora_sdxl
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from diffusers.training_utils import compute_snr
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sampler comparison — what works best on AMD for SDXL?
SAMPLER_MAP = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "dpm++": DPMSolverMultistepScheduler,
}


def generate_images(
    model_path: str,
    prompt: str,
    negative_prompt: str = "",
    num_images: int = 4,
    steps: int = 30,
    guidance_scale: float = 7.5,
    sampler: str = "dpm++",
    seed: int = 42,
    output_dir: str = "./output/sdxl_generated",
):
    """Generate images with SDXL using a trained model or LoRA."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading SDXL model from {model_path}")

    # Load pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Load LoRA if it's a LoRA directory
    lora_path = Path(model_path) / "adapter_model.safetensors"
    if lora_path.exists():
        logger.info("Loading LoRA weights...")
        pipe.load_lora_weights(model_path)

    pipe.scheduler = SAMPLER_MAP[sampler].from_config(pipe.scheduler.config)

    # Memory optimization — ROCm doesn't have xformers, use attention slicing
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    logger.info(f"Generating {num_images} images...")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Sampler: {sampler}, Steps: {steps}, CFG: {guidance_scale}")

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images

    for i, img in enumerate(images):
        save_path = output_path / f"sdxl_{i:03d}.png"
        img.save(save_path)
        logger.info(f"Saved: {save_path}")

    return images


def benchmark_samplers(
    model_path: str,
    prompt: str,
    output_dir: str = "./output/sampler_benchmark",
):
    """Compare different samplers on the same prompt + seed."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for sampler_name in SAMPLER_MAP:
        logger.info(f"Benchmarking sampler: {sampler_name}")
        generate_images(
            model_path=model_path,
            prompt=prompt,
            sampler=sampler_name,
            num_images=1,
            seed=42,
            output_dir=str(output_path / sampler_name),
        )

    logger.info(f"Benchmark complete. Results in {output_path}")


def vae_decode_test():
    """Test VAE decode speed and memory — useful for checking ROCm perf."""
    logger.info("Loading SDXL VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to("cuda")

    # Simulate a latent tensor (batch=1, channels=4, h=128, w=128 for 1024x1024)
    latent = torch.randn(1, 4, 128, 128, dtype=torch.float16, device="cuda")

    import time
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(10):
        with torch.no_grad():
            decoded = vae.decode(latent).sample

    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 10

    logger.info(f"VAE decode: {elapsed:.3f}s per image")
    logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="SDXL experiments")
    parser.add_argument("--mode", choices=["train", "generate", "benchmark", "vae_test"], default="generate")
    parser.add_argument("--config", type=str, help="Training config (for train mode)")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, default="a beautiful sunset over a mountain lake, photorealistic, 8k")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sampler", type=str, default="dpm++", choices=list(SAMPLER_MAP.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "generate":
        generate_images(
            model_path=args.model_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            sampler=args.sampler,
            seed=args.seed,
        )
    elif args.mode == "benchmark":
        benchmark_samplers(model_path=args.model_path, prompt=args.prompt)
    elif args.mode == "vae_test":
        vae_decode_test()
    elif args.mode == "train":
        logger.info("For SDXL training, use src/lora_trainer.py with configs/lora_sdxl.yaml")
        logger.info("Full SDXL fine-tuning coming soon...")


if __name__ == "__main__":
    main()
