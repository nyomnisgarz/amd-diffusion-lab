#!/usr/bin/env python3
"""
Unified inference pipeline for generating images from trained models.

Supports:
  - Base SD/SDXL models
  - LoRA adapters (via PEFT or diffusers load_lora_weights)
  - Textual inversion embeddings
  - ControlNet conditioning

Usage:
    # Basic generation
    python src/inference.py --prompt "a photo of sks dog in a park"

    # With LoRA
    python src/inference.py --prompt "..." --lora_path ./output/lora_sdxl

    # With ControlNet (canny conditioning)
    python src/inference.py --prompt "..." --controlnet_path ./output/controlnet \
        --control_image ./test_canny.png

    # With textual inversion
    python src/inference.py --prompt "a painting in <my-style> style" \
        --embed_path ./output/textual_inversion/learned_embeds.safetensors
"""

import argparse
import logging
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pipeline(
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lora_path: str = None,
    controlnet_path: str = None,
    embed_path: str = None,
    device: str = "cuda",
):
    """Load the appropriate pipeline with optional LoRA/ControlNet/embeddings."""

    is_sdxl = "xl" in model_name.lower()
    is_controlnet = controlnet_path is not None

    logger.info(f"Loading {'SDXL' if is_sdxl else 'SD'} pipeline...")

    if is_controlnet:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_name,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
    elif is_sdxl:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

    # Load LoRA
    if lora_path:
        logger.info(f"Loading LoRA from {lora_path}")
        pipe.load_lora_weights(lora_path)

    # Load textual inversion embeddings
    if embed_path:
        logger.info(f"Loading embeddings from {embed_path}")
        pipe.load_textual_inversion(embed_path)

    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    logger.info("Pipeline ready!")

    return pipe


def generate(
    pipe,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, deformed",
    num_images: int = 1,
    steps: int = 30,
    guidance_scale: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    control_image: Image.Image = None,
):
    """Generate images from a prompt."""

    generator = torch.Generator(device="cuda").manual_seed(seed)

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_images_per_prompt": num_images,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "generator": generator,
    }

    if control_image is not None and isinstance(pipe, StableDiffusionControlNetPipeline):
        kwargs["image"] = control_image

    logger.info(f"Generating {num_images} image(s)...")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Size: {width}x{height}, Steps: {steps}, CFG: {guidance_scale}")

    result = pipe(**kwargs)
    return result.images


def main():
    parser = argparse.ArgumentParser(description="Image generation / inference")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--controlnet_path", type=str, default=None)
    parser.add_argument("--control_image", type=str, default=None)
    parser.add_argument("--embed_path", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output/generated")
    args = parser.parse_args()

    # Load pipeline
    pipe = load_pipeline(
        model_name=args.model,
        lora_path=args.lora_path,
        controlnet_path=args.controlnet_path,
        embed_path=args.embed_path,
    )

    # Load control image if provided
    control_image = None
    if args.control_image:
        control_image = Image.open(args.control_image).convert("RGB")
        control_image = control_image.resize((args.width, args.height))

    # Generate
    images = generate(
        pipe=pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_images=args.num_images,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        seed=args.seed,
        control_image=control_image,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        save_path = output_dir / f"gen_{i:03d}.png"
        img.save(save_path)
        logger.info(f"Saved: {save_path}")

    logger.info(f"Done! {len(images)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
