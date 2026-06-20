# amd-diffusion-lab

This repo is a small diffusion playground for low-VRAM workflows: LoRA configs, inference notes, and practical experiments around getting stable outputs without needing a huge GPU.

## Why I built this

Most diffusion tutorials assume you have 24GB+ VRAM. I wanted to see how far I can push Stable Diffusion and LoRA training on a smaller GPU. The goal is practical workflows that work on 8-12GB VRAM.

## What's in here

- LoRA training configs optimized for low VRAM
- Inference settings for quality vs speed tradeoffs
- Prompt engineering notes

## Current experiments

1. SD 1.5 LoRA training on 8GB VRAM
2. Inference optimization: batch size, steps, sampler
3. Prompt templates for consistent style

## What I'm NOT doing

- Not training SDXL (too heavy for my setup)
- Not doing ControlNet (separate project)
- Not building a generation service

This is for personal creative work: generating consistent art styles for projects.

## Quick start

```bash
pip install -r requirements.txt
python diffusion_lab.py generate --prompt 'a cat in space' --steps 25
python diffusion_lab.py train --config configs/sd15_lora_low_vram.yaml
```

## Examples

- `examples/lora_training_plan.md` -- training a style LoRA
- `configs/sd15_lora_low_vram.yaml` -- low-VRAM training config


## Troubleshooting
**Q: Getting OOM errors?**
A: Reduce batch size or enable gradient checkpointing.