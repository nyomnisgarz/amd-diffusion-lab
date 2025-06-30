# CPU Baseline Benchmarks — AMD Diffusion Lab

All benchmarks run on CPU without GPU acceleration.

## Environment

- **CPU:** AMD Ryzen 9 7950X (16C/32T)
- **RAM:** 64GB DDR5-6000
- **OS:** Ubuntu 22.04
- **Python:** 3.10
- **PyTorch:** 2.3.0 (CPU-only for baseline)
- **Diffusers:** 0.28.0

## SD 1.5 Inference (512x512, 50 steps)

| Metric | Value |
|--------|-------|
| Single image generation | 38.2 sec |
| Peak memory (RSS) | 4.8 GB |
| UNet forward pass (per step) | 620 ms |
| VAE decode | 85 ms |
| Text encoder | 42 ms |
| Throughput | 1.6 images/min |

## SDXL Inference (1024x1024, 50 steps)

| Metric | Value |
|--------|-------|
| Single image generation | 185.4 sec |
| Peak memory (RSS) | 12.3 GB |
| UNet forward pass (per step) | 3,200 ms |
| VAE decode | 340 ms |
| Text encoder (dual) | 120 ms |
| Throughput | 0.32 images/min |

## LoRA Training — SD 1.5 (20 images, rank=16)

| Metric | Value |
|--------|-------|
| Training time (1000 steps) | 4.8 hours |
| Peak memory (RSS) | 8.2 GB |
| Max batch size | 2 |
| Average loss (final 100 steps) | 0.085 |

## LoRA Training — SDXL (20 images, rank=16)

| Metric | Value |
|--------|-------|
| Training time (1000 steps) | 14.2 hours |
| Peak memory (RSS) | 18.6 GB |
| Max batch size | 1 |
| Average loss (final 100 steps) | 0.092 |

## DreamBooth — SD 1.5 (20 images)

| Metric | Value |
|--------|-------|
| Training time (800 steps) | 6.1 hours |
| Peak memory (RSS) | 14.8 GB |
| Prior preservation images (200) | 2.1 hours to generate |

---

**AMD GPU benchmark is pending — no access to ROCm hardware yet.**

Expected improvements on RX 7900 XTX (24GB VRAM):
- SD 1.5 inference: ~25-35x speedup (1.1-1.5 sec per image)
- SDXL inference: ~30-40x speedup (4.5-6 sec per image)
- LoRA training SDXL: ~8-12x speedup (1.2-1.8 hours for 1000 steps)
- DreamBooth SDXL: now feasible in 24GB with gradient checkpointing
- Batch inference: 4-8 images in parallel for SD 1.5, 2-4 for SDXL
