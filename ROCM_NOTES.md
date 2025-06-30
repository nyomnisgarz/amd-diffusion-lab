# ROCm Notes — AMD Diffusion Lab

## Test Target

- **ROCm version:** 6.x (6.2 preferred)
- **PyTorch:** ROCm-enabled build with `diffusers` from Hugging Face
- **GPU strategy:** Single GPU, gradient checkpointing for SDXL
- **Primary card:** RX 7900 XTX (24GB VRAM)

## Current Blockers

- No ROCm cloud instances available for automated testing — all validation is local
- `xformers` memory-efficient attention doesn't build for ROCm — using PyTorch's `scaled_dot_product_attention` fallback
- DreamBooth prior preservation loss requires generating class images on-the-fly — VRAM spikes during training

## Planned Tests

| Test | Metric | Status |
|------|--------|--------|
| SD 1.5 LoRA fp16 | Training time, FID score | Pending |
| SDXL LoRA fp16 | Training time, VRAM peak | Pending |
| SDXL LoRA bf16 | Training time, VRAM peak | Pending |
| DreamBooth fp16 | Subject fidelity, VRAM peak | Pending |
| UNet attention memory | VRAM per attention block | Pending |
| VAE decode speed | ms per 512x512 / 1024x1024 | Pending |
| LoRA rank scaling | VRAM at rank 4/8/16/32/64 | Pending |
| Inference throughput | Images/min at various resolutions | Pending |

## Repo-Specific Notes

### UNet Attention Memory

The UNet's cross-attention and self-attention layers are the primary VRAM consumers during both training and inference:

- **SD 1.5 UNet:** ~3.4GB in fp32, ~1.7GB in fp16 for model weights alone
- **SDXL UNet:** ~6.9GB in fp32, ~3.5GB in fp16 for model weights
- **Attention activations** during training (512x512, batch=1): ~4.2GB peak for SD 1.5
- **Attention activations** during training (1024x1024, batch=1): ~8.8GB peak for SDXL

Without `xformers`, attention memory is ~40% higher. Gradient checkpointing reduces activation memory by ~60% at the cost of ~30% slower training.

### VAE Decode Speed

The VAE decoder is often the inference bottleneck, especially at high resolutions:

- **512x512 (SD 1.5):** ~85ms per image on CPU, expected ~8-12ms on RX 7900 XTX
- **1024x1024 (SDXL):** ~340ms per image on CPU, expected ~25-40ms on RX 7900 XTX
- The VAE decoder uses ~1.8GB VRAM for SDXL at 1024x1024
- Tiled VAE decoding can reduce peak VRAM by 50% with ~20% speed penalty

### LoRA Rank Scaling

VRAM usage scales approximately linearly with LoRA rank:

| LoRA Rank | SD 1.5 Extra VRAM | SDXL Extra VRAM |
|-----------|-------------------|-----------------|
| 4 | ~80 MB | ~150 MB |
| 8 | ~160 MB | ~300 MB |
| 16 | ~320 MB | ~600 MB |
| 32 | ~640 MB | ~1.2 GB |
| 64 | ~1.3 GB | ~2.4 GB |

On 24GB VRAM: SDXL LoRA rank 32 with batch=1 fits comfortably. Rank 64 needs gradient checkpointing.

### DreamBooth Memory Profile

DreamBooth is more VRAM-hungry than LoRA due to full UNet fine-tuning:

- **SD 1.5 DreamBooth:** ~18GB peak VRAM (fp16, batch=1, 512x512)
- **SDXL DreamBooth:** requires gradient checkpointing + optimizer offloading to fit in 24GB
- Prior preservation (generating 200 class images) can be done offline to save VRAM during training

### Known ROCm Quirks for Diffusion

- `HSA_OVERRIDE_GFX_VERSION=11.0.0` needed for RDNA3 (Navi 31) on older ROCm versions
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` prevents fragmentation during long training
- `torch.compile()` on UNet: ~15% speedup after warmup, but first 3-5 iterations are very slow
- Mixed precision: bf16 is more stable than fp16 for SDXL training (fewer NaN issues in attention)
- The `diffusers` library's VAE tiling works correctly on ROCm
