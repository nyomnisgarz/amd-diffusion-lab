# Low VRAM Diffusion Notes

## The challenge

SD 1.5 needs ~10GB for inference at 512x512. SDXL needs ~16GB. Training needs even more.

## Tricks for 8GB VRAM

### Inference
1. Use FP16 (half precision): saves ~40% memory
2. Use xformers attention: saves ~20% memory
3. Reduce batch size to 1
4. Use 512x512 resolution (not higher)
5. Use fewer steps: 20-25 is often enough

### Training
1. Gradient checkpointing: saves ~30% memory
2. 8-bit Adam optimizer: saves ~20% memory
3. Mixed precision (FP16): saves ~40% memory
4. Batch size 1 with gradient accumulation
5. LoRA instead of full fine-tuning

### Combined
With all tricks, SD 1.5 training fits in 8GB:
- Base model: ~5GB
- Training overhead: ~2GB
- Total: ~7GB

## Performance impact

| Trick | Memory saved | Speed impact |
|-------|-------------|-------------|
| FP16 | 40% | Minimal |
| xformers | 20% | 10% faster |
| Gradient checkpointing | 30% | 20% slower |
| 8-bit Adam | 20% | Minimal |

## What doesn't fit

- SDXL training: needs 16GB+ even with all tricks
- High resolution (1024x1024): OOM at inference
- Large batch sizes: batch_size > 1 OOMs
- ControlNet + LoRA training: too heavy

## Workarounds

- Use cloud GPU for SDXL training, then run inference locally
- Use img2img for higher resolution (512 -> 1024)
- Use sequential generation instead of batching
