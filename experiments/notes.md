# Experiment Notes

Dev journal — tracking VRAM usage, training times, and lessons learned.

## Hardware

- **GPU**: AMD Radeon RX 7900 XTX 24GB (Navi 31)
- **ROCm**: 6.0.x (installed via package manager on Ubuntu 22.04)
- **CPU**: Ryzen 9 7950X
- **RAM**: 64GB DDR5-6000
- **OS**: Ubuntu 22.04 LTS

---

## 2024-01-15: DreamBooth First Run

Finally got DreamBooth working on the 7900 XTX! Key findings:

**Config:**
- Model: SD 2.1
- Resolution: 512x512
- Batch size: 1
- Steps: 800
- FP16

**Results:**
- VRAM usage: ~12GB peak
- Training time: ~25 minutes for 800 steps
- Loss converged nicely, outputs look decent

**ROCm gotcha:** Had to set `HSA_OVERRIDE_GFX_VERSION=11.0.0` for Navi 31. Without it, got cryptic GPU fault errors.

**FP16 note:** Works but had a few NaN losses early on. Reduced learning rate from 1e-5 to 5e-6 and it stabilized.

---

## 2024-01-20: LoRA Training

Much more efficient than full DreamBooth fine-tuning.

**SD 1.5 LoRA:**
- Rank: 32, Alpha: 32
- VRAM: ~8GB
- Training time: ~15 min for 2000 steps
- Results: Excellent! Almost indistinguishable from full fine-tune for my use case

**SDXL LoRA:**
- Rank: 64, Alpha: 64
- Resolution: 1024x1024
- VRAM: ~14GB (with gradient checkpointing)
- Training time: ~45 min for 3000 steps
- Notes: gradient checkpointing is essential. Without it, OOM at step 1.

**FP16 issue on ROCm:** Some operations produce NaN with fp16 on ROCm but not CUDA. The `xformers` memory-efficient attention doesn't work on ROCm — had to use attention slicing instead. Added a workaround in the trainer.

---

## 2024-02-05: SDXL Experiments

Testing different configurations for SDXL training.

| Config | Resolution | Batch | Grad Accum | VRAM | Time (3k steps) |
|--------|-----------|-------|-------------|------|-----------------|
| LoRA r=32 | 1024 | 1 | 4 | 12GB | 35 min |
| LoRA r=64 | 1024 | 1 | 4 | 14GB | 42 min |
| LoRA r=128 | 1024 | 1 | 4 | 16GB | 55 min |
| Full fine-tune | 1024 | 1 | 8 | 22GB | 2.5 hr |
| LoRA r=64 | 768 | 2 | 2 | 15GB | 30 min |

**Key takeaways:**
- LoRA rank 64 seems like the sweet spot for quality vs VRAM
- 1024x1024 is doable but 768 with rank 64 is nearly as good
- Full fine-tuning is too expensive on 24GB, barely fits
- Use gradient accumulation to simulate larger batch sizes

**Sampler comparison (same seed, same prompt):**
- DPM++ 2M Karras: Best quality, 30 steps is enough
- Euler a: Good variety, needs 40+ steps
- Euler: Clean but a bit boring, 30 steps works

---

## 2024-02-12: ControlNet WIP

Starting ControlNet training. This is the hardest one.

**SD 1.5 ControlNet (canny edges):**
- Resolution: 512x512
- Batch size: 1, Grad accum: 4
- VRAM: ~16GB
- Training time: ~3 hours for 10k steps
- Status: Training works, but quality needs improvement

**SDXL ControlNet:**
- Status: TOO MUCH VRAM
- At 1024x1024, hits 23GB with batch 1
- Need to try lower resolution or fewer ControlNet blocks
- Thinking about training at 768 then upscaling

**Issues encountered:**
- NaN losses again — clamping controlnet_cond values to [-1, 1] helped
- Slow convergence — might need higher learning rate for first 1k steps
- The official diffusers ControlNet example needs some ROCm-specific tweaks

---

## 2024-02-18: Textual Inversion

Super lightweight, works great on AMD.

**Results:**
- VRAM: ~6GB (!)
- Training time: ~10 min for 3000 steps
- Quality: Decent for simple concepts, struggles with complex ones
- Much faster than LoRA for single-concept learning

**Tips:**
- Use a good initializer token — it matters a lot
- More training images (>20) helps stability
- Learning rate of 5e-4 works well, higher causes instability

---

## 2024-03-01: FID Evaluation

Set up FID scoring to compare model outputs.

**Baseline FID scores (SDXL base, 50 steps, DPM++):**
- vs COCO val2017 subset (1000 images): FID ≈ 28.5
- vs my custom dataset (200 images): FID ≈ 42.1

**After LoRA training (my dog concept):**
- FID improved to ~35.2 vs custom dataset
- Subject consistency is much better

**Note:** pytorch-fid works fine on ROCm, no issues.

---

## General ROCm Tips

1. **Always set `HSA_OVERRIDE_GFX_VERSION=11.0.0`** for Navi 31 (7900 series)
2. **Disable xformers** — it's CUDA-only. Use `enable_attention_slicing()` instead
3. **FP16 works** but watch for NaN in early training. Lower LR or add grad clipping.
4. **BF16 is available** but some ops fall back to FP32, negating the benefit
5. **Gradient checkpointing** is your best friend for fitting large models
6. **torch.compile** doesn't work on ROCm yet (as of ROCm 6.0)
7. **bitsandbytes-rocm** exists but is flaky — stick with regular AdamW
8. **VRAM reporting:** `torch.cuda.max_memory_allocated()` works on ROCm
9. **ROCm SMI** (`rocm-smi`) is useful for monitoring GPU usage during training

---

## TODO

- [ ] Test with ROCm 6.2 when available
- [ ] Try training with bf16 to see if it's faster
- [ ] SDXL ControlNet at 768 resolution
- [ ] IP-Adapter integration
- [ ] Quantized inference (INT8) for faster generation
- [ ] Multi-GPU training (if I can get a second 7900 XTX)
