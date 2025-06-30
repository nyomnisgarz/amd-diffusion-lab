# amd-diffusion-lab

Stable Diffusion training experiments on AMD GPUs. Yeah, it works. Mostly.

Running on an **RX 7900 XTX 24GB** with ROCm 6.x. Turns out 24GB of VRAM is pretty sweet for diffusion models when you're not fighting CUDA dependencies every five minutes.

## What's in here

- **DreamBooth** fine-tuning — teach the model new subjects with just a handful of images
- **LoRA** adapters for SD 1.5 and SDXL — lightweight fine-tuning without melting your GPU
- **SDXL experiments** — full pipeline from training to inference
- **ControlNet** training — spatial conditioning (WIP, still tuning hyperparams)
- **Textual inversion** — learn new concepts via embedding optimization
- **Inference pipeline** — generate images from your trained models

## Setup

```bash
pip install -r requirements.txt

# Make sure ROCm is working
python -c "import torch; print(torch.cuda.is_available())"
# Should print True on ROCm. If not, check your ROCm installation.
```

## Quick Start

```bash
# Fine-tune with DreamBooth
python src/dreambooth_train.py --config configs/dreambooth.yaml

# Train a LoRA adapter
python src/lora_trainer.py --config configs/lora_sdxl.yaml

# Generate images
python src/inference.py --model_path ./output/my_lora --prompt "a photo of sks dog in a park"
```

## ROCm Notes

A few things I've learned the hard way:

- **FP16 training** works but you might hit NaN issues with certain optimizers. Check `experiments/notes.md` for workarounds.
- **SDXL at full resolution** (1024x1024) fits in 24GB with gradient checkpointing. Without it, you'll OOM.
- ROCm's `flash_attention` support is hit-or-miss. I disable it by default.
- The `HSA_OVERRIDE_GFX_VERSION=11.0.0` env var is sometimes needed for Navi 31 cards.

## Hardware Tested

| GPU | VRAM | Status |
|-----|------|--------|
| RX 7900 XTX | 24GB | ✅ Primary dev card |
| RX 7900 XT | 20GB | ✅ Works with smaller batches |
| RX 6900 XT | 16GB | ⚠️ SDXL is tight |

## Why AMD / ROCm

High-resolution image generation with Stable Diffusion is one of the most GPU-demanding workloads in deep learning. Running on AMD GPUs via ROCm unlocks practical training and inference without NVIDIA lock-in:

- **High-res generation at scale:** SDXL at 1024x1024 requires ~6-8GB VRAM just for inference. The RX 7900 XTX's 24GB lets us run full SDXL pipelines with room for batch inference (2-4 images in parallel), making production workflows viable.
- **LoRA/DreamBooth training:** Fine-tuning diffusion models is VRAM-intensive — DreamBooth on SDXL peaks at 18-22GB. AMD's 24GB consumer card handles this without optimizer offloading hacks.
- **UNet attention acceleration:** The cross-attention and self-attention layers in the UNet are the computational bottleneck. GPU parallelism delivers 30-40x speedup over CPU for the attention matrix multiplications.
- **VAE decode speed:** The VAE decoder goes from 340ms/image (CPU) to ~25-40ms on GPU — critical for interactive generation workflows.
- **Cost-effective training:** Train custom LoRA adapters and DreamBooth models on a $900 GPU that would otherwise require a $1,600+ NVIDIA card.

## AMD GPU Credit Use Plan

1. **Validate on ROCm GPUs** — Run SD 1.5 and SDXL inference, LoRA training, and DreamBooth end-to-end, verify image quality matches CUDA baseline
2. **Compare CPU vs GPU latency** — Benchmark UNet forward pass, VAE decode, text encoding, and full pipeline throughput
3. **Test fp16/bf16** — Profile mixed precision stability for UNet attention (NaN issues), VAE decode quality, and LoRA training convergence
4. **Document ROCm issues** — Track `xformers` fallback behavior, `torch.compile` warmup overhead, and VRAM fragmentation patterns
5. **Publish benchmarks** — Open results with FID scores, images/min throughput, VRAM profiles, and LoRA rank scaling curves

## License

MIT. Do whatever you want with it.

## Acknowledgements

- [diffusers](https://github.com/huggingface/diffusers) by Hugging Face — the backbone of everything here
- ROCm team at AMD for making this possible (finally)
- The AMD GPU community for all the shared troubleshooting threads
