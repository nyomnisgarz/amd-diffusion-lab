# LoRA Training Plan: Watercolor Style

## Goal

Train a LoRA that makes SD 1.5 generate watercolor-style illustrations.

## Dataset

- 20 watercolor illustrations from personal collection
- Resolution: 512x512
- Captioned manually with style description
- Trigger word: 'wtrcrl_style'

## Training config

See `configs/sd15_lora_low_vram.yaml`

Key settings:
- LoRA rank: 4 (small but enough for style)
- Steps: 1000 (sweet spot for 20 images)
- Learning rate: 1e-4 (stable, no divergence)

## Expected output

```
wtrcrl_style, a landscape painting
-> watercolor-style landscape

wtrcrl_style, a portrait of a woman
-> watercolor-style portrait
```

## Timeline

- Dataset prep: 1 hour
- Training: ~2 hours on 8GB GPU
- Testing: 30 minutes
- Total: ~3.5 hours

## What to watch for

1. Overfitting: if outputs look identical to training images, reduce steps
2. Artifacts: if outputs have weird colors, reduce learning rate
3. Style bleed: if style applies to everything, adjust trigger word weight
