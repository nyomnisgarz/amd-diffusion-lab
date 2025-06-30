# SDXL Generation Example — AMD Diffusion Lab

## Prompt

```
a photorealistic portrait of a shiba inu wearing sunglasses,
sitting on a beach at sunset, golden hour lighting,
bokeh background, 8k uhd, DSLR quality
```

## Command

```bash
python src/inference.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "a photorealistic portrait of a shiba inu wearing sunglasses, sitting on a beach at sunset, golden hour lighting, bokeh background, 8k uhd, DSLR quality" \
  --negative "blurry, low quality, distorted, deformed" \
  --width 1024 \
  --height 1024 \
  --steps 30 \
  --guidance_scale 7.5 \
  --seed 42 \
  --output shiba_beach.png
```

## Expected Output Description

The generated image should show:
- A shiba inu centered in the frame, wearing reflective aviator sunglasses
- Warm golden-orange sunset lighting from the left side
- Soft bokeh background with blurred ocean and sand dunes
- High detail on fur texture, especially around the ears and muzzle
- The sunglasses should reflect a miniature beach scene
- Overall warm color temperature with orange/amber tones

## Performance Comparison

| Metric | CPU (baseline) | Expected RX 7900 XTX |
|--------|----------------|----------------------|
| Generation time | 185 sec | ~5 sec |
| Peak memory | 12.3 GB | 6.8 GB (VRAM) |
| Images/min | 0.32 | ~12 |

## LoRA Training Example

Train a LoRA on 20 photos of a specific dog:

```bash
python src/lora_trainer.py \
  --config configs/lora_sdxl.yaml \
  --instance_data ./data/my_dog/ \
  --instance_prompt "a photo of sks dog" \
  --class_prompt "a photo of dog" \
  --max_steps 1000 \
  --learning_rate 1e-4 \
  --rank 16
```

After training, generate images of the learned subject:

```bash
python src/inference.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora_path ./output/my_dog_lora/ \
  --prompt "a photo of sks dog playing in the snow, winter wonderland" \
  --width 1024 --height 1024
```

**Expected:** The dog's distinctive features (color, markings, face shape) should be preserved in the generated image while adapting to the new scene.
