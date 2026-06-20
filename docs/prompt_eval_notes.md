# Prompt Evaluation Notes

## Why prompt engineering matters

The same model with different prompts gives wildly different results. Good prompts are specific, structured, and tested.

## Prompt structure

```
[style trigger], [subject], [setting], [lighting], [mood], [details]
```

Example:
```
wtrcrl_style, a cat sitting on a windowsill, rainy day, soft natural light, peaceful, watercolor painting
```

## What I've tested

### Negative prompts
- 'blurry, low quality, deformed': helps avoid bad outputs
- 'photorealistic, 3d render': helps maintain style
- Don't overdo it: too many negative terms hurt quality

### CFG scale
- 7-9: good balance of prompt adherence and quality
- Below 5: model ignores prompt
- Above 12: artifacts and oversaturation

### Steps
- 20-25: fast, decent quality
- 30-40: sweet spot for quality
- 50+: diminishing returns, slower

### Samplers
- Euler a: fast, good for testing
- DPM++ 2M Karras: best quality, slower
- DDIM: deterministic, good for comparison

## Style consistency

For consistent style across outputs:
1. Use the same trigger word
2. Keep CFG scale consistent
3. Use the same sampler
4. Seed doesn't matter as much as you think

## Prompt templates

### Portrait
```
wtrcrl_style, portrait of [person], [expression], [lighting], watercolor painting
```

### Landscape
```
wtrcrl_style, [location], [time of day], [weather], watercolor painting
```

### Object
```
wtrcrl_style, [object], [setting], [lighting], watercolor painting
```
