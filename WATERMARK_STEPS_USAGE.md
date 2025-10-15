# Watermark Steps Usage Guide

The `watermark_steps` parameter controls at which generation steps watermarking is applied. **All step numbers are 1-indexed** for user-friendliness.

## Options

### 1. Watermark All Steps (Default)
```python
watermark_steps=None
```
Applies watermarking at every generation step.

### 2. Watermark First N Steps
```python
watermark_steps=100
```
Applies watermarking at steps 1, 2, 3, ..., 100 only.

### 3. Watermark Specific Steps (List)
```python
watermark_steps=[1, 2, 5, 10, 50, 100]
```
Applies watermarking only at steps 1, 2, 5, 10, 50, and 100.

**Note:** List values are **1-indexed** (not 0-indexed), matching the integer behavior.

## Examples

### Example 1: Simple generation script
```python
python simple_generate.py \
    --prompt "What is the capital of France?" \
    --watermark_steps 100  # Watermark steps 1-100
```

### Example 2: Evaluation with specific steps
```bash
accelerate launch eval_llada.py \
    --tasks gsm8k \
    --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',watermark_type=aaronson,watermark_steps=50
```

### Example 3: Using list of steps (in code)
```python
from generate import generate

output = generate(
    model=model,
    prompt=prompt_tensor,
    steps=256,
    gen_length=256,
    watermark_type='aaronson',
    watermark_steps=[1, 2, 5, 10, 25, 50, 100],  # Only watermark these specific steps
    aaronson_seed=42
)
```

## Implementation Details

Internally, the `_should_watermark(i, watermark_steps)` helper function handles the conversion:
- Loop variable `i` is 0-indexed (i=0 is the first step)
- User-facing step numbers are 1-indexed (step 1 is the first step)
- The helper converts list values: `[1, 2, 5]` → `{0, 1, 4}` internally

This ensures consistent, intuitive behavior regardless of how you specify the watermark steps.



