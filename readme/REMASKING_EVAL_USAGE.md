# Remasking Strategy Evaluation Guide

This guide explains how to evaluate LLaDA with different Aaronson remasking strategies.

## Quick Start

### Option 1: Test a Single Strategy

Edit `eval_aaronson_watermarked.sh` and set the strategy you want:

```bash
# Options: original, dual_gate, blend, hard_favor
REMASKING_STRATEGY="original"
TAU_WM=0.2          # For dual_gate
TAU_ORIG=0.01       # For dual_gate
LAMBDA=0.7          # For blend
```

Then run:
```bash
sbatch eval_aaronson_watermarked.sh
```

### Option 2: Test All Strategies Automatically

Run the comparison script that tests all 4 strategies:
```bash
sbatch eval_all_remasking_strategies.sh
```

This will run each strategy sequentially and output results to `aaronson_all_strategies.log`.

## The Four Strategies

### 1. **original** (Best Quality)
```bash
REMASKING_STRATEGY="original"
```
- Uses only model confidence for remasking
- Best text quality
- Moderate detectability
- **Recommended for:** High-quality generation

### 2. **dual_gate** (Balanced)
```bash
REMASKING_STRATEGY="dual_gate"
TAU_WM=0.2
TAU_ORIG=0.01
```
- Requires both watermark confidence ≥ tau_wm AND model confidence ≥ tau_orig
- Balanced quality and detectability
- **Recommended for:** Most use cases

**Tuning:**
- Higher `TAU_WM` → more selective, better quality
- Higher `TAU_ORIG` → more protective of quality
- Lower values → stronger watermark

### 3. **blend** (Configurable)
```bash
REMASKING_STRATEGY="blend"
LAMBDA=0.7
```
- Blends watermark and model confidences: `conf = λ*wm_conf + (1-λ)*model_conf`
- Smooth trade-off control
- **Recommended for:** Fine-grained control

**Tuning:**
- `λ = 0.0` → equivalent to `original` (best quality)
- `λ = 0.5` → equal weighting
- `λ = 0.7` → default, detectability-focused
- `λ = 1.0` → maximum detectability

### 4. **hard_favor** (Maximum Detectability)
```bash
REMASKING_STRATEGY="hard_favor"
```
- Gives watermarked tokens high confidence (0.99)
- Strongest watermark
- **Risk:** Can degrade quality
- **Recommended for:** When detectability is critical

## Examples

### Example 1: Original Strategy (Best Quality)
```bash
# Edit eval_aaronson_watermarked.sh:
REMASKING_STRATEGY="original"
WATERMARK_STEPS=100

sbatch eval_aaronson_watermarked.sh
```

### Example 2: Dual Gate with Custom Thresholds
```bash
# Edit eval_aaronson_watermarked.sh:
REMASKING_STRATEGY="dual_gate"
TAU_WM=0.3
TAU_ORIG=0.05
WATERMARK_STEPS=150

sbatch eval_aaronson_watermarked.sh
```

### Example 3: Blend with Strong Watermark
```bash
# Edit eval_aaronson_watermarked.sh:
REMASKING_STRATEGY="blend"
LAMBDA=0.8
WATERMARK_STEPS=200

sbatch eval_aaronson_watermarked.sh
```

### Example 4: Test All Strategies
```bash
sbatch eval_all_remasking_strategies.sh
```

## Manual Command Line Usage

You can also run evaluations directly without SLURM:

```bash
# Original strategy
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,\
block_length=1024,watermark_type=aaronson,aaronson_seed=42,watermark_steps=100,\
aaronson_remasking_strategy=original,max_prompts=1

# Dual gate strategy
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,\
block_length=1024,watermark_type=aaronson,aaronson_seed=42,watermark_steps=100,\
aaronson_remasking_strategy=dual_gate,aaronson_tau_wm=0.2,aaronson_tau_orig=0.01,\
max_prompts=1

# Blend strategy
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,\
block_length=1024,watermark_type=aaronson,aaronson_seed=42,watermark_steps=100,\
aaronson_remasking_strategy=blend,aaronson_lambda=0.7,max_prompts=1

# Hard favor strategy
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,\
block_length=1024,watermark_type=aaronson,aaronson_seed=42,watermark_steps=100,\
aaronson_remasking_strategy=hard_favor,max_prompts=1
```

## Comparing Results

After running evaluations, compare:
1. **Watermark Scores:** Higher = stronger detectability
2. **Generation Quality:** Read the output text
3. **Task Performance:** Check task-specific metrics (e.g., accuracy on GSM8K)

Expected trends:
- `original`: Best quality, moderate scores
- `dual_gate`: Balanced quality and scores
- `blend (λ=0.7)`: Good scores, acceptable quality
- `hard_favor`: Highest scores, potential quality issues

## Files Modified

- `eval_aaronson_watermarked.sh`: Single strategy evaluation
- `eval_all_remasking_strategies.sh`: All strategies comparison
- `eval_llada.py`: Added remasking strategy parameters

## See Also

- `AARONSON_REMASKING_STRATEGIES.md`: Detailed strategy documentation
- `simple_generate.py`: Simple generation with strategies
- `REMASKING_SUMMARY.txt`: Quick reference


