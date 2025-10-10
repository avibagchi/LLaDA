# Aaronson Watermark Remasking Strategies

This document explains the different remasking strategies available for Aaronson watermarking in LLaDA, allowing you to control the trade-off between watermark detectability and generation quality.

## Overview

When using Aaronson watermarking, the model selects tokens that maximize `r^(1/p)` where `r` is a pseudorandom value and `p` is the model's probability. However, the **remasking strategy** determines which tokens are "committed" (kept) vs. re-masked for refinement in subsequent iterations.

## The Four Strategies

### 1. **`original`** (Default - Best Quality)

Uses only the original model's confidence for remasking decisions. Watermarked tokens compete fairly with non-watermarked ones based on the model's true probability distribution.

**When to use:** When generation quality is paramount and you want minimal quality degradation.

**Command:**
```bash
python simple_generate.py \
    --prompt "What is the capital of France?" \
    --watermark_steps 100 \
    --aaronson_remasking_strategy original
```

### 2. **`dual_gate`** (Balanced - Recommended)

Commits a token only if BOTH conditions are met:
- Watermark confidence ≥ `tau_wm` (default: 0.2)
- Original model confidence ≥ `tau_orig` (default: 0.01)

This protects quality by rejecting watermarked tokens that have very low model probability, giving them another chance in later iterations.

**When to use:** Good default choice balancing detectability and quality.

**Command:**
```bash
python simple_generate.py \
    --prompt "What is the capital of France?" \
    --watermark_steps 100 \
    --aaronson_remasking_strategy dual_gate \
    --aaronson_tau_wm 0.2 \
    --aaronson_tau_orig 0.01
```

**Tuning:**
- Higher `tau_wm`: More selective, better quality, potentially weaker watermark
- Higher `tau_orig`: More protective of quality
- Lower values: Stronger watermark, risk lower quality

### 3. **`blend`** (Configurable)

Combines watermark and original confidences:
```
confidence = λ * wm_conf + (1-λ) * orig_conf
```

Where `λ` (lambda) controls the blend (default: 0.7).

**When to use:** When you want fine-grained control over the detectability vs. quality trade-off.

**Command:**
```bash
python simple_generate.py \
    --prompt "What is the capital of France?" \
    --watermark_steps 100 \
    --aaronson_remasking_strategy blend \
    --aaronson_lambda 0.7
```

**Tuning:**
- `λ = 1.0`: Use only watermark confidence (strongest watermark, risk to quality)
- `λ = 0.7-0.8`: Balanced (recommended range)
- `λ = 0.5`: Equal weighting
- `λ = 0.0`: Use only original confidence (equivalent to `original`)

### 4. **`hard_favor`** (Strongest Watermark - Highest Risk)

Gives watermarked tokens a high sentinel confidence (0.99), strongly biasing the remasking process to commit them immediately.

**When to use:** When maximum detectability is required and you can tolerate potential quality degradation.

**Command:**
```bash
python simple_generate.py \
    --prompt "What is the capital of France?" \
    --watermark_steps 100 \
    --aaronson_remasking_strategy hard_favor
```

**Warning:** This strategy forces watermarked tokens to be committed even if they have very low model probability. Use with caution!

## Comparison Table

| Strategy | Detectability | Quality | Configurability | Use Case |
|----------|--------------|---------|-----------------|----------|
| `original` | Moderate | Best | None | High-quality generation |
| `dual_gate` | Good | Good | 2 thresholds | Balanced (recommended) |
| `blend` | Configurable | Configurable | 1 parameter | Fine-grained control |
| `hard_favor` | Strongest | Risky | None | Maximum detectability |

## Examples

### Example 1: High Quality with Moderate Watermark
```bash
python simple_generate.py \
    --prompt "Explain machine learning" \
    --watermark_steps 100 \
    --aaronson_remasking_strategy dual_gate \
    --aaronson_tau_wm 0.3 \
    --aaronson_tau_orig 0.05
```

### Example 2: Strong Watermark with Acceptable Quality
```bash
python simple_generate.py \
    --prompt "Explain machine learning" \
    --watermark_steps 150 \
    --aaronson_remasking_strategy blend \
    --aaronson_lambda 0.8
```

### Example 3: Maximum Watermark Strength
```bash
python simple_generate.py \
    --prompt "Explain machine learning" \
    --watermark_steps 200 \
    --aaronson_remasking_strategy hard_favor
```

## Experimental Comparison

To compare all strategies, you can modify the experiment script:

```bash
# Test different remasking strategies
for strategy in original dual_gate blend hard_favor; do
    python simple_generate.py \
        --prompt "What is the capital of France?" \
        --watermark_steps 100 \
        --aaronson_remasking_strategy $strategy
done
```

## Implementation Details

The remasking strategies work by modifying the confidence scores used to decide which tokens to commit at each iteration:

1. **Original confidence** (`orig_conf`): The model's probability `p(token|context)` for the chosen token
2. **Watermark confidence** (`wm_conf`): The model's probability for the watermark-chosen token

The strategy determines how these are combined or thresholded to make commitment decisions.

## Recommendations

1. **Start with `dual_gate`** (default thresholds) for most applications
2. **Use `original`** if generation quality is critical and you have other ways to enhance detectability
3. **Use `blend` with λ=0.7-0.8** if you want smooth control
4. **Use `hard_favor` only** for specialized applications where detectability is paramount

## See Also

- `simple_generate.py` - Single generation with configurable strategy
- `experiment_watermark_steps.py` - Compare across multiple prompts and watermark steps
- `generate.py` - Core implementation

