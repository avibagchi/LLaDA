# Aaronson Gumbel Softmax Watermarking Implementation

This implementation adds the Aaronson Gumbel softmax watermarking strategy to the LLaDA model.

## Overview

The Aaronson watermarking scheme works as follows:

### Generation
At each position t, choose the token i=i(t) that maximizes:
```
r_{t,i}^{1/p_{t,i}}
```
where:
- `r_{t,i}` is a pseudorandom function based on position t and token i
- `p_{t,i}` is the model's probability for token i at position t

### Detection
Calculate the watermark detection score as:
```
sum_{t=1}^{n} ln(1 / (1 - r_{t,i(t)}))
```
A higher score indicates the text is more likely to be watermarked.

## New Functions Added

### In `generate.py`:

1. **`generate_pseudo_random_values(position, vocab_size, seed=42)`**
   - Generates pseudorandom values r_{t,i} for each token at a position
   - Uses position-based seeding for deterministic pseudorandom function

2. **`apply_aaronson_gumbel_watermark(logits, mask_positions, vocab_size, position_offset=0)`**
   - Applies the Aaronson watermarking scheme during generation
   - Modifies logits to select tokens that maximize r^{1/p}

3. **`calculate_aaronson_watermark_score(generated_tokens, vocab_size=126464, seed=42)`**
   - Calculates the watermark detection score
   - Returns: watermark_score, actual_length, per_token_scores

### Modified Functions:

1. **`generate()`**
   - Added parameters:
     - `watermark_type='green_list'`: Choose 'green_list' or 'aaronson'
     - `aaronson_seed=42`: Seed for pseudorandom function
   - Now supports both green list and Aaronson watermarking

2. **`LLaDAEvalHarness.__init__()`** in `eval_llada.py`
   - Added parameters:
     - `watermark_type='green_list'`
     - `aaronson_seed=42`

3. **`generate_until()`** in `eval_llada.py`
   - Now detects watermarks using the appropriate method based on `watermark_type`
   - For Aaronson, reports: score, length, and normalized score

## Usage Examples

### Example 1: Using Aaronson Watermarking in eval_llada.py

```bash
# Run GSM8K with Aaronson watermarking
accelerate launch eval_llada.py \
    --tasks gsm8k \
    --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,watermark_type=aaronson,aaronson_seed=42,max_prompts=10
```

### Example 2: Using Aaronson Watermarking with Shell Script

Update `eval_watermarked_llada.sh` to use Aaronson watermarking:

```bash
# Aaronson watermarking parameters
WATERMARK_TYPE="aaronson"
AARONSON_SEED=42
WATERMARK_STEPS=200

echo "Running GSM8K with Aaronson watermarking..."
accelerate launch eval_llada.py \
    --tasks gsm8k \
    --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,watermark_type=$WATERMARK_TYPE,aaronson_seed=$AARONSON_SEED,watermark_steps=$WATERMARK_STEPS,max_prompts=100
```

### Example 3: Using Green List Watermarking (Original Method)

```bash
# Green list watermarking parameters (original)
WATERMARK_TYPE="green_list"
GAMMA=0.025
AMPLIFICATION=0
WATERMARK_STEPS=200

echo "Running GSM8K with green list watermarking..."
accelerate launch eval_llada.py \
    --tasks gsm8k \
    --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,watermark_type=$WATERMARK_TYPE,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS,max_prompts=100
```

## Output Format

The JSON output file now includes:
- `watermark_type`: Either 'green_list' or 'aaronson'
- `watermark_detection`: Detection information specific to the watermarking method used

### For Green List:
```json
{
  "watermark_type": "green_list",
  "watermark_detection": "Green token matches: 45/200 (22.50%), Z-score: 3.45"
}
```

### For Aaronson:
```json
{
  "watermark_type": "aaronson",
  "watermark_detection": "Aaronson score: 157.23, Length: 200, Normalized: 0.7862"
}
```

## Parameters

### For All Watermarking Methods:
- `watermark_type`: 'green_list' or 'aaronson' (default: 'green_list')
- `watermark_steps`: Maximum step to watermark at, or None for all steps

### For Green List Method:
- `gamma`: Fraction of tokens that are "green" (default: 0.5)
- `amplification`: Amplification factor for green tokens (default: 0.0)

### For Aaronson Method:
- `aaronson_seed`: Seed for pseudorandom function (default: 42)

## Detection Metrics

### Green List Method:
- **Green Token Matches**: Number and percentage of tokens in the green list
- **Z-score**: Statistical significance of the watermark

### Aaronson Method:
- **Aaronson Score**: Sum of ln(1/(1-r)) for all tokens
- **Length**: Number of tokens analyzed
- **Normalized Score**: Aaronson score divided by length

A higher normalized score (typically > 0.7) indicates watermarked text.

## Technical Notes

1. **Numerical Stability**: The implementation uses log-space computations to avoid numerical issues with very small probabilities.

2. **Position-Based Seeding**: The pseudorandom function uses position-based seeding to ensure deterministic watermarking and detection.

3. **Token Selection**: Unlike the green list method which biases probabilities, Aaronson's method directly selects tokens that maximize r^{1/p}, which is implemented by computing log(r)/p and selecting the argmax.

4. **Detection**: The detection formula ln(1/(1-r)) grows as r approaches 1, so watermarked text will have higher scores.

## References

- Aaronson, S. (2002). "Quantum computing, postselection, and probabilistic polynomial-time"
- The watermarking scheme is based on the Gumbel softmax distribution

## Files Modified

1. `generate.py`: Added Aaronson watermarking functions and modified generation
2. `eval_llada.py`: Added support for Aaronson watermarking in evaluation





