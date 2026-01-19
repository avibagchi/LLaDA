# Llama Model Evaluation with HuggingFace Watermarking

This directory contains evaluation scripts for Llama models using HuggingFace Transformers' built-in `WatermarkingConfig` and `WatermarkDetector`.

## Overview

The watermarking implementation uses HuggingFace's official watermarking utilities to embed a statistical watermark into generated text by biasing the model's logits during generation. The watermark can be detected using the `WatermarkDetector` class.

## Files

- `eval_llama_synthid.py`: Main evaluation script with watermarking support using `WatermarkingConfig` and `WatermarkDetector`
- `eval_llama_synthid.sh`: Full benchmark evaluation script (SLURM job)
- `README_SYNTHID.md`: This documentation file

## How HuggingFace Watermarking Works

### Watermark Embedding

The watermarking uses HuggingFace's `WatermarkingConfig` which implements the approach from "A Watermark for Large Language Models" (Kirchenbauer et al.):

1. **Green List Generation**: At each generation step, a hash function (seeded by the context) divides the vocabulary into "green" and "red" tokens.

2. **Logit Biasing**: Green tokens receive a positive bias added to their logits, making them more likely to be selected.

3. **Seeding Schemes**:
   - `"selfhash"`: Uses the current token to seed the green list for the next token
   - `"lefthash"`: Uses the previous token(s) to seed the green list

4. **Parameter Control**:
   - `bias`: Watermark strength - bias added to green tokens (default: 2.5)
   - `hashing_key`: Random key for hashing (default: 0)
   - `seeding_scheme`: Which seeding method to use

### Watermark Detection

The `WatermarkDetector` automatically detects watermarks:
1. Recreates the green lists using the same configuration
2. Computes detection statistics
3. Returns a prediction (True/False) and optional scores

**Detection Output**:
- `prediction`: Boolean indicating if watermark is detected
- `score`: Detection confidence score (if available)
- `z_score`: Statistical significance of watermark (if available)

## Installation

Ensure you have the required packages:

```bash
pip install transformers accelerate lm-eval
```

Make sure you have transformers version 4.40.0 or later for `WatermarkingConfig` and `WatermarkDetector` support.

## Usage

### Quick Test (Limited Prompts)

To run a quick test with a few prompts:

```bash
sbatch test_llama_synthid.sh
```

This will evaluate:
- GSM8K (5 prompts)
- TruthfulQA MC2 (3 prompts)

### Full Evaluation

To run the full benchmark suite:

```bash
sbatch eval_llama_synthid.sh
```

This evaluates multiple benchmarks including:

**Conditional Likelihood Tasks** (Multiple Choice):
- GPQA
- TruthfulQA MC2
- ARC Challenge
- HellaSwag
- WinoGrande
- PIQA
- MMLU

**Generation Tasks**:
- TruthfulQA Generation
- BBH (Big-Bench Hard)
- GSM8K (Math)
- Minerva Math
- HumanEval (Code)
- MBPP (Code)

### Customizing Parameters

Edit the script to modify watermarking parameters:

```bash
# In eval_llama_synthid.sh or test_llama_synthid.sh

# Enable/disable watermarking
USE_WATERMARK=True  # or False

# Watermark strength
GAMMA=0.5    # Higher = more tokens marked green (0.25-0.75 typical)
DELTA=2.0    # Higher = stronger watermark, more quality impact (1.0-5.0 typical)

# Watermark key (for reproducibility)
WATERMARK_KEY=42

# Model selection
MODEL_PATH="meta-llama/Llama-2-7b-hf"
# Or use: "meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B", etc.
```

### Running Specific Benchmarks

To run a specific benchmark:

```bash
accelerate launch eval_llama_synthid.py \
    --tasks gsm8k \
    --model llama_synthid \
    --batch_size 8 \
    --model_args model_path=meta-llama/Llama-2-7b-hf,use_watermark=True,gamma=0.5,delta=2.0,watermark_key=42
```

### Testing with Limited Prompts

You can limit the number of prompts for quick testing:

```bash
accelerate launch eval_llama_synthid.py \
    --tasks gsm8k \
    --model llama_synthid \
    --batch_size 8 \
    --model_args model_path=meta-llama/Llama-2-7b-hf,use_watermark=True,gamma=0.5,delta=2.0,max_prompts=10
```

## Model Arguments

Available model arguments (comma-separated in `model_args`):

- `model_path`: HuggingFace model identifier (required)
- `use_watermark`: Enable/disable watermarking (default: True)
- `gamma`: Green list fraction 0-1 (default: 0.5)
- `delta`: Watermark bias strength (default: 2.0)
- `watermark_key`: Random seed for watermark (default: 0)
- `max_prompts`: Limit number of prompts for testing (default: None = all)
- `batch_size`: Batch size for evaluation (default: 8)
- `max_length`: Maximum sequence length (default: 4096)

## Output

The evaluation script produces:

1. **Per-Prompt Statistics** (for generation tasks):
   - Question text
   - Generated answer
   - Green token count and fraction
   - Z-score (watermark strength indicator)
   - P-value (statistical significance)

2. **Final Metrics**:
   - Task accuracy/performance
   - Overall benchmark scores

Example output:
```
=== PROMPT 1 ===
Question: What is 2 + 2?
Generated: The answer is 4...
Green tokens: 52/100 (52.00%)
Z-score: 8.45
P-value: 0.0001
==================================================
```

## Interpreting Results

### Watermark Strength

- **Z-score > 4**: Strong watermark detected
- **Z-score 2-4**: Moderate watermark
- **Z-score < 2**: Weak/no watermark
- **Green fraction ≈ gamma**: Expected for watermarked text

### Quality vs. Watermark Tradeoff

- **Higher delta**: Stronger watermark but potentially lower quality
- **gamma ≈ 0.5**: Good balance (half vocabulary marked green)
- **Lower delta**: Less impact on quality but weaker watermark

## Comparison with LLaDA Watermarking

This SynthID implementation differs from the LLaDA watermarking:

| Feature | LLaDA | SynthID (This Implementation) |
|---------|-------|-------------------------------|
| Method | Custom diffusion-based | Logit biasing |
| Model | LLaDA-specific | Any autoregressive LLM |
| Detection | Green token matching | Statistical z-score |
| Integration | Built into model | External logits processor |

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
--batch_size 4  # or lower
```

Or use a smaller model variant.

### Model Access Issues

Some Llama models require authentication:
```bash
huggingface-cli login
```

Then accept the model's terms on HuggingFace Hub.

### Import Errors

Ensure scipy is installed:
```bash
pip install scipy
```

## References

- SynthID Text Watermarking: [Google DeepMind Blog](https://deepmind.google/discover/blog/watermarking-ai-generated-text/)
- HuggingFace Transformers: [Documentation](https://huggingface.co/docs/transformers/)
- LM Evaluation Harness: [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)

## Notes

- The SynthID implementation uses a simplified version of the watermarking approach
- For production use, consider using Google's official SynthID implementation if available
- Watermarking may slightly impact generation quality - test with different delta values
- Detection works best with longer generated sequences (>50 tokens)

