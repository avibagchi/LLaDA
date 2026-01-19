# Aaronson Watermarking - Quick Reference Card

## Quick Start

### 1. Use Aaronson Watermarking (Recommended for Testing)
```bash
accelerate launch eval_llada.py \
    --tasks gsm8k \
    --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',\
watermark_type=aaronson,\
aaronson_seed=42,\
gen_length=1024,\
steps=1024,\
block_length=1024,\
max_prompts=10
```

### 2. Use Green List Watermarking (Original Method)
```bash
accelerate launch eval_llada.py \
    --tasks gsm8k \
    --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',\
watermark_type=green_list,\
gamma=0.025,\
amplification=0,\
gen_length=1024,\
steps=1024,\
block_length=1024
```

## Key Differences

| Feature | Green List | Aaronson |
|---------|-----------|----------|
| Method | Bias green token probabilities | Maximize r^(1/p) |
| Detection | Count green tokens | Sum of log scores |
| Parameters | gamma, amplification | aaronson_seed |
| Detection Metric | Z-score | Normalized score |
| Typical Threshold | Z > 3 | Normalized > 0.7 |

## Parameter Quick Guide

### Common Parameters
- `watermark_type`: `'green_list'` or `'aaronson'`
- `watermark_steps`: `None` (all steps), `200` (first 200), or `[10,20,30]` (specific steps)
- `max_prompts`: `10` for testing, `None` for full evaluation

### Green List Parameters
- `gamma`: `0.025` to `0.5` (fraction of green tokens)
- `amplification`: `0.0` to `2.0` (bias strength)

### Aaronson Parameters
- `aaronson_seed`: `42` (any integer for different watermarks)

## Understanding Detection Scores

### Green List
```
Green token matches: 45/200 (22.50%), Z-score: 3.45
```
- 45 out of 200 tokens were in green list
- Z-score of 3.45 indicates strong watermark (> 3 is significant)

### Aaronson
```
Aaronson score: 157.23, Length: 200, Normalized: 0.7862
```
- Raw score: 157.23 (sum of log scores)
- Sequence length: 200 tokens
- Normalized: 0.7862 (score/length, > 0.7 indicates watermark)

## Testing Your Setup

### Run Unit Tests
```bash
cd /work/nvme/bemc/abagchi2/LLaDA
python test_aaronson_watermark.py
```

### Quick Test with 10 Prompts
```bash
sbatch eval_aaronson_watermarked.sh
# or
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist \
    --model_args watermark_type=aaronson,max_prompts=10
```

## Common Issues & Solutions

### Issue: No watermark detected
- **Check**: Is `watermark_type` set correctly?
- **Check**: Are you using the same seed for generation and detection?
- **Try**: Increase `watermark_steps` to apply watermarking to more steps

### Issue: Import errors
- **Solution**: Make sure you're in the LLaDA directory
- **Solution**: Check that dependencies are installed in your environment

### Issue: Memory errors
- **Solution**: Reduce `batch_size` parameter
- **Solution**: Reduce `max_prompts` for testing

## Interpreting Results

### Watermarked Text (Aaronson)
- Normalized score > 0.7
- Tokens chosen consistently maximize r^(1/p)
- Detection is reliable even with short sequences

### Non-Watermarked Text (Aaronson)
- Normalized score ≈ 0.5 (expected for uniform random)
- Detection score varies randomly
- Cannot distinguish from random text

### Watermarked Text (Green List)
- Z-score > 3 (statistically significant)
- Green token percentage >> gamma
- Consistent across different random seeds

## File Locations

- **Implementation**: `generate.py` (lines 195-320)
- **Evaluation**: `eval_llada.py`
- **Example Script**: `eval_aaronson_watermarked.sh`
- **Tests**: `test_aaronson_watermark.py`
- **Documentation**: `AARONSON_WATERMARKING_README.md`

## Need Help?

1. Check `AARONSON_WATERMARKING_README.md` for detailed usage
2. Check `IMPLEMENTATION_SUMMARY.md` for technical details
3. Run `test_aaronson_watermark.py` to verify setup
4. Check your output JSON file for watermark_detection field

## Example Output

```json
{
  "prompt_number": 1,
  "question": "What is 2 + 2?",
  "answer": "4",
  "watermark_detection": "Aaronson score: 12.34, Length: 15, Normalized: 0.8227",
  "watermark_type": "aaronson"
}
```

## Advanced Usage

### Watermark Only First N Steps
```bash
--model_args watermark_type=aaronson,watermark_steps=100
```

### Use Different Seed
```bash
--model_args watermark_type=aaronson,aaronson_seed=12345
```

### Compare Methods on Same Task
```bash
# Run both and compare results
--model_args watermark_type=green_list,gamma=0.5
--model_args watermark_type=aaronson,aaronson_seed=42
```






