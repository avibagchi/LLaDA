#!/bin/bash
# Interactive version (no SLURM directives) - run directly with bash

# Load correct CUDA (try to load, but ignore errors if modules don't exist)
module load gcc/11.4.0 2>/dev/null || true
module load cuda/12.3.0 2>/dev/null || true
module load cray-python/3.11.5 2>/dev/null || true

# Activate your Python environment (matching old-llada)
source /work/nvme/bemc/python_envs/llada_env_5/bin/activate

# Verify torch is available
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch (torch) is not available in the Python environment."
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Change to LLaDA directory
cd /work/nvme/bemc/abagchi2/LLaDA

# Parameters (matching old-llada exactly)
PROMPTS_FILE="prompts.txt"
OUTPUT_FILE="prompts_aaronson_watermarked_outputs.json"
MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
GEN_LENGTH=512
STEPS=512
BLOCK_LENGTH=512
TEMPERATURE=1.0  # Standard sampling temperature (matching old-llada)
MAX_PROMPTS=10   # Limit for testing (set to None or remove for all prompts)

# Aaronson watermarking parameters (matching old-llada exactly)
WATERMARK_TYPE="aaronson"
AARONSON_SEED=42
WATERMARK_STEPS=0  # 0 = all steps (matching old-llada behavior)
REMASKING_STRATEGY="original"
TAU_WM=0.2
TAU_ORIG=0.01
LAMBDA=0.7

echo "Starting batch generation with Aaronson watermarking..."
echo "Prompts file: $PROMPTS_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Max prompts: $MAX_PROMPTS"
echo "Temperature: $TEMPERATURE"
echo "Watermark type: $WATERMARK_TYPE"
echo "Remasking strategy: $REMASKING_STRATEGY"
echo "Watermark steps: $WATERMARK_STEPS (0 = all steps)"
echo ""
echo "Note: Perplexity will be calculated using GPT-2 on CPU (external judge)."
echo "      This avoids CUDA errors from vocabulary mismatches."
echo ""

# Build command
CMD="python batch_generate.py \
  --prompts_file $PROMPTS_FILE \
  --output_file $OUTPUT_FILE \
  --model_path $MODEL_PATH \
  --gen_length $GEN_LENGTH \
  --steps $STEPS \
  --block_length $BLOCK_LENGTH \
  --temperature $TEMPERATURE \
  --watermark_type $WATERMARK_TYPE \
  --aaronson_seed $AARONSON_SEED \
  --aaronson_remasking_strategy $REMASKING_STRATEGY \
  --max_prompts $MAX_PROMPTS"

# Add remasking parameters if needed
if [ "$REMASKING_STRATEGY" = "dual_gate" ]; then
    CMD="$CMD --aaronson_tau_wm $TAU_WM --aaronson_tau_orig $TAU_ORIG"
elif [ "$REMASKING_STRATEGY" = "blend" ]; then
    CMD="$CMD --aaronson_lambda $LAMBDA"
fi

# Add watermark_steps if specified (0 or empty means all steps = None in Python)
# When WATERMARK_STEPS=0, we don't pass the argument (defaults to None = all steps)
if [ -n "$WATERMARK_STEPS" ] && [ "$WATERMARK_STEPS" != "0" ] && [ "$WATERMARK_STEPS" != "" ]; then
    CMD="$CMD --watermark_steps $WATERMARK_STEPS"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Run the command
eval $CMD

echo ""
echo "Batch generation with Aaronson watermarking completed!"
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "The output includes:"
echo "  - Generated text for each prompt"
echo "  - GPT-2 perplexity score (calculated on CPU, external judge)"
echo "  - Average perplexity across all prompts"
echo "  - Watermark detection scores"
echo ""
echo "Lower perplexity = more natural text (better quality)."
echo "Compare with baseline perplexity to assess quality impact of watermarking."
