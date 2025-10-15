#!/bin/bash
#SBATCH --job-name=prompts_no_watermarked      # Job name
#SBATCH --output=prompts_no_watermarked.log   # Output log file
#SBATCH --error=no_water_error_prompts_watermarked.log  # Error log file
#SBATCH --partition=gpuA100x4         
#SBATCH --account=bemc-delta-gpu         # Your valid Slurm account
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # One task
#SBATCH --cpus-per-task=16             # 16 cores per GPU
#SBATCH --mem=96G                      # Memory for the job
#SBATCH --time=24:00:00                # Time limit

# Load correct CUDA
module load gcc/11.4.0
module load cuda/12.3.0
module load cray-python/3.11.5

# Activate your Python environment
source /work/nvme/bemc/python_envs/llada_env_5/bin/activate

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Set CUDA_LAUNCH_BLOCKING to help with debugging CUDA errors
# export CUDA_LAUNCH_BLOCKING=1  # Uncomment if you see CUDA assertion errors

# Parameters
PROMPTS_FILE="prompts.txt"
OUTPUT_FILE="prompts_no_watermarked_outputs.json"
MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
GEN_LENGTH=512
STEPS=512
BLOCK_LENGTH=512
TEMPERATURE=1.0  # Standard sampling temperature
MAX_PROMPTS=500   # Limit for testing (set to None or remove for all prompts)

# Aaronson watermarking parameters
WATERMARK_TYPE="aaronson"
AARONSON_SEED=42
WATERMARK_STEPS=0  # Empty = all steps
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

# Add watermark_steps if specified
if [ -n "$WATERMARK_STEPS" ]; then
    CMD="$CMD --watermark_steps $WATERMARK_STEPS"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Run the command
eval $CMD

echo ""
echo "Batch generation with watermarking completed!"
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

