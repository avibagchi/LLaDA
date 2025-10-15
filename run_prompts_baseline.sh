#!/bin/bash
#SBATCH --job-name=prompts_baseline      # Job name
#SBATCH --output=prompts_baseline.log   # Output log file
#SBATCH --error=error_prompts_baseline.log  # Error log file
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

# Parameters
PROMPTS_FILE="prompts.txt"
OUTPUT_FILE="prompts_baseline_outputs.json"
MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
GEN_LENGTH=512
STEPS=512
BLOCK_LENGTH=512
TEMPERATURE=1.0  # Standard sampling temperature (fair comparison to Aaronson)
MAX_PROMPTS=50   # Limit for testing (set to None or remove for all prompts)

# NO watermarking for baseline
WATERMARK_TYPE="none"

echo "Starting batch generation WITHOUT watermarking (baseline)..."
echo "Prompts file: $PROMPTS_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Max prompts: $MAX_PROMPTS"
echo "Temperature: $TEMPERATURE"
echo "Watermark type: $WATERMARK_TYPE"
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
  --max_prompts $MAX_PROMPTS"

echo "Running command:"
echo "$CMD"
echo ""

# Run the command
eval $CMD

echo ""
echo "Batch generation (baseline) completed!"
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "The output includes:"
echo "  - Generated text for each prompt"
echo "  - GPT-2 perplexity score (calculated on CPU, external judge)"
echo "  - Average perplexity across all prompts"
echo ""
echo "Lower perplexity = more natural text (better quality)."
echo "Compare with watermarked perplexity to assess quality impact."

