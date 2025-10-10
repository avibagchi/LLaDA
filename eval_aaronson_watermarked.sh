#!/bin/bash
#SBATCH --job-name=aaronson_watermark      # Job name
#SBATCH --output=aaronson_watermarked.log   # Output log file
#SBATCH --error=error_aaronson_watermarked.log  # Error log file
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

# Aaronson watermarking parameters
WATERMARK_TYPE="aaronson"
AARONSON_SEED=42
WATERMARK_STEPS=100

echo "Starting Aaronson watermarked LLaDA model evaluation..."
echo "Watermarking parameters: watermark_type=$WATERMARK_TYPE, aaronson_seed=$AARONSON_SEED, watermark_steps=$WATERMARK_STEPS"

# Test with a single prompt for quick verification
echo "Running GSM8K with Aaronson watermarking (test run with 1 prompt)..."
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,watermark_type=$WATERMARK_TYPE,aaronson_seed=$AARONSON_SEED,watermark_steps=$WATERMARK_STEPS,max_prompts=1

# Uncomment below for full evaluation
# echo "Running GSM8K with Aaronson watermarking (full evaluation)..."
# accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,watermark_type=$WATERMARK_TYPE,aaronson_seed=$AARONSON_SEED,watermark_steps=$WATERMARK_STEPS

echo "Aaronson watermarked evaluation completed!"

