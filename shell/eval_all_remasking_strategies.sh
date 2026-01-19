#!/bin/bash
#SBATCH --job-name=aaronson_all_strategies      # Job name
#SBATCH --output=aaronson_all_strategies.log   # Output log file
#SBATCH --error=error_aaronson_all_strategies.log  # Error log file
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

# Common parameters
WATERMARK_TYPE="aaronson"
AARONSON_SEED=42
WATERMARK_STEPS=100

echo "================================================================================"
echo "TESTING ALL AARONSON REMASKING STRATEGIES"
echo "================================================================================"
echo ""

# ================================================================================
# STRATEGY 1: ORIGINAL (Best Quality)
# ================================================================================
echo "================================================================================"
echo "STRATEGY 1: ORIGINAL (Best Quality)"
echo "================================================================================"
MODEL_ARGS="model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024,watermark_type=$WATERMARK_TYPE,aaronson_seed=$AARONSON_SEED,watermark_steps=$WATERMARK_STEPS,aaronson_remasking_strategy=original"

echo "Running GSM8K with original strategy (1 prompt test)..."
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args ${MODEL_ARGS},max_prompts=1

echo ""
echo "Strategy 1 (original) completed!"
echo ""

# ================================================================================
# STRATEGY 2: DUAL_GATE (Balanced)
# ================================================================================
echo "================================================================================"
echo "STRATEGY 2: DUAL_GATE (Balanced)"
echo "================================================================================"
TAU_WM=0.2
TAU_ORIG=0.01
MODEL_ARGS="model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024,watermark_type=$WATERMARK_TYPE,aaronson_seed=$AARONSON_SEED,watermark_steps=$WATERMARK_STEPS,aaronson_remasking_strategy=dual_gate,aaronson_tau_wm=$TAU_WM,aaronson_tau_orig=$TAU_ORIG"

echo "Running GSM8K with dual_gate strategy (tau_wm=$TAU_WM, tau_orig=$TAU_ORIG)..."
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args ${MODEL_ARGS},max_prompts=1

echo ""
echo "Strategy 2 (dual_gate) completed!"
echo ""

# ================================================================================
# STRATEGY 3: BLEND with lambda=0.7 (Strong Detectability)
# ================================================================================
echo "================================================================================"
echo "STRATEGY 3: BLEND with lambda=0.7 (Strong Detectability)"
echo "================================================================================"
LAMBDA=0.7
MODEL_ARGS="model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024,watermark_type=$WATERMARK_TYPE,aaronson_seed=$AARONSON_SEED,watermark_steps=$WATERMARK_STEPS,aaronson_remasking_strategy=blend,aaronson_lambda=$LAMBDA"

echo "Running GSM8K with blend strategy (lambda=$LAMBDA)..."
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args ${MODEL_ARGS},max_prompts=1

echo ""
echo "Strategy 3 (blend lambda=0.7) completed!"
echo ""

# ================================================================================
# STRATEGY 4: HARD_FAVOR (Maximum Detectability)
# ================================================================================
echo "================================================================================"
echo "STRATEGY 4: HARD_FAVOR (Maximum Detectability - Risky)"
echo "================================================================================"
MODEL_ARGS="model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024,watermark_type=$WATERMARK_TYPE,aaronson_seed=$AARONSON_SEED,watermark_steps=$WATERMARK_STEPS,aaronson_remasking_strategy=hard_favor"

echo "Running GSM8K with hard_favor strategy..."
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args ${MODEL_ARGS},max_prompts=1

echo ""
echo "Strategy 4 (hard_favor) completed!"
echo ""

# ================================================================================
# SUMMARY
# ================================================================================
echo "================================================================================"
echo "ALL REMASKING STRATEGIES TESTED!"
echo "================================================================================"
echo ""
echo "Check the log files for results from each strategy:"
echo "  1. original    - Best quality, moderate detectability"
echo "  2. dual_gate   - Balanced quality and detectability"
echo "  3. blend       - Configurable (lambda=0.7 used)"
echo "  4. hard_favor  - Maximum detectability, quality risk"
echo ""
echo "Compare the watermark scores and generation quality across strategies."
echo "================================================================================"

