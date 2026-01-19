#!/bin/bash
#SBATCH --job-name=new_bbh_aaronson_watermark      # Job name
#SBATCH --output=new_bbh_aaronson_watermarked.log   # Output log file
#SBATCH --error=new_bbh_error_aaronson_watermarked.log  # Error log file
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
WATERMARK_STEPS=2000

# Remasking strategy parameters
# Options: original, dual_gate, blend, hard_favor
REMASKING_STRATEGY="original"
Temperature=1
TAU_WM=0.2          # For dual_gate
TAU_ORIG=0.01       # For dual_gate
LAMBDA=0.7          # For blend

echo "Starting Aaronson watermarked LLaDA model evaluation..."
echo "Watermarking parameters:"
echo "  watermark_type=$WATERMARK_TYPE"
echo "  aaronson_seed=$AARONSON_SEED"
echo "  watermark_steps=$WATERMARK_STEPS"
echo "  remasking_strategy=$REMASKING_STRATEGY"
if [ "$REMASKING_STRATEGY" = "dual_gate" ]; then
    echo "  tau_wm=$TAU_WM"
    echo "  tau_orig=$TAU_ORIG"
elif [ "$REMASKING_STRATEGY" = "blend" ]; then
    echo "  lambda=$LAMBDA"
fi
echo ""

# Build model args based on remasking strategy
# temperature=0 for watermarked run (all positions will be watermarked anyway)
MODEL_ARGS="model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024,temperature=$Temperature,watermark_type=$WATERMARK_TYPE,aaronson_seed=$AARONSON_SEED,watermark_steps=$WATERMARK_STEPS,aaronson_remasking_strategy=$REMASKING_STRATEGY"

if [ "$REMASKING_STRATEGY" = "dual_gate" ]; then
    MODEL_ARGS="${MODEL_ARGS},aaronson_tau_wm=$TAU_WM,aaronson_tau_orig=$TAU_ORIG"
elif [ "$REMASKING_STRATEGY" = "blend" ]; then
    MODEL_ARGS="${MODEL_ARGS},aaronson_lambda=$LAMBDA"
fi

# Test with a single prompt for quick verification
echo "Running open-ended generation benchmarks with Aaronson watermarking (test run with 5 prompts)..."
echo ""

# echo "1. GSM8K (Math word problems)..."
# accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args ${MODEL_ARGS},max_prompts=100

# echo ""
echo "2. BBH (Big Bench Hard)..."
accelerate launch eval_llada.py --tasks bbh --model llada_dist --model_args ${MODEL_ARGS},max_prompts=3

# echo ""
# echo "3. Minerva Math (Advanced math problems)..."
# accelerate launch eval_llada.py --tasks minerva_math --model llada_dist --model_args ${MODEL_ARGS},max_prompts=1

# echo ""
# echo "4. HumanEval (Code generation)..."
# accelerate launch eval_llada.py --tasks humaneval --model llada_dist --confirm_run_unsafe_code --model_args ${MODEL_ARGS},max_prompts=100

# echo ""
# echo "5. MBPP (Python code generation)..."
# accelerate launch eval_llada.py --tasks mbpp --model llada_dist --confirm_run_unsafe_code --model_args ${MODEL_ARGS},max_prompts=5

# Uncomment below for full evaluation
# echo ""
# echo "Running full evaluation on all open-ended benchmarks..."
# accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args ${MODEL_ARGS}
# accelerate launch eval_llada.py --tasks bbh --model llada_dist --model_args ${MODEL_ARGS}
# accelerate launch eval_llada.py --tasks minerva_math --model llada_dist --model_args ${MODEL_ARGS}
# accelerate launch eval_llada.py --tasks humaneval --model llada_dist --confirm_run_unsafe_code --model_args ${MODEL_ARGS}
# accelerate launch eval_llada.py --tasks mbpp --model llada_dist --confirm_run_unsafe_code --model_args ${MODEL_ARGS}

echo ""
echo "Aaronson watermarked evaluation completed!"

