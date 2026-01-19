#!/bin/bash
#SBATCH --job-name=no_llada_testing_nowatermark_job      # Job name
#SBATCH --output=no_llada_watermarked.log            # Output log file
#SBATCH --error=no_error_llada_watermarked.log             # Error log file
#SBATCH --partition=gpuA100x4         
#SBATCH --account=bemc-delta-gpu         # Your valid Slurm account
#SBATCH --gres=gpu:1                   # Request 2 GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # One task (you can adjust for multi-GPU)
#SBATCH --cpus-per-task=16             # 16 cores per GPU is safe
#SBATCH --mem=96G                      # Memory for the job
#SBATCH --time=24:00:00                # Time limit

# Load correct CUDA for H200
# module purge
module load gcc/11.4.0
module load cuda/12.3.0
module load cray-python/3.11.5

# Activate your Python environment
source /work/nvme/bemc/python_envs/llada_env_5/bin/activate

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Watermarking parameters
GAMMA=0.025 # 0.025
AMPLIFICATION=0
WATERMARK_STEPS=200

# echo "Starting watermarked LLaDA model evaluation..."
# echo "Watermarking parameters: gamma=$GAMMA, amplification=$AMPLIFICATION, watermark_steps=$WATERMARK_STEPS"

# conditional likelihood estimation benchmarks with watermarking
# echo "Running GPQA with watermarking..."
# accelerate launch eval_llada.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running TruthfulQA with watermarking..."
# accelerate launch eval_llada.py --tasks truthfulqa_mc2 --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS,max_prompts=1

# echo "Running TruthfulQA with watermarking..."
# accelerate launch eval_llada.py --tasks truthfulqa_gen --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS,max_prompts=3

# echo "Running ARC Challenge with watermarking..."
# accelerate launch eval_llada.py --tasks arc_challenge --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running HellaSwag with watermarking..."
# accelerate launch eval_llada.py --tasks hellaswag --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running WinoGrande with watermarking..."
# accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running PIQA with watermarking..."
# accelerate launch eval_llada.py --tasks piqa --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running MMLU with watermarking..."
# accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running CMMLU with watermarking..."
# accelerate launch eval_llada.py --tasks cmmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running C-Eval with watermarking..."
# accelerate launch eval_llada.py --tasks ceval-valid --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# # conditional generation benchmarks with watermarking
echo "Running BBH with watermarking with $AMPLIFICATION amplification..."
accelerate launch eval_llada.py --tasks bbh --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS,max_prompts=100

# echo "Running GSM8K with watermarking..."
# accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS,max_prompts=100

# echo "Running Minerva Math with watermarking..."
# accelerate launch eval_llada.py --tasks minerva_math --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running HumanEval with watermarking..."
# accelerate launch eval_llada.py --tasks humaneval --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running MBPP with watermarking..."
# accelerate launch eval_llada.py --tasks mbpp --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Watermarked evaluation completed!"
