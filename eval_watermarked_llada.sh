#!/bin/bash
#SBATCH --job-name=llada_job       # Job name
#SBATCH --output=output.txt            # Output log file
#SBATCH --error=error.txt              # Error log file
#SBATCH --partition=ghx4               # Match the interactive partition
#SBATCH --account=bemc-dtai-gh         # Your Slurm account
#SBATCH --gres=gpu:h100:1 
#SBATCH --cpus-per-gpu=72              # 72 CPUs per GPU (like interactive)
#SBATCH --mem=0                        # Let Slurm auto-assign full memory
#SBATCH --time=24:00:00                # Time limit (48 hours)
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks=1                     # Single task


module load cuda/12.2.0

source /work/nvme/bemc/python_envs/llada_env/bin/activate

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Watermarking parameters
GAMMA=0.1
AMPLIFICATION=5
WATERMARK_STEPS=100

echo "Starting watermarked LLaDA model evaluation..."
echo "Watermarking parameters: gamma=$GAMMA, amplification=$AMPLIFICATION, watermark_steps=$WATERMARK_STEPS"

# conditional likelihood estimation benchmarks with watermarking
# echo "Running GPQA with watermarking..."
# accelerate launch eval_llada.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running TruthfulQA with watermarking..."
# accelerate launch eval_llada.py --tasks truthfulqa_mc2 --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running ARC Challenge with watermarking..."
# accelerate launch eval_llada.py --tasks arc_challenge --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

echo "Running HellaSwag with watermarking..."
accelerate launch eval_llada.py --tasks hellaswag --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running WinoGrande with watermarking..."
# accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running PIQA with watermarking..."
# accelerate launch eval_llada.py --tasks piqa --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=128,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

echo "Running MMLU with watermarking..."
accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running CMMLU with watermarking..."
# accelerate launch eval_llada.py --tasks cmmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running C-Eval with watermarking..."
# accelerate launch eval_llada.py --tasks ceval-valid --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=1,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# # conditional generation benchmarks with watermarking
# echo "Running BBH with watermarking..."
# accelerate launch eval_llada.py --tasks bbh --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

echo "Running GSM8K with watermarking..."
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running Minerva Math with watermarking..."
# accelerate launch eval_llada.py --tasks minerva_math --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

echo "Running HumanEval with watermarking..."
accelerate launch eval_llada.py --tasks humaneval --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Running MBPP with watermarking..."
# accelerate launch eval_llada.py --tasks mbpp --confirm_run_unsafe_code --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024,gamma=$GAMMA,amplification=$AMPLIFICATION,watermark_steps=$WATERMARK_STEPS

# echo "Watermarked evaluation completed!"
