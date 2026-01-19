#!/bin/bash
#SBATCH --job-name=testing_nowatermark_job      # Job name
#SBATCH --output=llada_watermarked.log            # Output log file
#SBATCH --error=error_llada_watermarked.log             # Error log file
#SBATCH --partition=gpuA100x4         
#SBATCH --account=bemc-delta-gpu         # Your valid Slurm account
#SBATCH --gres=gpu:1                   # Request 2 GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # One task (you can adjust for multi-GPU)
#SBATCH --cpus-per-task=16             # 16 cores per GPU is safe
#SBATCH --mem=96G                      # Memory for the job
#SBATCH --time=24:00:00                # Time limit

module load gcc/11.4.0
module load cuda/12.3.0
module load cray-python/3.11.5
source /work/nvme/bemc/python_envs/llada_env_5/bin/activate
cd /work/nvme/bemc/abagchi2/LLaDA

echo "Quick test with 3 prompts..."
accelerate launch eval_llada.py \
    --tasks truthfulqa_gen \
    --num_fewshot 0 \
    --model llada_dist \
    --batch_size 1 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128,gamma=0.025,amplification=4.5,watermark_steps=50,max_prompts=3
