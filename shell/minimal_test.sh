#!/bin/bash
#SBATCH --job-name=minimal_test
#SBATCH --output=minimal_output.log
#SBATCH --error=minimal_error.log
#SBATCH --partition=gpuA100x4
#SBATCH --account=bemc-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1:00:00

module load gcc/11.4.0
module load cuda/12.3.0
module load cray-python/3.11.5
source /work/nvme/bemc/python_envs/llada_env_5/bin/activate
cd /work/nvme/bemc/abagchi2/LLaDA

echo "Testing with just 1 prompt..."

accelerate launch eval_llada.py \
    --tasks truthfulqa_gen \
    --num_fewshot 0 \
    --model llada_dist \
    --batch_size 1 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128,gamma=0.025,amplification=4.5,watermark_steps=50,max_prompts=1

echo "Test completed!"
