#!/bin/bash
#SBATCH --job-name=llada_test_few
#SBATCH --output=test_few_output.log
#SBATCH --error=test_few_error.log
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Load modules
module load gcc/11.4.0
module load cuda/12.3.0
module load cray-python/3.11.5

# Activate environment
source /work/nvme/bemc/python_envs/llada_env_5/bin/activate

# Change to LLaDA directory
cd /work/nvme/bemc/abagchi2/LLaDA

echo "Testing with first 5 prompts only..."

# Test with only 5 prompts
accelerate launch eval_llada.py \
    --tasks truthfulqa_gen \
    --num_fewshot 0 \
    --model llada_dist \
    --batch_size 1 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=2.0,is_check_greedy=False,mc_num=128,gamma=0.025,amplification=4.5,watermark_steps=50,max_prompts=5

echo "Test completed!"
