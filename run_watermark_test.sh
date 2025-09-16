#!/bin/bash
#SBATCH --job-name=testing_watermark_job      # Job name
#SBATCH --output=output.log            # Output log file
#SBATCH --error=error.log             # Error log file
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
module load python/3.10.13
module load cuda/12.3.0

# Activate your Python environment
source /work/nvme/bemc/python_envs/sedd_env_3/bin/activate


python test_watermark_metrics.py


