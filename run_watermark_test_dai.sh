#!/bin/bash
#SBATCH --job-name=testing_nowatermark_job      # Job name
#SBATCH --output=output_nowatermark.log            # Output log file
#SBATCH --error=error_nowatermark.log             # Error log file
#SBATCH --partition=ghx4         
#SBATCH --account=bemc-dtai-gh         # Your valid Slurm account
#SBATCH --gres=gpu:h100:1                   # Request 2 GPUs
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # One task (you can adjust for multi-GPU)
#SBATCH --cpus-per-task=16             # 16 cores per GPU is safe
#SBATCH --mem=96G                      # Memory for the job
#SBATCH --time=24:00:00                # Time limit

# Load correct CUDA for H200
# module purge
module load cuda/12.2.0 

# Activate your Python environment
source /work/nvme/bemc/python_envs/sedd_env/bin/activate


python test_watermark_metrics.py


