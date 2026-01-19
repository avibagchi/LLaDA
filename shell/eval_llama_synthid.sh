#!/bin/bash
#SBATCH --job-name=llama_synthid_watermark      # Job name
#SBATCH --output=llama_synthid.log              # Output log file
#SBATCH --error=error_llama_synthid.log         # Error log file
#SBATCH --partition=gpuA100x4         
#SBATCH --account=bemc-delta-gpu                # Your valid Slurm account
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --nodes=1                               # Request 1 node
#SBATCH --ntasks=1                              # One task
#SBATCH --cpus-per-task=16                      # 16 cores per GPU
#SBATCH --mem=96G                               # Memory for the job
#SBATCH --time=24:00:00                         # Time limit

# Load modules
module load gcc/11.4.0
module load cuda/12.3.0
module load cray-python/3.11.5

# Activate Python environment
source /work/nvme/bemc/python_envs/llada_env_5/bin/activate

# Load HuggingFace token from file
if [ -f "hf_token.txt" ]; then
    export HUGGINGFACE_HUB_TOKEN=$(cat hf_token.txt)
    echo "Loaded HuggingFace token from hf_token.txt"
else
    echo "Error: hf_token.txt not found. Please create it with your token."
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Watermarking parameters (HuggingFace WatermarkingConfig)
USE_WATERMARK=True
BIAS=2                    # Watermark bias strength (default: 2.5)
SEEDING_SCHEME="selfhash"   # Seeding scheme: "selfhash" or "lefthash"
HASHING_KEY=0               # Random key for hashing
GREENLIST_RATIO=0.25        # Green list ratio (default: 0.25)

# Testing parameters
MAX_PROMPTS=100             # Limit number of prompts for testing (None = all)

# Llama model to use
MODEL_PATH="meta-llama/Meta-Llama-3-8B"  # LLaMA-3 8B model
# Alternative models:
# MODEL_PATH="meta-llama/Llama-2-7b-hf"
# MODEL_PATH="meta-llama/Llama-2-13b-hf"
# MODEL_PATH="meta-llama/Meta-Llama-3-8B"

echo "Starting Llama model evaluation with HuggingFace watermarking..."
echo "Model: $MODEL_PATH"
echo "Watermarking enabled: $USE_WATERMARK"
echo "Watermark parameters: bias=$BIAS, seeding_scheme=$SEEDING_SCHEME, hashing_key=$HASHING_KEY"
echo ""

# Conditional likelihood estimation benchmarks

# echo "Running GPQA with watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks gpqa_main_n_shot \
#     --num_fewshot 5 \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running TruthfulQA MC2 with watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks truthfulqa_mc2 \
#     --num_fewshot 0 \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running ARC Challenge with watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks arc_challenge \
#     --num_fewshot 0 \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running HellaSwag with SynthID watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks hellaswag \
#     --num_fewshot 0 \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running WinoGrande with SynthID watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks winogrande \
#     --num_fewshot 5 \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running PIQA with SynthID watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks piqa \
#     --num_fewshot 0 \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running MMLU with SynthID watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks mmlu \
#     --num_fewshot 5 \
#     --model llama_synthid \
#     --batch_size 1 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,gamma=$GAMMA,delta=$DELTA,watermark_key=$WATERMARK_KEY

# # Conditional generation benchmarks

# echo "Running TruthfulQA Generation with SynthID watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks truthfulqa_gen \
#     --num_fewshot 0 \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

echo "Running BBH with SynthID watermarking..."
accelerate launch eval_llama_synthid.py \
    --tasks bbh \
    --model llama_watermark \
    --batch_size 8 \
    --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running GSM8K with SynthID watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks gsm8k \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running Minerva Math with SynthID watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks minerva_math \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running HumanEval with SynthID watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks humaneval \
#     --confirm_run_unsafe_code \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

# echo "Running MBPP with SynthID watermarking..."
# accelerate launch eval_llama_synthid.py \
#     --tasks mbpp \
#     --confirm_run_unsafe_code \
#     --model llama_watermark \
#     --batch_size 8 \
#     --model_args model_path=$MODEL_PATH,use_watermark=$USE_WATERMARK,bias=$BIAS,seeding_scheme=$SEEDING_SCHEME,hashing_key=$HASHING_KEY,greenlist_ratio=$GREENLIST_RATIO,max_prompts=$MAX_PROMPTS

echo "Watermarked evaluation completed!"

