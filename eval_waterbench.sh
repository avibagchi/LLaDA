#!/bin/bash
#SBATCH --job-name=waterbench_eval      # Job name
#SBATCH --output=waterbench_eval.log   # Output log file
#SBATCH --error=waterbench_error.log   # Error log file
#SBATCH --partition=gpuA100x4         
#SBATCH --account=bemc-delta-gpu         # Your valid Slurm account
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # One task
#SBATCH --cpus-per-task=16             # 16 cores per GPU
#SBATCH --mem=96G                      # Memory for the job
#SBATCH --time=24:00:00                # Time limit

# Load correct CUDA (try to load, but ignore errors if modules don't exist)
# Note: Module availability may differ between login and compute nodes
# When running interactively, modules may already be loaded or unavailable
module load gcc/11.4.0 2>/dev/null || true
module load cuda/12.3.0 2>/dev/null || true
module load cray-python/3.11.5 2>/dev/null || true

# Activate your Python environment
source /work/nvme/bemc/python_envs/sedd_env_3/bin/activate

# Verify torch is available
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch (torch) is not available in the Python environment."
    echo "Please install PyTorch in llada_env_5 or check your environment setup."
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Parse command line arguments
WATERMARK_TYPE="aaronson"  # Options: aaronson, green_list, none
JSONL_FILE=""  # Will be generated from sampled prompts if not specified
OUTPUT_FILE="robust_500_m=50.json" # OUTPUT_FILE="run_gamma=0.9_delta=10_steps=100_waterbench_2-2_finance_qa.json"
MAX_PROMPTS="500"  # Number of random prompts to sample from all water-bench files
USE_ALL_WATERBENCH=true  # If true, sample from all water-bench files; if false, use specific JSONL_FILE
RANDOM_SEED=42  # Seed for random sampling (for reproducibility), used 43 for ablations
GEN_LENGTH=300
STEPS=300
TEMPERATURE=0.5
BLOCK_LENGTH=25

# Aaronson watermarking parameters
AARONSON_SEED=42
WATERMARK_STEPS=300
# Watermark param m: RNG seed = position mod m (thwarts prefix deletion). Empty = disabled.
AARONSON_WM_PARAM_M="50"
# Random prefix deletion before scoring: int (max tokens) or float (max fraction). Empty = no deletion.
AARONSON_PREFIX_DELETE_MAX="100"
AARONSON_PREFIX_DELETE_SEED=123

# Green list watermarking parameters
# [0.1, 0.5, 0.9]
GAMMA=0.1
# [0.5, 1, 2, 5, 10]
AMPLIFICATION=8
# [100, 200, 300]
GREEN_LIST_WATERMARK_STEPS="10"  # Empty means all steps, set to int (e.g., 100) to watermark steps <= N

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --watermark_type)
            WATERMARK_TYPE="$2"
            shift 2
            ;;
        --jsonl_file)
            JSONL_FILE="$2"
            USE_ALL_WATERBENCH=false  # If specific file provided, don't sample
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --max_prompts)
            MAX_PROMPTS="$2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --use_all_waterbench)
            USE_ALL_WATERBENCH=true
            shift
            ;;
        --use_specific_file)
            USE_ALL_WATERBENCH=false
            shift
            ;;
        --gen_length)
            GEN_LENGTH="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --block_length)
            BLOCK_LENGTH="$2"
            shift 2
            ;;
        --aaronson_seed)
            AARONSON_SEED="$2"
            shift 2
            ;;
        --watermark_steps)
            WATERMARK_STEPS="$2"
            shift 2
            ;;
        --aaronson_wm_param_m)
            AARONSON_WM_PARAM_M="$2"
            shift 2
            ;;
        --aaronson_prefix_delete_max)
            AARONSON_PREFIX_DELETE_MAX="$2"
            shift 2
            ;;
        --aaronson_prefix_delete_seed)
            AARONSON_PREFIX_DELETE_SEED="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --amplification)
            AMPLIFICATION="$2"
            shift 2
            ;;
        --green_list_watermark_steps)
            GREEN_LIST_WATERMARK_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If using all water-bench files, create sampled JSONL file
if [ "$USE_ALL_WATERBENCH" = true ]; then
    SAMPLED_JSONL="water-bench-sampled_${MAX_PROMPTS}_seed${RANDOM_SEED}.jsonl"
    MAX_PROMPT_TOKENS=500  # Filter out prompts with contexts longer than this (to avoid OOM)
    echo "Sampling $MAX_PROMPTS random prompts from all water-bench files..."
    echo "Filtering out prompts with contexts longer than $MAX_PROMPT_TOKENS tokens..."
    python sample_waterbench_prompts.py \
        --num_samples "$MAX_PROMPTS" \
        --seed "$RANDOM_SEED" \
        --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
        --output_file "$SAMPLED_JSONL"
    
    if [ ! -f "$SAMPLED_JSONL" ]; then
        echo "Error: Failed to create sampled JSONL file"
        exit 1
    fi
    
    JSONL_FILE="$SAMPLED_JSONL"
    echo "Using sampled file: $JSONL_FILE"
    echo ""
fi

# Check required arguments
if [ -z "$JSONL_FILE" ]; then
    echo "Error: --jsonl_file is required, or use --use_all_waterbench to sample from all files"
    echo "Usage: $0 [--jsonl_file <path> | --use_all_waterbench] [options]"
    echo ""
    echo "Options:"
    echo "  --jsonl_file <path>                          (specific JSONL file to use)"
    echo "  --use_all_waterbench                         (sample from all water-bench files, default: true)"
    echo "  --max_prompts <N>                             (number of prompts to sample, default: 500)"
    echo "  --random_seed <N>                             (seed for random sampling, default: 42)"
    echo "  --watermark_type <aaronson|green_list|none>  (default: green_list)"
    echo "  --output_file <path>                         (optional, auto-generated if not specified)"
    echo "  --gen_length <N>                             (default: 300)"
    echo "  --steps <N>                                   (default: 300)"
    echo "  --temperature <float>                        (default: 0.5)"
    echo "  --block_length <N>                            (default: 25)"
    echo "  --aaronson_seed <N>                          (default: 42)"
    echo "  --watermark_steps <N>                        (default: 300, for aaronson; None for all steps)"
    echo "  --aaronson_wm_param_m <N>                    (optional; enable m for prefix-deletion robustness)"
    echo "  --aaronson_prefix_delete_max <N|float>       (optional; random prefix deletion before scoring)"
    echo "  --green_list_watermark_steps <N>              (default: 100, for green_list; set to N to watermark steps <= N)"
    echo "  --gamma <float>                              (default: 0.9, for green_list)"
    echo "  --amplification <float>                       (default: 10, for green_list)"
    exit 1
fi

# Change to LLaDA directory
cd /work/nvme/bemc/abagchi2/LLaDA

echo "Starting WaterBench evaluation..."
echo "Watermarking parameters:"
echo "  watermark_type=$WATERMARK_TYPE"
echo "  jsonl_file=$JSONL_FILE"
if [ "$USE_ALL_WATERBENCH" = true ]; then
    echo "  sampling_mode=random from all water-bench files"
    echo "  num_samples=$MAX_PROMPTS"
    echo "  random_seed=$RANDOM_SEED"
else
    echo "  sampling_mode=specific file"
fi
if [ -n "$OUTPUT_FILE" ]; then
    echo "  output_file=$OUTPUT_FILE"
fi
echo "  gen_length=$GEN_LENGTH"
echo "  steps=$STEPS"
echo "  temperature=$TEMPERATURE"
echo "  block_length=$BLOCK_LENGTH"

if [ "$WATERMARK_TYPE" = "aaronson" ]; then
    echo "  aaronson_seed=$AARONSON_SEED"
    echo "  watermark_steps=$WATERMARK_STEPS"
    if [ -n "$AARONSON_WM_PARAM_M" ]; then
        echo "  aaronson_wm_param_m=$AARONSON_WM_PARAM_M"
    else
        echo "  aaronson_wm_param_m=disabled"
    fi
    if [ -n "$AARONSON_PREFIX_DELETE_MAX" ]; then
        echo "  aaronson_prefix_delete_max=$AARONSON_PREFIX_DELETE_MAX"
        echo "  aaronson_prefix_delete_seed=$AARONSON_PREFIX_DELETE_SEED"
    fi
    echo "  remasking_strategy=original"
elif [ "$WATERMARK_TYPE" = "green_list" ]; then
    echo "  gamma=$GAMMA"
    echo "  amplification=$AMPLIFICATION"
    if [ -n "$GREEN_LIST_WATERMARK_STEPS" ]; then
        echo "  watermark_steps=$GREEN_LIST_WATERMARK_STEPS"
    else
        echo "  watermark_steps=all (all steps watermarked)"
    fi
fi
echo ""

# Build command
CMD="python eval_waterbench.py"
CMD="$CMD --jsonl_file $JSONL_FILE"
CMD="$CMD --watermark_type $WATERMARK_TYPE"
CMD="$CMD --gen_length $GEN_LENGTH"
CMD="$CMD --steps $STEPS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --block_length $BLOCK_LENGTH"
CMD="$CMD --remasking low_confidence"

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output_file $OUTPUT_FILE"
fi

# Only pass --max_prompts if using a specific file (not sampled)
# If sampled, the file already contains exactly MAX_PROMPTS prompts
if [ "$USE_ALL_WATERBENCH" = false ] && [ -n "$MAX_PROMPTS" ]; then
    CMD="$CMD --max_prompts $MAX_PROMPTS"
fi

if [ "$WATERMARK_TYPE" = "aaronson" ]; then
    CMD="$CMD --aaronson_seed $AARONSON_SEED"
    if [ -n "$WATERMARK_STEPS" ] && [ "$WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --watermark_steps $WATERMARK_STEPS"
    fi
    if [ -n "$AARONSON_WM_PARAM_M" ]; then
        CMD="$CMD --aaronson_wm_param_m $AARONSON_WM_PARAM_M"
    fi
    if [ -n "$AARONSON_PREFIX_DELETE_MAX" ]; then
        CMD="$CMD --aaronson_prefix_delete_max $AARONSON_PREFIX_DELETE_MAX --aaronson_prefix_delete_seed $AARONSON_PREFIX_DELETE_SEED"
    fi
elif [ "$WATERMARK_TYPE" = "green_list" ]; then
    CMD="$CMD --gamma $GAMMA --amplification $AMPLIFICATION"
    if [ -n "$GREEN_LIST_WATERMARK_STEPS" ] && [ "$GREEN_LIST_WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --watermark_steps $GREEN_LIST_WATERMARK_STEPS"
    fi
fi

# Run the evaluation
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "WaterBench evaluation completed!"
