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
WATERMARK_TYPE="none"  # Options: aaronson, green_list, none
JSONL_FILE="water-bench/2-2_finance_qa.jsonl"
OUTPUT_FILE="no_waterbench_2-2_finance_qa.json"
MAX_PROMPTS="20"
GEN_LENGTH=300
STEPS=300
TEMPERATURE=0.5
BLOCK_LENGTH=25

# Aaronson watermarking parameters
AARONSON_SEED=42
WATERMARK_STEPS=2000

# Green list watermarking parameters
GAMMA=0.5
AMPLIFICATION=2.0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --watermark_type)
            WATERMARK_TYPE="$2"
            shift 2
            ;;
        --jsonl_file)
            JSONL_FILE="$2"
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
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --amplification)
            AMPLIFICATION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$JSONL_FILE" ]; then
    echo "Error: --jsonl_file is required"
    echo "Usage: $0 --jsonl_file <path> [options]"
    echo ""
    echo "Options:"
    echo "  --watermark_type <aaronson|green_list|none>  (default: aaronson)"
    echo "  --output_file <path>                         (optional, auto-generated if not specified)"
    echo "  --max_prompts <N>                            (optional, limit number of prompts)"
    echo "  --gen_length <N>                             (default: 300)"
    echo "  --steps <N>                                   (default: 300)"
    echo "  --temperature <float>                        (default: 0.5)"
    echo "  --block_length <N>                            (default: 25)"
    echo "  --aaronson_seed <N>                          (default: 42)"
    echo "  --watermark_steps <N>                        (default: 2000, None for all steps)"
    echo "  --remasking_strategy <strategy>               (default: original)"
    echo "  --gamma <float>                              (default: 0.5, for green_list)"
    echo "  --amplification <float>                       (default: 2.0, for green_list)"
    exit 1
fi

# Change to LLaDA directory
cd /work/nvme/bemc/abagchi2/LLaDA

echo "Starting WaterBench evaluation..."
echo "Watermarking parameters:"
echo "  watermark_type=$WATERMARK_TYPE"
echo "  jsonl_file=$JSONL_FILE"
if [ -n "$OUTPUT_FILE" ]; then
    echo "  output_file=$OUTPUT_FILE"
fi
if [ -n "$MAX_PROMPTS" ]; then
    echo "  max_prompts=$MAX_PROMPTS"
fi
echo "  gen_length=$GEN_LENGTH"
echo "  steps=$STEPS"
echo "  temperature=$TEMPERATURE"
echo "  block_length=$BLOCK_LENGTH"

if [ "$WATERMARK_TYPE" = "aaronson" ]; then
    echo "  aaronson_seed=$AARONSON_SEED"
    echo "  watermark_steps=$WATERMARK_STEPS"
    echo "  remasking_strategy=original"
elif [ "$WATERMARK_TYPE" = "green_list" ]; then
    echo "  gamma=$GAMMA"
    echo "  amplification=$AMPLIFICATION"
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

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output_file $OUTPUT_FILE"
fi

if [ -n "$MAX_PROMPTS" ]; then
    CMD="$CMD --max_prompts $MAX_PROMPTS"
fi

if [ "$WATERMARK_TYPE" = "aaronson" ]; then
    CMD="$CMD --aaronson_seed $AARONSON_SEED"
    if [ -n "$WATERMARK_STEPS" ] && [ "$WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --watermark_steps $WATERMARK_STEPS"
    fi
elif [ "$WATERMARK_TYPE" = "green_list" ]; then
    CMD="$CMD --gamma $GAMMA --amplification $AMPLIFICATION"
fi

# Run the evaluation
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "WaterBench evaluation completed!"
