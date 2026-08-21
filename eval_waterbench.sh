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
WATERMARK_TYPE="gloaguen"  # Options: aaronson, green_list, gloaguen, none
JSONL_FILE=""  # Will be generated from sampled prompts if not specified
OUTPUT_FILE="gloaguen_1.json" # OUTPUT_FILE="run_gamma=0.9_delta=10_steps=100_waterbench_2-2_finance_qa.json"
MAX_PROMPTS="100"  # Number of random prompts to sample from all water-bench files
USE_ALL_WATERBENCH=true  # If true, sample from all water-bench files; if false, use specific JSONL_FILE
RANDOM_SEED=42  # Seed for random sampling (for reproducibility), used 43 for ablations
GEN_LENGTH=300
STEPS=300
TEMPERATURE=0.5
BLOCK_LENGTH=25

# Aaronson watermarking parameters
AARONSON_SEED=42
WATERMARK_STEPS=300

# Green list watermarking parameters
# [0.1, 0.5, 0.9]
GAMMA=0.1
# [0.5, 1, 2, 5, 10]
AMPLIFICATION=8
# [100, 200, 300]
GREEN_LIST_WATERMARK_STEPS="20"  # Empty means all steps, set to int (e.g., 100) to watermark steps <= N

# Gloaguen et al. (Diffusion-KGW optimal Gaussian; OurWatermark in diffusion-lm-watermark)
GLOAGUEN_DELTA=4
GLOAGUEN_CONV_KERNEL="-1"   # comma-separated offsets, e.g. -1 or -2,-1 (use = in CLI to avoid negative parsed as flag)
GLOAGUEN_SEEDING_SCHEME="sumhash"
GLOAGUEN_GREENLIST_TYPE="bernoulli"
GLOAGUEN_GAMMA=0.25
GLOAGUEN_TOPK=50
GLOAGUEN_N_ITER=1
GLOAGUEN_WATERMARK_STEPS="300"  # same semantics as green_list: steps 1..N; empty/None = all steps

# DMark watermarking parameters (Wu et al., 2025)
DMARK_VARIANT="predictive_bidirectional"   # predictive | bidirectional | predictive_bidirectional
DMARK_SEED=42
DMARK_GAMMA=0.5
DMARK_DELTA=2
DMARK_WATERMARK_STEPS=""  # empty = all steps

# CDMArk watermarking parameters (zero-bit adaptation)
CDMARK_SEED=42
CDMARK_M=1
CDMARK_GAMMA=0.5
CDMARK_DELTA=2
CDMARK_WATERMARK_STEPS=""

# dgMARK watermarking parameters
DGMARK_SEED=42
DGMARK_WATERMARK_STEPS=""

# LR-DWM watermarking parameters
LRDWM_SEED=42
LRDWM_GAMMA=0.5
LRDWM_DELTA=2
LRDWM_WATERMARK_STEPS=""

# UMR watermarking parameters (zero-bit adaptation)
UMR_SEED=42
UMR_GAMMA=0.5
UMR_DELTA=2
UMR_WATERMARK_STEPS=""

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
        --gloaguen_delta)
            GLOAGUEN_DELTA="$2"
            shift 2
            ;;
        --gloaguen_conv_kernel)
            GLOAGUEN_CONV_KERNEL="$2"
            shift 2
            ;;
        --gloaguen_seeding_scheme)
            GLOAGUEN_SEEDING_SCHEME="$2"
            shift 2
            ;;
        --gloaguen_greenlist_type)
            GLOAGUEN_GREENLIST_TYPE="$2"
            shift 2
            ;;
        --gloaguen_gamma)
            GLOAGUEN_GAMMA="$2"
            shift 2
            ;;
        --gloaguen_topk)
            GLOAGUEN_TOPK="$2"
            shift 2
            ;;
        --gloaguen_n_iter)
            GLOAGUEN_N_ITER="$2"
            shift 2
            ;;
        --gloaguen_watermark_steps)
            GLOAGUEN_WATERMARK_STEPS="$2"
            shift 2
            ;;
        --no-gloaguen-enforce-kl)
            GLOAGUEN_ENFORCE_KL="0"
            shift
            ;;
        --gloaguen-enforce-kl)
            GLOAGUEN_ENFORCE_KL="1"
            shift
            ;;
        --dmark_variant)
            DMARK_VARIANT="$2"
            shift 2
            ;;
        --dmark_seed)
            DMARK_SEED="$2"
            shift 2
            ;;
        --dmark_gamma)
            DMARK_GAMMA="$2"
            shift 2
            ;;
        --dmark_delta)
            DMARK_DELTA="$2"
            shift 2
            ;;
        --dmark_watermark_steps)
            DMARK_WATERMARK_STEPS="$2"
            shift 2
            ;;
        --cdmark_seed)
            CDMARK_SEED="$2"
            shift 2
            ;;
        --cdmark_m)
            CDMARK_M="$2"
            shift 2
            ;;
        --cdmark_gamma)
            CDMARK_GAMMA="$2"
            shift 2
            ;;
        --cdmark_delta)
            CDMARK_DELTA="$2"
            shift 2
            ;;
        --cdmark_watermark_steps)
            CDMARK_WATERMARK_STEPS="$2"
            shift 2
            ;;
        --dgmark_seed)
            DGMARK_SEED="$2"
            shift 2
            ;;
        --dgmark_watermark_steps)
            DGMARK_WATERMARK_STEPS="$2"
            shift 2
            ;;
        --lrdwm_seed)
            LRDWM_SEED="$2"
            shift 2
            ;;
        --lrdwm_gamma)
            LRDWM_GAMMA="$2"
            shift 2
            ;;
        --lrdwm_delta)
            LRDWM_DELTA="$2"
            shift 2
            ;;
        --lrdwm_watermark_steps)
            LRDWM_WATERMARK_STEPS="$2"
            shift 2
            ;;
        --umr_seed)
            UMR_SEED="$2"
            shift 2
            ;;
        --umr_gamma)
            UMR_GAMMA="$2"
            shift 2
            ;;
        --umr_delta)
            UMR_DELTA="$2"
            shift 2
            ;;
        --umr_watermark_steps)
            UMR_WATERMARK_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default: enforce KL unless --no-gloaguen_enforce_kl was passed
if [ -z "${GLOAGUEN_ENFORCE_KL:-}" ]; then
    GLOAGUEN_ENFORCE_KL="1"
fi

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
    echo "  --watermark_type <aaronson|green_list|gloaguen|none>"
    echo "  --output_file <path>                         (optional, auto-generated if not specified)"
    echo "  --gen_length <N>                             (default: 300)"
    echo "  --steps <N>                                   (default: 300)"
    echo "  --temperature <float>                        (default: 0.5)"
    echo "  --block_length <N>                            (default: 25)"
    echo "  --aaronson_seed <N>                          (default: 42)"
    echo "  --watermark_steps <N>                        (default: 300, for aaronson; None for all steps)"
    echo "  --green_list_watermark_steps <N>              (default: 100, for green_list; set to N to watermark steps <= N)"
    echo "  --gamma <float>                              (default: 0.9, for green_list)"
    echo "  --amplification <float>                       (default: 10, for green_list)"
    echo "  Gloaguen (Gloaguen et al.; diffusion-lm-watermark OurWatermark):"
    echo "  --gloaguen_delta <float>                      (default: 2.0)"
    echo "  --gloaguen_conv_kernel <str>                  (default: -1; use e.g. -2,-1)"
    echo "  --gloaguen_seeding_scheme <sumhash|minhash>   (default: sumhash)"
    echo "  --gloaguen_greenlist_type <bernoulli|gaussian|lognormal> (default: bernoulli)"
    echo "  --gloaguen_gamma <float>                     (default: 0.25; bernoulli only)"
    echo "  --gloaguen_topk <N>                           (default: 100)"
    echo "  --gloaguen_n_iter <N>                         (default: 1)"
    echo "  --gloaguen_watermark_steps <N>                (default: 10; empty string = all steps)"
    echo "  --gloaguen-enforce-kl / --no-gloaguen-enforce-kl  (default: enforce on)"
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
    echo "  remasking_strategy=original"
elif [ "$WATERMARK_TYPE" = "green_list" ]; then
    echo "  gamma=$GAMMA"
    echo "  amplification=$AMPLIFICATION"
    if [ -n "$GREEN_LIST_WATERMARK_STEPS" ]; then
        echo "  watermark_steps=$GREEN_LIST_WATERMARK_STEPS"
    else
        echo "  watermark_steps=all (all steps watermarked)"
    fi
elif [ "$WATERMARK_TYPE" = "gloaguen" ]; then
    echo "  delta=$GLOAGUEN_DELTA enforce_kl=$GLOAGUEN_ENFORCE_KL"
    echo "  conv_kernel=$GLOAGUEN_CONV_KERNEL seeding=$GLOAGUEN_SEEDING_SCHEME"
    echo "  greenlist_type=$GLOAGUEN_GREENLIST_TYPE gamma=$GLOAGUEN_GAMMA"
    echo "  topk=$GLOAGUEN_TOPK n_iter=$GLOAGUEN_N_ITER"
    if [ -n "$GLOAGUEN_WATERMARK_STEPS" ] && [ "$GLOAGUEN_WATERMARK_STEPS" != "None" ]; then
        echo "  watermark_steps=$GLOAGUEN_WATERMARK_STEPS"
    else
        echo "  watermark_steps=all"
    fi
elif [ "$WATERMARK_TYPE" = "dmark" ]; then
    echo "  variant=$DMARK_VARIANT"
    echo "  gamma=$DMARK_GAMMA  delta=$DMARK_DELTA  seed=$DMARK_SEED"
    if [ -n "$DMARK_WATERMARK_STEPS" ] && [ "$DMARK_WATERMARK_STEPS" != "None" ]; then
        echo "  watermark_steps=$DMARK_WATERMARK_STEPS"
    else
        echo "  watermark_steps=all"
    fi
elif [ "$WATERMARK_TYPE" = "cdmark" ]; then
    echo "  seed=$CDMARK_SEED  m=$CDMARK_M  gamma=$CDMARK_GAMMA  delta=$CDMARK_DELTA"
    [ -n "$CDMARK_WATERMARK_STEPS" ] && echo "  watermark_steps=$CDMARK_WATERMARK_STEPS" || echo "  watermark_steps=all"
elif [ "$WATERMARK_TYPE" = "dgmark" ]; then
    echo "  seed=$DGMARK_SEED"
    [ -n "$DGMARK_WATERMARK_STEPS" ] && echo "  watermark_steps=$DGMARK_WATERMARK_STEPS" || echo "  watermark_steps=all"
elif [ "$WATERMARK_TYPE" = "lrdwm" ]; then
    echo "  seed=$LRDWM_SEED  gamma=$LRDWM_GAMMA  delta=$LRDWM_DELTA"
    [ -n "$LRDWM_WATERMARK_STEPS" ] && echo "  watermark_steps=$LRDWM_WATERMARK_STEPS" || echo "  watermark_steps=all"
elif [ "$WATERMARK_TYPE" = "umr" ]; then
    echo "  seed=$UMR_SEED  gamma=$UMR_GAMMA  delta=$UMR_DELTA"
    [ -n "$UMR_WATERMARK_STEPS" ] && echo "  watermark_steps=$UMR_WATERMARK_STEPS" || echo "  watermark_steps=all"
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
elif [ "$WATERMARK_TYPE" = "green_list" ]; then
    CMD="$CMD --gamma $GAMMA --amplification $AMPLIFICATION"
    if [ -n "$GREEN_LIST_WATERMARK_STEPS" ] && [ "$GREEN_LIST_WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --watermark_steps $GREEN_LIST_WATERMARK_STEPS"
    fi
elif [ "$WATERMARK_TYPE" = "gloaguen" ]; then
    CMD="$CMD --gloaguen_delta $GLOAGUEN_DELTA"
    CMD="$CMD --gloaguen_conv_kernel $GLOAGUEN_CONV_KERNEL"
    CMD="$CMD --gloaguen_seeding_scheme $GLOAGUEN_SEEDING_SCHEME"
    CMD="$CMD --gloaguen_greenlist_type $GLOAGUEN_GREENLIST_TYPE"
    CMD="$CMD --gloaguen_gamma $GLOAGUEN_GAMMA"
    CMD="$CMD --gloaguen_topk $GLOAGUEN_TOPK"
    CMD="$CMD --gloaguen_n_iter $GLOAGUEN_N_ITER"
    if [ "$GLOAGUEN_ENFORCE_KL" = "0" ]; then
        CMD="$CMD --no-gloaguen-enforce-kl"
    fi
    if [ -n "$GLOAGUEN_WATERMARK_STEPS" ] && [ "$GLOAGUEN_WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --watermark_steps $GLOAGUEN_WATERMARK_STEPS"
    fi
elif [ "$WATERMARK_TYPE" = "dmark" ]; then
    CMD="$CMD --dmark_variant $DMARK_VARIANT"
    CMD="$CMD --dmark_seed $DMARK_SEED"
    CMD="$CMD --gamma $DMARK_GAMMA"
    CMD="$CMD --amplification $DMARK_DELTA"
    if [ -n "$DMARK_WATERMARK_STEPS" ] && [ "$DMARK_WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --dmark_watermark_steps $DMARK_WATERMARK_STEPS"
    fi
elif [ "$WATERMARK_TYPE" = "cdmark" ]; then
    CMD="$CMD --cdmark_seed $CDMARK_SEED --cdmark_m $CDMARK_M"
    CMD="$CMD --gamma $CDMARK_GAMMA --amplification $CDMARK_DELTA"
    if [ -n "$CDMARK_WATERMARK_STEPS" ] && [ "$CDMARK_WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --cdmark_watermark_steps $CDMARK_WATERMARK_STEPS"
    fi
elif [ "$WATERMARK_TYPE" = "dgmark" ]; then
    CMD="$CMD --dgmark_seed $DGMARK_SEED"
    if [ -n "$DGMARK_WATERMARK_STEPS" ] && [ "$DGMARK_WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --dgmark_watermark_steps $DGMARK_WATERMARK_STEPS"
    fi
elif [ "$WATERMARK_TYPE" = "lrdwm" ]; then
    CMD="$CMD --lrdwm_seed $LRDWM_SEED"
    CMD="$CMD --gamma $LRDWM_GAMMA --amplification $LRDWM_DELTA"
    if [ -n "$LRDWM_WATERMARK_STEPS" ] && [ "$LRDWM_WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --lrdwm_watermark_steps $LRDWM_WATERMARK_STEPS"
    fi
elif [ "$WATERMARK_TYPE" = "umr" ]; then
    CMD="$CMD --umr_seed $UMR_SEED"
    CMD="$CMD --gamma $UMR_GAMMA --amplification $UMR_DELTA"
    if [ -n "$UMR_WATERMARK_STEPS" ] && [ "$UMR_WATERMARK_STEPS" != "None" ]; then
        CMD="$CMD --umr_watermark_steps $UMR_WATERMARK_STEPS"
    fi
fi

# Run the evaluation
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "WaterBench evaluation completed!"
