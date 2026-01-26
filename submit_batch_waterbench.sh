#!/bin/bash
# Batch submission script for WaterBench evaluation with different parameter combinations
# This script submits multiple SLURM jobs for all combinations of:
#   - GAMMA: [0.1, 0.5, 0.9]
#   - AMPLIFICATION: [0.5, 1, 2, 5, 10]
#   - GREEN_LIST_WATERMARK_STEPS: [100, 200, 300]

# Base parameters
WATERMARK_TYPE="green_list"
MAX_PROMPTS="100"  # Number of random prompts to sample from all water-bench files
RANDOM_SEED=43  # used seed of 43 for ablation study, use 42 once you found optimal
MAX_PROMPT_TOKENS=500  # Filter out prompts with contexts longer than this (to avoid OOM)
GEN_LENGTH=300
STEPS=300
TEMPERATURE=0.5
BLOCK_LENGTH=25

# Change to LLaDA directory
cd /work/nvme/bemc/abagchi2/LLaDA

# Sample random prompts from all water-bench files
SAMPLED_JSONL="water-bench-sampled_${MAX_PROMPTS}_seed${RANDOM_SEED}.jsonl"
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

# Extract base name from JSONL file for output naming
JSONL_BASENAME="sampled_${MAX_PROMPTS}"

# Parameter arrays
GAMMA_VALUES=(0.1 0.25 0.5 0.75 0.9)
AMPLIFICATION_VALUES=(0.5 1 2 4 8)
WATERMARK_STEPS_VALUES=(5 10 20 40 80 160 300) # (50 100 150 200 250 300)

# Counter for submitted jobs
JOB_COUNT=0

echo "=========================================="
echo "Batch Job Submission for WaterBench"
echo "=========================================="
echo "Watermark Type: $WATERMARK_TYPE"
echo "Sampled JSONL File: $JSONL_FILE"
echo "Number of sampled prompts: $MAX_PROMPTS"
echo "Random seed: $RANDOM_SEED"
echo "Max prompt tokens (filter): $MAX_PROMPT_TOKENS"
echo ""
echo "Parameter ranges:"
echo "  GAMMA: ${GAMMA_VALUES[@]}"
echo "  AMPLIFICATION: ${AMPLIFICATION_VALUES[@]}"
echo "  WATERMARK_STEPS: ${WATERMARK_STEPS_VALUES[@]}"
echo ""
echo "Total combinations: $((${#GAMMA_VALUES[@]} * ${#AMPLIFICATION_VALUES[@]} * ${#WATERMARK_STEPS_VALUES[@]}))"
echo "=========================================="
echo ""

# Loop through all combinations
for GAMMA in "${GAMMA_VALUES[@]}"; do
    for AMPLIFICATION in "${AMPLIFICATION_VALUES[@]}"; do
        for WATERMARK_STEPS in "${WATERMARK_STEPS_VALUES[@]}"; do
            # Create output file name: run_gamma=<gamma>_delta=<amplification>_steps=<steps>_<jsonl_basename>.json
            OUTPUT_FILE="run_gamma=${GAMMA}_delta=${AMPLIFICATION}_steps=${WATERMARK_STEPS}_${JSONL_BASENAME}.json"
            
            # Build the sbatch command
            SBATCH_CMD="sbatch eval_waterbench.sh"
            SBATCH_CMD="$SBATCH_CMD --watermark_type $WATERMARK_TYPE"
            SBATCH_CMD="$SBATCH_CMD --jsonl_file $JSONL_FILE"
            SBATCH_CMD="$SBATCH_CMD --output_file $OUTPUT_FILE"
            # Don't pass --max_prompts since the sampled file already has exactly MAX_PROMPTS prompts
            SBATCH_CMD="$SBATCH_CMD --gen_length $GEN_LENGTH"
            SBATCH_CMD="$SBATCH_CMD --steps $STEPS"
            SBATCH_CMD="$SBATCH_CMD --temperature $TEMPERATURE"
            SBATCH_CMD="$SBATCH_CMD --block_length $BLOCK_LENGTH"
            SBATCH_CMD="$SBATCH_CMD --gamma $GAMMA"
            SBATCH_CMD="$SBATCH_CMD --amplification $AMPLIFICATION"
            SBATCH_CMD="$SBATCH_CMD --green_list_watermark_steps $WATERMARK_STEPS"
            
            # Submit the job
            echo "Submitting job $((++JOB_COUNT)):"
            echo "  GAMMA=$GAMMA, AMPLIFICATION=$AMPLIFICATION, WATERMARK_STEPS=$WATERMARK_STEPS"
            echo "  OUTPUT_FILE=$OUTPUT_FILE"
            echo ""
            
            # Execute the sbatch command
            eval $SBATCH_CMD
            
            # Small delay to avoid overwhelming the scheduler
            sleep 0.5
        done
    done
done

echo "=========================================="
echo "Batch submission completed!"
echo "Total jobs submitted: $JOB_COUNT"
echo "=========================================="
echo ""
echo "To check job status, use:"
echo "  squeue -u \$USER"
echo ""
echo "To cancel all submitted jobs, use:"
echo "  scancel -u \$USER"
