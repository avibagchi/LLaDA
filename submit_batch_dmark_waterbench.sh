#!/bin/bash
# Batch submission for WaterBench with DMark (Wu et al., 2025).
#
# Ablates (matching Kirchenbauer/Gloaguen ablation structure):
#   γ (green fraction):  0.1, 0.25, 0.5, 0.75, 0.9   → --dmark_gamma
#   δ (logit bias):      0.5, 1, 2, 4, 8              → --dmark_delta
#   t_end (steps 1..N): 5, 10, 20, 40, 80, 160, 300   → --dmark_watermark_steps
#
# By default runs predictive_bidirectional (best DMark variant per the paper).
# Set DMARK_VARIANT to override.
#
# Total jobs per variant: 5 × 5 × 7 = 175
#
# Usage:
#   bash submit_batch_dmark_waterbench.sh                    # predictive_bidirectional
#   bash submit_batch_dmark_waterbench.sh predictive         # predictive only
#   bash submit_batch_dmark_waterbench.sh bidirectional      # bidirectional only
#   bash submit_batch_dmark_waterbench.sh all                # all three variants (525 jobs)

DMARK_VARIANT="${1:-predictive_bidirectional}"

# Base parameters
WATERMARK_TYPE="dmark"
MAX_PROMPTS="100"
RANDOM_SEED=43          # same as other ablations
MAX_PROMPT_TOKENS=500
GEN_LENGTH=300
STEPS=300
TEMPERATURE=0.5
BLOCK_LENGTH=25
DMARK_SEED=42

cd /work/nvme/bemc/abagchi2/LLaDA || exit 1

# Sample prompts (same file shared across all ablation jobs)
SAMPLED_JSONL="water-bench-sampled_${MAX_PROMPTS}_seed${RANDOM_SEED}.jsonl"
echo "Sampling $MAX_PROMPTS random prompts from all water-bench files..."
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
JSONL_BASENAME="sampled_${MAX_PROMPTS}"
echo "Using sampled file: $JSONL_FILE"
echo ""

GAMMA_VALUES=(0.1 0.25 0.5 0.75 0.9)
DELTA_VALUES=(0.5 1 2 4 8)
TEND_VALUES=(5 10 20 40 80 160 300)

# Determine which variants to run
if [ "$DMARK_VARIANT" = "all" ]; then
    VARIANTS=("predictive" "bidirectional" "predictive_bidirectional")
else
    VARIANTS=("$DMARK_VARIANT")
fi

JOB_COUNT=0

echo "=========================================="
echo "Batch Job Submission — DMark WaterBench"
echo "=========================================="
echo "Sampled JSONL: $JSONL_FILE"
echo "MAX_PROMPTS=$MAX_PROMPTS  RANDOM_SEED=$RANDOM_SEED"
echo "Variants: ${VARIANTS[*]}"
echo ""
echo "  γ (dmark_gamma): ${GAMMA_VALUES[*]}"
echo "  δ (dmark_delta): ${DELTA_VALUES[*]}"
echo "  t_end:           ${TEND_VALUES[*]}"
echo ""
TOTAL=$(( ${#VARIANTS[@]} * ${#GAMMA_VALUES[@]} * ${#DELTA_VALUES[@]} * ${#TEND_VALUES[@]} ))
echo "Total jobs: $TOTAL"
echo "=========================================="
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    for GAMMA in "${GAMMA_VALUES[@]}"; do
        for DELTA in "${DELTA_VALUES[@]}"; do
            for TEND in "${TEND_VALUES[@]}"; do
                OUTPUT_FILE="dmark_${VARIANT}_gamma=${GAMMA}_delta=${DELTA}_tend=${TEND}_${JSONL_BASENAME}.json"

                SBATCH_CMD="sbatch eval_waterbench.sh"
                SBATCH_CMD="$SBATCH_CMD --watermark_type $WATERMARK_TYPE"
                SBATCH_CMD="$SBATCH_CMD --jsonl_file $JSONL_FILE"
                SBATCH_CMD="$SBATCH_CMD --output_file $OUTPUT_FILE"
                SBATCH_CMD="$SBATCH_CMD --gen_length $GEN_LENGTH"
                SBATCH_CMD="$SBATCH_CMD --steps $STEPS"
                SBATCH_CMD="$SBATCH_CMD --temperature $TEMPERATURE"
                SBATCH_CMD="$SBATCH_CMD --block_length $BLOCK_LENGTH"
                SBATCH_CMD="$SBATCH_CMD --dmark_variant $VARIANT"
                SBATCH_CMD="$SBATCH_CMD --dmark_seed $DMARK_SEED"
                SBATCH_CMD="$SBATCH_CMD --dmark_gamma $GAMMA"
                SBATCH_CMD="$SBATCH_CMD --dmark_delta $DELTA"
                SBATCH_CMD="$SBATCH_CMD --dmark_watermark_steps $TEND"

                echo "Job $((++JOB_COUNT)): variant=$VARIANT γ=$GAMMA δ=$DELTA t_end=$TEND → $OUTPUT_FILE"
                eval "$SBATCH_CMD"
                sleep 0.5
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Batch submission completed. Jobs: $JOB_COUNT"
echo "JSON outputs: water-bench-results/json-outputs/"
echo "  squeue -u \$USER"
echo "  scancel -u \$USER   # cancel all"
echo ""
echo "After jobs complete, find optimal hyperparams with:"
echo "  python find_optimal_red_green_hyperparams.py \\"
echo "    water-bench-results/json-outputs/ \\"
echo "    --z-threshold 4.0"
echo "=========================================="
