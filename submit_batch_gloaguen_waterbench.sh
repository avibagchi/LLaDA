#!/bin/bash
# Batch submission for WaterBench with Gloaguen et al. (OurWatermark) only.
#
# Ablates:
#   γ (Bernoulli greenlist rate): 0.1, 0.25, 0.5, 0.75, 0.9  → --gloaguen_gamma
#   δ (KL / booster budget):       0.5, 1, 2, 4, 8           → --gloaguen_delta
#   SW = [t_start=1, t_end]:       t_end ∈ {5,10,20,40,80,160,300} → --gloaguen_watermark_steps
#        (same semantics as green_list: watermark on diffusion substeps 1..t_end per block)
#
# Total jobs: 5 × 5 × 7 = 175
#
# Usage:
#   bash submit_batch_gloaguen_waterbench.sh
#
# JSON outputs go to LLaDA/gloaguen_outputs/ (see eval_waterbench.py when watermark_type=gloaguen).
# Logs: eval_waterbench.sh uses fixed SBATCH log names; consider %j in those files if jobs overwrite logs.

# Base parameters
WATERMARK_TYPE="gloaguen"
MAX_PROMPTS="100"
RANDOM_SEED=43
MAX_PROMPT_TOKENS=500
GEN_LENGTH=300
STEPS=300
TEMPERATURE=0.5
BLOCK_LENGTH=25

# Fixed Gloaguen knobs (not ablated here)
GLOAGUEN_CONV_KERNEL="-1"
GLOAGUEN_SEEDING_SCHEME="sumhash"
GLOAGUEN_GREENLIST_TYPE="bernoulli"
GLOAGUEN_TOPK=50
GLOAGUEN_N_ITER=1

cd /work/nvme/bemc/abagchi2/LLaDA || exit 1

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
JSONL_BASENAME="sampled_${MAX_PROMPTS}"

GAMMA_VALUES=(0.1 0.25 0.5 0.75 0.9)
DELTA_VALUES=(0.5 1 2 4 8)
TEND_VALUES=(5 10 20 40 80 160 300)

JOB_COUNT=0

echo "=========================================="
echo "Batch Job Submission — Gloaguen WaterBench"
echo "=========================================="
echo "Sampled JSONL: $JSONL_FILE"
echo "MAX_PROMPTS=$MAX_PROMPTS  RANDOM_SEED=$RANDOM_SEED"
echo ""
echo "  γ (gloaguen_gamma):     ${GAMMA_VALUES[*]}"
echo "  δ (gloaguen_delta):     ${DELTA_VALUES[*]}"
echo "  t_end (watermark 1..N): ${TEND_VALUES[*]}"
echo ""
echo "Total combinations: $((${#GAMMA_VALUES[@]} * ${#DELTA_VALUES[@]} * ${#TEND_VALUES[@]}))"
echo "=========================================="
echo ""

for GAMMA in "${GAMMA_VALUES[@]}"; do
    for DELTA in "${DELTA_VALUES[@]}"; do
        for TEND in "${TEND_VALUES[@]}"; do
            OUTPUT_FILE="glo_gamma=${GAMMA}_delta=${DELTA}_tend=${TEND}_${JSONL_BASENAME}.json"

            SBATCH_CMD="sbatch eval_waterbench.sh"
            SBATCH_CMD="$SBATCH_CMD --watermark_type $WATERMARK_TYPE"
            SBATCH_CMD="$SBATCH_CMD --jsonl_file $JSONL_FILE"
            SBATCH_CMD="$SBATCH_CMD --output_file $OUTPUT_FILE"
            SBATCH_CMD="$SBATCH_CMD --max_prompts $MAX_PROMPTS"
            SBATCH_CMD="$SBATCH_CMD --gen_length $GEN_LENGTH"
            SBATCH_CMD="$SBATCH_CMD --steps $STEPS"
            SBATCH_CMD="$SBATCH_CMD --temperature $TEMPERATURE"
            SBATCH_CMD="$SBATCH_CMD --block_length $BLOCK_LENGTH"
            SBATCH_CMD="$SBATCH_CMD --gloaguen_gamma $GAMMA"
            SBATCH_CMD="$SBATCH_CMD --gloaguen_delta $DELTA"
            SBATCH_CMD="$SBATCH_CMD --gloaguen_conv_kernel $GLOAGUEN_CONV_KERNEL"
            SBATCH_CMD="$SBATCH_CMD --gloaguen_seeding_scheme $GLOAGUEN_SEEDING_SCHEME"
            SBATCH_CMD="$SBATCH_CMD --gloaguen_greenlist_type $GLOAGUEN_GREENLIST_TYPE"
            SBATCH_CMD="$SBATCH_CMD --gloaguen_topk $GLOAGUEN_TOPK"
            SBATCH_CMD="$SBATCH_CMD --gloaguen_n_iter $GLOAGUEN_N_ITER"
            SBATCH_CMD="$SBATCH_CMD --gloaguen_watermark_steps $TEND"

            echo "Submitting job $((++JOB_COUNT)): γ=$GAMMA δ=$DELTA t_end=$TEND → $OUTPUT_FILE"
            eval $SBATCH_CMD
            sleep 0.5
        done
    done
done

echo "=========================================="
echo "Batch submission completed. Jobs: $JOB_COUNT"
echo "JSON outputs: LLaDA/gloaguen_outputs/"
echo "=========================================="
echo "  squeue -u \$USER"
echo "  scancel -u \$USER   # cancel all"
