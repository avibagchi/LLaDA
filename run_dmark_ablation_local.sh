#!/bin/bash
# Run DMark ablation locally (no SLURM).
#
# Runs 175 jobs SERIALLY on a single GPU.
# Set PARALLEL_GPUS to a space-separated list to stripe across multiple GPUs,
# e.g.  PARALLEL_GPUS="2 3"  runs 2 jobs at a time (one per GPU).
#
# Usage:
#   bash run_dmark_ablation_local.sh                     # GPU 3, serial
#   PARALLEL_GPUS="2 3" bash run_dmark_ablation_local.sh # GPUs 2+3, 2 at a time
#   DMARK_VARIANT=predictive bash run_dmark_ablation_local.sh

DMARK_VARIANT="${DMARK_VARIANT:-predictive_bidirectional}"
PARALLEL_GPUS="${PARALLEL_GPUS:-3}"   # default: GPU 3 only
IFS=' ' read -r -a GPU_LIST <<< "$PARALLEL_GPUS"
N_GPUS=${#GPU_LIST[@]}

MAX_PROMPTS=100
RANDOM_SEED=43
MAX_PROMPT_TOKENS=500
GEN_LENGTH=300
STEPS=300
TEMPERATURE=0.5
BLOCK_LENGTH=25
DMARK_SEED=42

GAMMA_VALUES=(0.1 0.25 0.5 0.75 0.9)
DELTA_VALUES=(0.5 1 2 4 8)
TEND_VALUES=(5 10 20 40 80 160 300)

OUTPUT_DIR="water-bench-results/json-outputs"
LOG_DIR="dmark_ablation_logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

cd /home/avb985/LLaDA || exit 1
source /home/avb985/miniforge3/bin/activate ml

SAMPLED_JSONL="water-bench-sampled_${MAX_PROMPTS}_seed${RANDOM_SEED}.jsonl"
if [ ! -f "$SAMPLED_JSONL" ]; then
    echo "Sampling $MAX_PROMPTS prompts (seed $RANDOM_SEED)..."
    python sample_waterbench_prompts.py \
        --num_samples "$MAX_PROMPTS" \
        --seed "$RANDOM_SEED" \
        --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
        --output_file "$SAMPLED_JSONL"
fi
JSONL_FILE="$SAMPLED_JSONL"
echo "Prompt file: $JSONL_FILE"
echo "GPUs: ${GPU_LIST[*]}  (${N_GPUS} parallel slots)"
echo ""

# Build job list
JOBS=()
for GAMMA in "${GAMMA_VALUES[@]}"; do
    for DELTA in "${DELTA_VALUES[@]}"; do
        for TEND in "${TEND_VALUES[@]}"; do
            OUT="dmark_${DMARK_VARIANT}_gamma=${GAMMA}_delta=${DELTA}_tend=${TEND}_sampled_${MAX_PROMPTS}.json"
            JOBS+=("$GAMMA|$DELTA|$TEND|$OUT")
        done
    done
done
TOTAL=${#JOBS[@]}
echo "Total jobs: $TOTAL (variant=$DMARK_VARIANT)"
echo "Estimated time per job on H100: ~15 min → total ~$((TOTAL * 15 / N_GPUS)) min with $N_GPUS GPU(s)"
echo ""

# Run jobs
GPU_IDX=0
PIDS=()
GPU_FOR_PID=()

for JOB in "${JOBS[@]}"; do
    IFS='|' read -r GAMMA DELTA TEND OUT_FILE <<< "$JOB"
    OUT_PATH="$OUTPUT_DIR/$OUT_FILE"

    # Skip already-finished jobs (resume-safe)
    if [ -f "$OUT_PATH" ]; then
        echo "SKIP (exists): $OUT_FILE"
        continue
    fi

    GPU="${GPU_LIST[$GPU_IDX]}"
    LOG="$LOG_DIR/${OUT_FILE%.json}.log"

    echo "START GPU=$GPU: γ=$GAMMA δ=$DELTA t_end=$TEND"

    CUDA_VISIBLE_DEVICES=$GPU python eval_waterbench.py \
        --jsonl_file "$JSONL_FILE" \
        --output_file "$OUT_FILE" \
        --watermark_type dmark \
        --dmark_variant "$DMARK_VARIANT" \
        --dmark_seed $DMARK_SEED \
        --gamma "$GAMMA" \
        --amplification "$DELTA" \
        --dmark_watermark_steps "$TEND" \
        --gen_length $GEN_LENGTH \
        --steps $STEPS \
        --temperature $TEMPERATURE \
        --block_length $BLOCK_LENGTH \
        --remasking low_confidence \
        > "$LOG" 2>&1 &

    PID=$!
    PIDS+=("$PID")
    GPU_FOR_PID+=("$GPU")

    GPU_IDX=$(( (GPU_IDX + 1) % N_GPUS ))

    # If we've filled all GPU slots, wait for any one to finish before launching next
    if [ ${#PIDS[@]} -ge $N_GPUS ]; then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
        GPU_FOR_PID=("${GPU_FOR_PID[@]:1}")
    fi
done

# Wait for remaining jobs
for PID in "${PIDS[@]}"; do
    wait "$PID"
done

echo ""
echo "All jobs done. Results in $OUTPUT_DIR/"
echo "Find optimal hyperparams:"
echo "  python find_optimal_red_green_hyperparams.py $OUTPUT_DIR/"
