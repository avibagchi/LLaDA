#!/bin/bash
# Interactive version (no SLURM directives) - run directly with bash

# Load correct CUDA (try to load, but ignore errors if modules don't exist)
module load gcc/11.4.0 2>/dev/null || true
module load cuda/12.3.0 2>/dev/null || true
module load cray-python/3.11.5 2>/dev/null || true

# Activate your Python environment
source /work/nvme/bemc/python_envs/sedd_env_3/bin/activate

# Verify torch is available
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch (torch) is not available in the Python environment."
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Change to LLaDA directory
cd /work/nvme/bemc/abagchi2/LLaDA

# Create custom prompts JSONL file
JSONL_FILE="custom_prompts.jsonl"
echo "Creating custom prompts JSONL file: $JSONL_FILE"

cat > "$JSONL_FILE" << 'EOF'
{"input": "Write a short story about a cat.", "context": "", "outputs": [], "dataset": "custom", "_id": "1"}
{"input": "Write a short story about a prince who saves a princess from a dragon.", "context": "", "outputs": [], "dataset": "custom", "_id": "2"}
{"input": "Explain the concept of photosynthesis in simple terms.", "context": "", "outputs": [], "dataset": "custom", "_id": "3"}
{"input": "What are the benefits of renewable energy?", "context": "", "outputs": [], "dataset": "custom", "_id": "4"}
{"input": "Describe the process of making bread from scratch.", "context": "", "outputs": [], "dataset": "custom", "_id": "5"}
{"input": "Write a poem about the ocean.", "context": "", "outputs": [], "dataset": "custom", "_id": "6"}
{"input": "Explain how a computer processor works.", "context": "", "outputs": [], "dataset": "custom", "_id": "7"}
{"input": "What are the main causes of climate change?", "context": "", "outputs": [], "dataset": "custom", "_id": "8"}
EOF

echo "Created $JSONL_FILE with $(wc -l < "$JSONL_FILE") prompts"

# Aaronson watermarking parameters
WATERMARK_TYPE="aaronson"
AARONSON_SEED=42
WATERMARK_STEPS=300  # Watermark at steps <= 300 (all steps if steps=300)
STEPS=300
GEN_LENGTH=300
TEMPERATURE=0.5
BLOCK_LENGTH=25

# Generate output filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="custom_prompts_aaronson_${TIMESTAMP}.json"

echo ""
echo "Starting Aaronson watermarking evaluation..."
echo "Watermarking parameters:"
echo "  watermark_type=$WATERMARK_TYPE"
echo "  jsonl_file=$JSONL_FILE"
echo "  output_file=$OUTPUT_FILE"
echo "  aaronson_seed=$AARONSON_SEED"
echo "  watermark_steps=$WATERMARK_STEPS"
echo "  steps=$STEPS"
echo "  gen_length=$GEN_LENGTH"
echo "  temperature=$TEMPERATURE"
echo "  block_length=$BLOCK_LENGTH"
echo ""

# Build command
CMD="python eval_waterbench.py"
CMD="$CMD --jsonl_file $JSONL_FILE"
CMD="$CMD --watermark_type $WATERMARK_TYPE"
CMD="$CMD --aaronson_seed $AARONSON_SEED"
CMD="$CMD --watermark_steps $WATERMARK_STEPS"
CMD="$CMD --gen_length $GEN_LENGTH"
CMD="$CMD --steps $STEPS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --block_length $BLOCK_LENGTH"
CMD="$CMD --output_file $OUTPUT_FILE"
# In eval_waterbench.py call, add:
CMD="$CMD --remasking low_confidence"

# Run the evaluation
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "Aaronson watermarking evaluation completed!"
echo "Results saved to: water-bench-results/json-outputs/$OUTPUT_FILE"
