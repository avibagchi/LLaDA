#!/bin/bash

# Run Aaronson watermark experiment across 10 prompts and 10 step values
# Generates JSON and CSV output with normalized scores

echo "Starting Aaronson watermark experiment..."
echo ""

# Default configuration
STEPS=256
GEN_LENGTH=256
BLOCK_LENGTH=256
STEP_VALUES="0,256"

# Optional: activate environment if needed
# source /work/nvme/bemc/python_envs/llada_env_5/bin/activate

cd /work/nvme/bemc/abagchi2/LLaDA

python experiment_watermark_steps.py \
    --gen_length $GEN_LENGTH \
    --steps $STEPS \
    --block_length $BLOCK_LENGTH \
    --step_values "$STEP_VALUES" \
    --temperature 0.0 \
    --device cuda

echo ""
echo "Experiment complete!"
echo ""
echo "Check the generated files:"
echo "  - watermark_experiment_*.json (detailed results)"
echo "  - watermark_experiment_*.csv (matrix format for easy analysis)"

