#!/bin/bash

# Test original remasking strategy across 15 prompts
# Tracks normalized Aaronson watermark scores

echo "Testing Original Remasking Strategy (15 prompts)..."
echo ""

# Configuration
GEN_LENGTH=256
STEPS=256
BLOCK_LENGTH=256
NUM_PROMPTS=15

# Test 4 different watermark step values
STEP_VALUES="0,64,128,256"

# Optional: activate environment if needed
# source /work/nvme/bemc/python_envs/llada_env_5/bin/activate

cd /work/nvme/bemc/abagchi2/LLaDA

# Run the test
python test_original_strategy.py \
    --gen_length $GEN_LENGTH \
    --steps $STEPS \
    --block_length $BLOCK_LENGTH \
    --temperature 0.0 \
    --device cuda \
    --num_prompts $NUM_PROMPTS \
    --step_values "$STEP_VALUES"

echo ""
echo "Test complete!"
echo ""
echo "Check the generated files:"
echo "  - original_strategy_results_*.json (detailed results)"
echo "  - original_strategy_results_*.csv (table format)"

