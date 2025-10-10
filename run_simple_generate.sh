#!/bin/bash

# Simple script to run LLaDA generation with Aaronson watermarking
# Usage: ./run_simple_generate.sh "Your prompt here" [watermark_steps]

PROMPT="${1:-What is the capital of France?}"
WATERMARK_STEPS="${2:-100}"

# Activate environment if needed
# source /work/nvme/bemc/python_envs/llada_env_5/bin/activate

echo "Running LLaDA generation with Aaronson watermarking..."
echo "Prompt: $PROMPT"
echo "Watermark steps: $WATERMARK_STEPS"
echo ""

python simple_generate.py \
    --prompt "$PROMPT" \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --watermark_steps $WATERMARK_STEPS \
    --aaronson_seed 42 \
    --temperature 0.0 \
    --device cuda

echo ""
echo "Generation complete!"


