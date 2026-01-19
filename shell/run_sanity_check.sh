#!/bin/bash

# Run sanity check for Aaronson watermark detection alignment

echo "Running Aaronson watermark detection sanity check..."
echo ""

cd /work/nvme/bemc/abagchi2/LLaDA

# Optional: activate environment if needed
# source /work/nvme/bemc/python_envs/llada_env_5/bin/activate

python sanity_check_detection.py \
    --gen_length 128 \
    --steps 128 \
    --num_tests 3 \
    --device cuda

echo ""
echo "Sanity check complete!"

