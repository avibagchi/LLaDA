#!/bin/bash
# Run 500-prompt final evaluations for the optimal hyperparameter configs.
# Configs selected by find_optimal_baseline_hyperparams.py (maximize GPT-4 score
# subject to P(z>=4) >= 1-beta on the 100-prompt ablation set).
#
# Usage:
#   bash run_final_evals.sh
#
# Requires: llada conda env, cuda:3

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DEVICE="cuda:3"
JSONL="water-bench-sampled_500_seed42.jsonl"
OUTDIR="water-bench-results/json-outputs"
PY="conda run -n llada python eval_waterbench.py"
COMMON="--device $DEVICE --jsonl_file $JSONL --output_dir $OUTDIR \
        --temperature 0.5 --block_length 25 --steps 300 --gen_length 300"

echo "=== Final 500-prompt evaluations ==="
echo "Output dir: $OUTDIR"
echo ""

# ---- CDMArk (m=1) --------------------------------------------------------
# beta=0.15 optimal
$PY $COMMON \
  --watermark_type cdmark --gamma 0.9 --amplification 2.0 \
  --cdmark_watermark_steps 40 \
  --output_file cdmark_final_gamma=0.9_delta=2.0_tend=40_500.json
echo "[1/7] CDMArk gamma=0.9 delta=2.0 tend=40 done"

# beta=0.10 / beta=0.05 optimal
$PY $COMMON \
  --watermark_type cdmark --gamma 0.5 --amplification 4.0 \
  --cdmark_watermark_steps 20 \
  --output_file cdmark_final_gamma=0.5_delta=4.0_tend=20_500.json
echo "[2/7] CDMArk gamma=0.5 delta=4.0 tend=20 done"

# beta=0.01 optimal
$PY $COMMON \
  --watermark_type cdmark --gamma 0.25 --amplification 4.0 \
  --cdmark_watermark_steps 20 \
  --output_file cdmark_final_gamma=0.25_delta=4.0_tend=20_500.json
echo "[3/7] CDMArk gamma=0.25 delta=4.0 tend=20 done"

# ---- DMark ---------------------------------------------------------------
# beta=0.15 / beta=0.10 optimal
$PY $COMMON \
  --watermark_type dmark --gamma 0.1 --amplification 4.0 \
  --dmark_watermark_steps 300 \
  --output_file dmark_final_gamma=0.1_delta=4.0_tend=300_500.json
echo "[4/7] DMark gamma=0.1 delta=4.0 tend=300 done"

# beta=0.05 optimal
$PY $COMMON \
  --watermark_type dmark --gamma 0.25 --amplification 4.0 \
  --dmark_watermark_steps 80 \
  --output_file dmark_final_gamma=0.25_delta=4.0_tend=80_500.json
echo "[5/7] DMark gamma=0.25 delta=4.0 tend=80 done"

# beta=0.01 optimal
$PY $COMMON \
  --watermark_type dmark --gamma 0.5 --amplification 8.0 \
  --dmark_watermark_steps 160 \
  --output_file dmark_final_gamma=0.5_delta=8.0_tend=160_500.json
echo "[6/7] DMark gamma=0.5 delta=8.0 tend=160 done"

# ---- LR-DWM (all betas same config) -------------------------------------
$PY $COMMON \
  --watermark_type lrdwm --gamma 0.9 --amplification 4.0 \
  --lrdwm_watermark_steps 300 \
  --output_file lrdwm_final_gamma=0.9_delta=4.0_tend=300_500.json
echo "[7/7] LR-DWM gamma=0.9 delta=4.0 tend=300 done"

echo ""
echo "All done. Next: run GPT-4 eval on these 7 output files."
echo "  conda run -n llada python run_gpt4_eval_final.py"
