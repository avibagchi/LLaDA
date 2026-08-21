#!/usr/bin/env python3
"""
Run GPT-4 judge on the 7 final 500-prompt evaluation outputs.
Skips any file that already has a corresponding _gpt4_eval.json.

Usage:
    python run_gpt4_eval_final.py
"""
import subprocess, sys
from pathlib import Path

OUTDIR = Path("water-bench-results/json-outputs")
GPT4_DIR = OUTDIR / "gpt4-outputs"

FINAL_FILES = [
    "cdmark_final_gamma=0.9_delta=2.0_tend=40_500.json",
    "cdmark_final_gamma=0.5_delta=4.0_tend=20_500.json",
    "cdmark_final_gamma=0.25_delta=4.0_tend=20_500.json",
    "dmark_final_gamma=0.1_delta=4.0_tend=300_500.json",
    "dmark_final_gamma=0.25_delta=4.0_tend=80_500.json",
    "dmark_final_gamma=0.5_delta=8.0_tend=160_500.json",
    "lrdwm_final_gamma=0.9_delta=4.0_tend=300_500.json",
]

GPT4_DIR.mkdir(parents=True, exist_ok=True)

for i, fname in enumerate(FINAL_FILES, 1):
    fp = OUTDIR / fname
    out_fp = GPT4_DIR / (fp.stem + "_gpt4_eval.json")
    if out_fp.exists():
        print(f"[{i}/{len(FINAL_FILES)}] Skip (exists): {fname}")
        continue
    if not fp.exists():
        print(f"[{i}/{len(FINAL_FILES)}] MISSING: {fname} — run run_final_evals.sh first")
        continue
    print(f"[{i}/{len(FINAL_FILES)}] Evaluating: {fname}")
    result = subprocess.run(
        [sys.executable, "evaluate_with_gpt4.py", str(fp),
         "--output_dir", str(GPT4_DIR)],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  WARNING: returned {result.returncode}")

print("Done.")
