#!/usr/bin/env python3
"""
Run GPT-4 judge on all 175 UMR ablation outputs.
Skips files that already have a corresponding _gpt4_eval.json.

Usage:
    conda run -n llada python run_gpt4_eval_umr.py
    conda run -n llada python run_gpt4_eval_umr.py --max_prompts 50
"""
import subprocess
import sys
import time
import os
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

OUTDIR = Path("water-bench-results/json-outputs")
GPT4_DIR = OUTDIR / "gpt4-outputs"

GPT4_DIR.mkdir(parents=True, exist_ok=True)

umr_files = sorted(OUTDIR.glob("umr_*.json"))
total = len(umr_files)

if total == 0:
    print("No umr_*.json files found in", OUTDIR)
    sys.exit(1)

# Pass through any extra args (e.g. --max_prompts 50) to evaluate_with_gpt4.py
extra_args = sys.argv[1:]

print(f"Found {total} UMR files.")
already_done = sum(1 for f in umr_files if (GPT4_DIR / (f.stem + "_gpt4_eval.json")).exists())
print(f"Already evaluated: {already_done} / {total}\n")

skipped = 0
succeeded = 0
failed = 0
start_time = time.time()

for i, fp in enumerate(umr_files, 1):
    out_fp = GPT4_DIR / (fp.stem + "_gpt4_eval.json")

    if out_fp.exists():
        skipped += 1
        print(f"[{i:3d}/{total}] SKIP  {fp.name}")
        continue

    remaining = total - i + 1 - skipped
    elapsed = time.time() - start_time
    avg_per_file = elapsed / max(succeeded + failed, 1)
    eta_s = avg_per_file * remaining
    eta_str = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s" if succeeded + failed > 0 else "?"

    print(f"[{i:3d}/{total}] RUN   {fp.name}  (ETA {eta_str})", flush=True)

    result = subprocess.run(
        [sys.executable, "evaluate_with_gpt4.py", str(fp),
         "--output_dir", str(GPT4_DIR)] + extra_args,
        capture_output=False,
    )

    if result.returncode == 0:
        succeeded += 1
        print(f"         -> done ({succeeded} completed so far)\n")
    else:
        failed += 1
        print(f"         -> WARNING: returned {result.returncode}\n")

total_elapsed = time.time() - start_time
print("=" * 60)
print(f"Finished. {succeeded} succeeded, {skipped} skipped, {failed} failed")
print(f"Total time: {int(total_elapsed // 60)}m{int(total_elapsed % 60):02d}s")
print(f"Output dir: {GPT4_DIR}")
