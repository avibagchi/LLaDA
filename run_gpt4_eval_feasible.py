#!/usr/bin/env python3
"""
Run GPT-4 judge evaluation on all feasible ablation configs.
Feasible = P(z >= 4) >= min_detect_pct on the 100-prompt ablation set.
Skips files that already have a corresponding _gpt4_eval.json output.

Usage:
    python run_gpt4_eval_feasible.py
    python run_gpt4_eval_feasible.py --min_detect 90  # stricter
"""
import json
import glob
import subprocess
import sys
import argparse
from pathlib import Path

ABLATION_DIR = Path("water-bench-results/json-outputs")
GPT4_OUT_DIR = ABLATION_DIR / "gpt4-outputs"
METHODS = ["cdmark", "dmark", "lrdwm"]  # umr excluded (not detectable)


def get_detect_pct(fp, z_thresh=4.0):
    try:
        d = json.load(open(fp))
        zscores = [r["watermark_metrics"]["z_score"]
                   for r in d["results"] if "z_score" in r.get("watermark_metrics", {})]
        if not zscores:
            return 0.0
        return 100.0 * sum(z >= z_thresh for z in zscores) / len(zscores)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_detect", type=float, default=85.0,
                        help="Minimum P(z>=4) percent to include (default: 85)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would run without running it")
    args = parser.parse_args()

    GPT4_OUT_DIR.mkdir(parents=True, exist_ok=True)

    feasible = []
    for fp in sorted(ABLATION_DIR.glob("*.json")):
        method = next((m for m in METHODS if fp.name.startswith(m)), None)
        if method is None:
            continue
        if get_detect_pct(fp) >= args.min_detect:
            feasible.append(fp)

    print(f"Found {len(feasible)} feasible configs (P(z>=4) >= {args.min_detect}%)")

    for i, fp in enumerate(feasible, 1):
        out_fp = GPT4_OUT_DIR / (fp.stem + "_gpt4_eval.json")
        if out_fp.exists():
            print(f"[{i}/{len(feasible)}] Skip (exists): {fp.name}")
            continue
        print(f"[{i}/{len(feasible)}] Evaluating: {fp.name}")
        if args.dry_run:
            continue
        result = subprocess.run(
            [sys.executable, "evaluate_with_gpt4.py", str(fp),
             "--output_dir", str(GPT4_OUT_DIR)],
            capture_output=False
        )
        if result.returncode != 0:
            print(f"  WARNING: evaluate_with_gpt4.py returned {result.returncode}")

    print("Done.")


if __name__ == "__main__":
    main()
