#!/usr/bin/env python3
"""
Find optimal UMR hyperparameters using the paper's selection criterion (Section VII):
    For each beta in {0.01, 0.05, 0.10, 0.15}:
        Filter to configurations where P(z >= 4) >= 1 - beta  (completeness constraint)
        Among those, pick the one with highest average GPT-4 judge score

Detection criterion: z_score >= 4  (same as paper's green-list threshold)
z_score is stored per-result in watermark_metrics['z_score'].

Usage:
    python find_optimal_umr_hyperparams.py
"""
import json
import re
import math
from pathlib import Path
from collections import defaultdict

OUTDIR = Path("water-bench-results/json-outputs")
GPT4_DIR = OUTDIR / "gpt4-outputs"
Z_THRESH = 4.0
BETAS = [0.01, 0.05, 0.10, 0.15]


def parse_params(stem):
    """Extract (gamma, delta, tend) from filename stem like umr_gamma=0.9_delta=4.0_tend=300_sampled_100"""
    m = re.search(r"gamma=([\d.]+)_delta=([\d.]+)_tend=(\d+)", stem)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def load_config(gen_path, gpt4_path):
    """Load generation file (for z-scores) and GPT-4 eval file (for judge scores)."""
    gen = json.load(open(gen_path))
    gpt4 = json.load(open(gpt4_path))

    z_scores = []
    judge_scores = []

    # Build prompt_id → gpt4 result map for alignment
    gpt4_map = {r["prompt_id"]: r for r in gpt4["results"]}

    for r in gen["results"]:
        pid = r["prompt_id"]
        wm = r.get("watermark_metrics", {})
        z = wm.get("z_score")
        if z is None or not math.isfinite(z):
            continue
        z_scores.append(z)

        gpt4_r = gpt4_map.get(pid)
        if gpt4_r is not None:
            avg = gpt4_r.get("average_score")
            if avg is not None and math.isfinite(avg):
                judge_scores.append(avg)

    if not z_scores:
        return None

    n = len(z_scores)
    completeness = sum(1 for z in z_scores if z >= Z_THRESH) / n
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else float("nan")
    avg_z = sum(z_scores) / n
    perplexity = gen.get("average_perplexity", float("nan"))

    return {
        "completeness": completeness,
        "avg_judge": avg_judge,
        "avg_z": avg_z,
        "perplexity": perplexity,
        "n": n,
    }


def main():
    gen_files = sorted(OUTDIR.glob("umr_*.json"))
    configs = {}

    print(f"Loading {len(gen_files)} UMR configurations...")
    missing_gpt4 = []
    for gf in gen_files:
        params = parse_params(gf.stem)
        if params is None:
            continue
        gpt4_path = GPT4_DIR / (gf.stem + "_gpt4_eval.json")
        if not gpt4_path.exists():
            missing_gpt4.append(gf.name)
            continue
        result = load_config(gf, gpt4_path)
        if result is not None:
            configs[params] = result

    if missing_gpt4:
        print(f"  WARNING: {len(missing_gpt4)} files missing GPT-4 eval (skipped)")

    print(f"  Loaded {len(configs)} configurations.\n")

    # -------------------------------------------------------------------------
    # Full table sorted by completeness desc, then judge score desc
    # -------------------------------------------------------------------------
    print(f"{'gamma':>6} {'delta':>6} {'tend':>5} {'Comp%':>7} {'AvgZ':>7} {'Judge':>7} {'PPL':>8}")
    print("-" * 55)
    for (gamma, delta, tend), v in sorted(configs.items(),
            key=lambda x: (-x[1]["completeness"], -x[1]["avg_judge"])):
        print(f"  {gamma:>5.2f}  {delta:>5.1f}  {tend:>4d}  "
              f"{v['completeness']:>6.1%}  {v['avg_z']:>6.2f}  "
              f"{v['avg_judge']:>6.3f}  {v['perplexity']:>7.3f}")

    # -------------------------------------------------------------------------
    # Paper's selection criterion
    # -------------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"Paper's selection criterion: max judge score s.t. P(z≥{Z_THRESH}) ≥ 1−β")
    print(f"{'='*65}")

    for beta in BETAS:
        threshold = 1.0 - beta
        eligible = {k: v for k, v in configs.items() if v["completeness"] >= threshold}
        if not eligible:
            print(f"\nβ={beta:.2f}: No configuration meets P(z≥{Z_THRESH}) ≥ {threshold:.0%}")
            continue
        best_params = max(eligible, key=lambda k: eligible[k]["avg_judge"])
        best = eligible[best_params]
        gamma, delta, tend = best_params
        print(f"\nβ={beta:.2f}  (completeness ≥ {threshold:.0%}):")
        print(f"  γ*={gamma}, δ*={delta}, t_end={tend}")
        print(f"  Completeness={best['completeness']:.1%}, "
              f"Avg z={best['avg_z']:.2f}, "
              f"Judge={best['avg_judge']:.3f}, "
              f"PPL={best['perplexity']:.3f}")
        print(f"  (from {len(eligible)} eligible configurations)")

    # -------------------------------------------------------------------------
    # Best overall (no completeness constraint)
    # -------------------------------------------------------------------------
    if configs:
        best_judge = max(configs, key=lambda k: configs[k]["avg_judge"])
        best_comp  = max(configs, key=lambda k: configs[k]["completeness"])
        print(f"\n{'='*65}")
        print("Best judge score (no constraint):")
        g, d, t = best_judge
        v = configs[best_judge]
        print(f"  γ={g}, δ={d}, t_end={t} → Judge={v['avg_judge']:.3f}, "
              f"Completeness={v['completeness']:.1%}, PPL={v['perplexity']:.3f}")
        print("Best completeness (no constraint):")
        g, d, t = best_comp
        v = configs[best_comp]
        print(f"  γ={g}, δ={d}, t_end={t} → Completeness={v['completeness']:.1%}, "
              f"Judge={v['avg_judge']:.3f}, PPL={v['perplexity']:.3f}")


if __name__ == "__main__":
    main()
