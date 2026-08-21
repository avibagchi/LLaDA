#!/usr/bin/env python3
"""
Find optimal hyperparameters for each new baseline method (cdmark, dmark, lrdwm, dgmark).

For each method and each beta in {0.01, 0.05, 0.10, 0.15}:
  Select (gamma, delta, tend) that maximizes avg GPT-4 judge score
  subject to P(z >= 4) >= 1 - beta on the 100-prompt held-out set.

Requires:
  - Ablation JSONs in water-bench-results/json-outputs/
  - GPT-4 eval JSONs in water-bench-results/json-outputs/gpt4-outputs/

Usage:
    python find_optimal_baseline_hyperparams.py
"""
import json
import glob
import re
from pathlib import Path

ABLATION_DIR = Path("water-bench-results/json-outputs")
GPT4_DIR = ABLATION_DIR / "gpt4-outputs"
METHODS = ["cdmark", "dmark", "lrdwm", "dgmark", "umr"]
BETA_VALUES = [0.15, 0.10, 0.05, 0.01]
Z_THRESH = 4.0


def parse_params(name):
    gamma = re.search(r'gamma=([0-9.]+)', name)
    delta = re.search(r'delta=([0-9.]+)', name)
    tend  = re.search(r'tend=([0-9]+)', name)
    return (
        float(gamma.group(1)) if gamma else None,
        float(delta.group(1)) if delta else None,
        int(tend.group(1)) if tend else None,
    )


def load_zscores(fp):
    d = json.load(open(fp))
    return [r["watermark_metrics"]["z_score"]
            for r in d["results"] if "z_score" in r.get("watermark_metrics", {})]


def load_gpt4_score(fp):
    """Return overall average GPT-4 judge score from a gpt4_eval JSON, or None."""
    try:
        d = json.load(open(fp))
    except Exception:
        return None
    # Try summary field first
    ev = d.get("gpt4_evaluation_metrics", {})
    if ev.get("overall_average_score") is not None:
        return float(ev["overall_average_score"])
    # Fall back to per-result average
    scores = []
    for r in d.get("results", []):
        s = r.get("average_score")
        if s is not None:
            scores.append(float(s))
        else:
            sc = r.get("scores", {})
            if sc:
                scores.append(sum(sc.values()) / len(sc))
    return sum(scores) / len(scores) if scores else None


def main():
    rows = []
    for fp in sorted(ABLATION_DIR.glob("*.json")):
        method = next((m for m in METHODS if fp.name.startswith(m)), None)
        if method is None:
            continue
        gamma, delta, tend = parse_params(fp.name)

        zscores = load_zscores(fp)
        if not zscores:
            continue
        detect_pct = 100.0 * sum(z >= Z_THRESH for z in zscores) / len(zscores)

        # Look for matching GPT-4 eval file
        gpt4_fp = GPT4_DIR / (fp.stem + "_gpt4_eval.json")
        gpt4_score = load_gpt4_score(gpt4_fp) if gpt4_fp.exists() else None

        rows.append({
            "method": method,
            "gamma": gamma,
            "delta": delta,
            "tend": tend,
            "detect_pct": detect_pct,
            "gpt4_score": gpt4_score,
            "file": fp.name,
        })

    print(f"Loaded {len(rows)} configs total")
    print()

    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        print(f"{'='*60}")
        print(f"Method: {method}  ({len(method_rows)} configs)")
        print(f"{'='*60}")

        with_gpt4 = [r for r in method_rows if r["gpt4_score"] is not None]
        print(f"  GPT-4 eval available for {len(with_gpt4)} / {len(method_rows)} configs")
        print()

        for beta in BETA_VALUES:
            threshold_pct = (1 - beta) * 100
            eligible = [r for r in with_gpt4 if r["detect_pct"] >= threshold_pct]
            if not eligible:
                # Report best detect even if no GPT-4 scores
                eligible_no_gpt4 = [r for r in method_rows if r["detect_pct"] >= threshold_pct]
                print(f"  beta={beta:.2f} (P(z>=4)>={threshold_pct:.0f}%): "
                      f"No eligible configs with GPT-4 scores. "
                      f"{len(eligible_no_gpt4)} configs meet z-score threshold.")
                continue
            best = max(eligible, key=lambda r: r["gpt4_score"])
            print(f"  beta={beta:.2f} (P(z>=4)>={threshold_pct:.0f}%): "
                  f"gamma={best['gamma']}, delta={best['delta']}, tend={best['tend']}  "
                  f"[detect={best['detect_pct']:.0f}%, gpt4={best['gpt4_score']:.3f}, "
                  f"n={len(eligible)} eligible]")
        print()


if __name__ == "__main__":
    main()
