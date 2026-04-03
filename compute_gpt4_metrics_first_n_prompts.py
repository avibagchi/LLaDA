#!/usr/bin/env python3
"""
Compute gpt4_evaluation_metrics for only the first N prompts of a GPT-4 eval JSON file.
Also computes detectability: fraction of prompts where a chosen watermark stat
(default z_score, or binomial_z_score via --watermark-detect-stat) is >= threshold.

Input: JSON file like water-bench-results/json-outputs/gpt4-outputs/2000_rg_85_thresh_gpt4_eval.json
Output: Prints (and optionally writes) gpt4_evaluation_metrics for the first N prompts,
        plus detectability_pct (z_score >= 4).
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute gpt4_evaluation_metrics from a list of result dicts (same logic as evaluate_with_gpt4.py)."""
    total = len(results)
    score_categories = [
        "style (setting ethics aside)",
        "consistency (setting ethics aside)",
        "accuracy (setting ethics aside)",
        "ethics",
    ]
    category_averages = {}
    for category in score_categories:
        category_scores = [
            r.get("scores", {}).get(category, 0)
            for r in results
            if r.get("scores", {}).get(category) is not None
        ]
        if category_scores:
            category_averages[category] = sum(category_scores) / len(category_scores)
        else:
            category_averages[category] = 0.0

    average_scores = [
        r.get("average_score", 0.0)
        for r in results
        if r.get("average_score") is not None
    ]
    overall_average_score = (
        sum(average_scores) / len(average_scores) if average_scores else 0.0
    )

    perplexities = [
        r.get("perplexity")
        for r in results
        if r.get("perplexity") is not None and r.get("perplexity") <= 100
    ]
    avg_perplexity = (
        sum(perplexities) / len(perplexities) if perplexities else None
    )

    return {
        "total_prompts": total,
        "category_averages": category_averages,
        "overall_average_score": overall_average_score,
        "average_perplexity": avg_perplexity,
        "total_with_perplexity": len(perplexities),
    }


def detectability_pct(
    results: List[Dict[str, Any]],
    z_threshold: float = 4.0,
    stat_key: str = "z_score",
) -> float:
    """Percentage of prompts with chosen watermark stat >= z_threshold."""
    if not results:
        return 0.0
    count = 0
    for r in results:
        wm = r.get("watermark_metrics", {})
        if stat_key == "z_score":
            z = wm.get("z_score")
            if z is None:
                z = wm.get("normalized_score")
        else:
            z = wm.get(stat_key)
        if z is not None and z >= z_threshold:
            count += 1
    return 100.0 * count / len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compute gpt4_evaluation_metrics for the first N prompts and detectability (z_score >= 4)."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to GPT-4 evaluation JSON file (e.g. .../2000_rg_85_thresh_gpt4_eval.json)",
    )
    parser.add_argument(
        "-n",
        "--num_prompts",
        type=int,
        default=1000,
        help="Number of prompts to use (first N). Default: 500",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=4.0,
        help="Z-score threshold for detectability. Default: 4",
    )
    parser.add_argument(
        "--watermark-detect-stat",
        type=str,
        choices=["z_score", "binomial_z_score"],
        default="z_score",
        help=(
            "Which watermark_metrics field to compare to --z-threshold. "
            "Use binomial_z_score for Gloaguen/Bernoulli (standardized count z); "
            "default z_score also falls back to normalized_score for Aaronson."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional: write metrics JSON to this path (includes detectability_pct)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Error: File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    n = min(args.num_prompts, len(results))
    first_n = results[:n]

    if not first_n:
        print("No results in file.")
        return

    metrics = calculate_metrics(first_n)
    pct = detectability_pct(
        first_n,
        z_threshold=args.z_threshold,
        stat_key=args.watermark_detect_stat,
    )

    # Build output in same shape as gpt4_evaluation_metrics + detectability
    out = {
        "total_prompts": metrics["total_prompts"],
        "category_averages": metrics["category_averages"],
        "overall_average_score": metrics["overall_average_score"],
        "average_perplexity": metrics["average_perplexity"],
        "total_with_perplexity": metrics["total_with_perplexity"],
        "detectability_pct": round(pct, 2),
        "z_threshold": args.z_threshold,
        "watermark_detect_stat": args.watermark_detect_stat,
    }

    print(f"File: {input_path.name}")
    print(f"Using first {n} prompts (of {len(results)} total)")
    print()
    print("gpt4_evaluation_metrics (first {} prompts):".format(n))
    print(json.dumps({
        "total_prompts": metrics["total_prompts"],
        "category_averages": metrics["category_averages"],
        "overall_average_score": metrics["overall_average_score"],
        "average_perplexity": metrics["average_perplexity"],
        "total_with_perplexity": metrics["total_with_perplexity"],
    }, indent=2))
    print()
    print(
        f"Detectability (%% of prompts with {args.watermark_detect_stat} >= {args.z_threshold}): "
        f"{pct:.2f}%"
    )
    print(f"  (detectability_pct: {out['detectability_pct']})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote metrics to: {out_path}")


if __name__ == "__main__":
    main()
