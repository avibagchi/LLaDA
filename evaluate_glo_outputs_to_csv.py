#!/usr/bin/env python3
"""
Batch GPT-4 evaluation for all Gloaguen JSON outputs, with CSV summary.

For each input JSON file, this script:
1) Runs GPT-4 scoring on each prompt result (style/consistency/accuracy/ethics)
2) Computes per-file averages
3) Writes one CSV row per file (includes count of prompts with binomial_z_score >= threshold)

Default input directory:
  water-bench-results/json-outputs/glo-outputs
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from evaluate_with_gpt4 import evaluate_all_results, calculate_metrics


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all JSON files in glo-outputs with GPT-4 and write a CSV "
            "with avg perplexity, avg GPT-4 category scores, average GPT score, "
            "average binomial_z_score, and count of prompts with binomial_z_score "
            ">= z-threshold."
        )
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=4.0,
        help=(
            "Count prompts where watermark_metrics.binomial_z_score is >= this "
            "(default: 4). Missing values are not counted."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="water-bench-results/json-outputs/glo-outputs",
        help="Directory containing input JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="water-bench-results/json-outputs/glo-outputs_gpt4_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files to process (default: all).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel GPT-4 workers per file (default: 8).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (default: OPENAI_API_KEY from env/.env).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY, provide .env, or use --api_key."
        )

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if args.max_files is not None:
        json_files = json_files[: args.max_files]

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    client = openai.OpenAI(api_key=api_key)

    rows: list[dict[str, Any]] = []

    for json_file in tqdm(json_files, desc="Processing files"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results:
            continue

        evaluated_results = evaluate_all_results(
            client=client,
            results=results,
            max_workers=args.max_workers,
        )
        metrics = calculate_metrics(evaluated_results)
        cat = metrics.get("category_averages", {})

        binomial_scores = [
            _as_float_or_none(r.get("watermark_metrics", {}).get("binomial_z_score"))
            for r in evaluated_results
        ]
        binomial_scores = [v for v in binomial_scores if v is not None]

        n_prompts_z_ge = sum(
            1
            for r in evaluated_results
            if (z := _as_float_or_none(r.get("watermark_metrics", {}).get("binomial_z_score")))
            is not None
            and z >= args.z_threshold
        )

        row = {
            "file": json_file.name,
            "avg_perplexity": _as_float_or_none(metrics.get("average_perplexity")),
            "avg_style": _as_float_or_none(cat.get("style (setting ethics aside)")),
            "avg_consistency": _as_float_or_none(
                cat.get("consistency (setting ethics aside)")
            ),
            "avg_accuracy": _as_float_or_none(cat.get("accuracy (setting ethics aside)")),
            "avg_ethics": _as_float_or_none(cat.get("ethics")),
            "average_gpt_score": _as_float_or_none(metrics.get("overall_average_score")),
            "average_binomial_z_score": _safe_mean(binomial_scores),
            "n_prompts_binomial_z_ge": n_prompts_z_ge,
        }
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "avg_perplexity",
        "avg_style",
        "avg_consistency",
        "avg_accuracy",
        "avg_ethics",
        "average_gpt_score",
        "average_binomial_z_score",
        "n_prompts_binomial_z_ge",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV: {output_csv}")
    print(f"Files processed: {len(rows)}")
    print(f"n_prompts_binomial_z_ge uses binomial_z_score >= {args.z_threshold}")


if __name__ == "__main__":
    main()
