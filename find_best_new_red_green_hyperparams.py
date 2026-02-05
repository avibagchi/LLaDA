#!/usr/bin/env python3
"""
Find the red-green list hyperparameter combination (gamma, delta, steps) that:
  - has average z-score >= 4, and
  - among those, maximizes the average GPT-4 score.

Uses the GPT-4 evaluation summary CSV (from create_gpt4_summary_csv), which has
filename, gamma, delta, steps, overall_avg_score, avg_z_score, etc.
"""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Among configs with avg z-score >= 4, find (gamma, delta, steps) that maximizes average GPT-4 score. Uses GPT-4 evaluation summary CSV."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        nargs="?",
        default="gpt4_evaluation_summary.csv",
        help="Path to GPT-4 evaluation summary CSV (from create_gpt4_summary_csv). Default: gpt4_evaluation_summary.csv",
    )
    parser.add_argument(
        "--min_z",
        type=float,
        default=4.0,
        help="Minimum average z-score required. Default: 4",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gamma = row.get("gamma", "").strip()
                delta = row.get("delta", "").strip()
                steps = row.get("steps", "").strip()
                avg_z = float(row.get("avg_z_score", 0))
                overall = float(row.get("overall_avg_score", 0))
            except (TypeError, ValueError):
                continue
            if gamma == "" or delta == "" or steps == "":
                continue
            if avg_z < args.min_z:
                continue
            rows.append(
                {
                    "gamma": gamma,
                    "delta": delta,
                    "steps": steps,
                    "avg_z_score": avg_z,
                    "overall_avg_score": overall,
                    "filename": row.get("filename", ""),
                }
            )

    if not rows:
        print(f"No row in CSV with avg_z_score >= {args.min_z} found.")
        return

    best = max(rows, key=lambda r: r["overall_avg_score"])
    print("Among configurations with average z-score >= {}:".format(args.min_z))
    print("  Best hyperparameters (maximize average GPT-4 score):")
    print("    gamma   = {}".format(best["gamma"]))
    print("    delta   = {}".format(best["delta"]))
    print("    steps   = {}".format(best["steps"]))
    print("  Metrics:")
    print("    avg_z_score       = {:.4f}".format(best["avg_z_score"]))
    print("    overall_avg_score = {:.4f}".format(best["overall_avg_score"]))
    print("  File: {}".format(best["filename"]))


if __name__ == "__main__":
    main()
