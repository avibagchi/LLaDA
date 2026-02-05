#!/usr/bin/env python3
"""
Find optimal red-green list hyperparameters at different detectability thresholds.

Detectability = percentage of prompts with z-score >= 4.
For each threshold (85%, 90%, 95%, 99%), finds the (gamma, delta, steps) that
maximizes the average GPT-4-as-judge score among configs meeting that detectability.

Input: Directory of GPT-4 evaluation JSON files (or generated JSON files that have
       been run through evaluate_with_gpt4.py). Files must have per-result
       watermark_metrics.z_score and GPT-4 scores. Hyperparameters are parsed from
       the filename (e.g. run_gamma=0.1_delta=0.5_steps=10_...).
"""
import json
import argparse
import csv
import re
from pathlib import Path
from glob import glob


# Detectability thresholds (percent)
DETECTABILITY_THRESHOLDS = [85, 90, 95, 99]


def parse_hyperparams_from_filename(filename):
    """Extract gamma, delta, steps from filename. Returns (gamma, delta, steps) or (None, None, None)."""
    name = Path(filename).name
    gamma_match = re.search(r'gamma=([0-9.]+)', name)
    delta_match = re.search(r'delta=([0-9.]+)', name)
    steps_match = re.search(r'steps=([0-9]+)', name)
    gamma = float(gamma_match.group(1)) if gamma_match else None
    delta = float(delta_match.group(1)) if delta_match else None
    steps = int(steps_match.group(1)) if steps_match else None
    return gamma, delta, steps


def load_metrics_from_json(filepath, z_threshold=4.0):
    """
    Load detectability (% prompts with z_score >= z_threshold) and average GPT-4 score from a JSON file.
    File can be either a GPT-4 eval output (has results[].scores / gpt4_evaluation) or
    a generated output (has results[].watermark_metrics.z_score only; then avg_gpt4 is None).
    Returns (detectability_pct, avg_gpt4_score, num_prompts) or (None, None, 0) on error.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None, None, 0

    results = data.get('results', [])
    if not results:
        return None, None, 0

    # Detectability: % of prompts with z_score >= z_threshold
    count_above = 0
    for r in results:
        wm = r.get('watermark_metrics', {})
        z = wm.get('z_score') or wm.get('normalized_score')
        if z is not None and z >= z_threshold:
            count_above += 1
    detectability_pct = 100.0 * count_above / len(results)

    # Average GPT-4 score
    gpt4_metrics = data.get('gpt4_evaluation_metrics', {})
    overall = gpt4_metrics.get('overall_average_score')
    if overall is not None:
        avg_gpt4 = overall
    else:
        # Compute from per-result scores
        scores = []
        for r in results:
            s = r.get('average_score')
            if s is not None:
                scores.append(s)
            else:
                sc = r.get('scores', {})
                if sc:
                    scores.append(sum(sc.values()) / len(sc))
        avg_gpt4 = sum(scores) / len(scores) if scores else None
    return detectability_pct, avg_gpt4, len(results)


def load_summary_csv(csv_path):
    """Load CSV from create_gpt4_summary_csv; return dict filename -> overall_avg_score."""
    scores_by_file = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get('filename', '')
            try:
                scores_by_file[fn] = float(row.get('overall_avg_score', 0))
            except (TypeError, ValueError):
                pass
    return scores_by_file


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal red-green hyperparameters at different detectability thresholds.'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        nargs='?',
        default='water-bench-results/json-outputs/new-red-green-list',
        help='Directory containing JSON files with gamma/delta/steps in filename (GPT-4 eval or generated; see --summary_csv)'
    )
    parser.add_argument(
        '--summary_csv',
        type=str,
        default=None,
        help='Optional: path to CSV from create_gpt4_summary_csv. If provided, avg GPT-4 score is taken from CSV (matched by filename: JSON base + "_gpt4_eval.json"). Detectability still from JSON in input_dir.'
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=DETECTABILITY_THRESHOLDS,
        help=f'Detectability thresholds (%%). Default: {DETECTABILITY_THRESHOLDS}'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=4.0,
        help='Z-score threshold for detectability (prompts with z >= this count as detected). Default: 4'
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}")
        return

    csv_scores = {}
    if args.summary_csv:
        csv_path = Path(args.summary_csv)
        if csv_path.is_file():
            csv_scores = load_summary_csv(csv_path)
            print(f"Loaded GPT-4 scores for {len(csv_scores)} files from {args.summary_csv}")
        else:
            print(f"Warning: --summary_csv file not found: {args.summary_csv}")

    # Gather all JSON files that look like parameter sweeps (have gamma=, delta=, steps= in name)
    pattern = str(input_dir / '*.json')
    all_files = glob(pattern)
    rows = []
    for fp in all_files:
        gamma, delta, steps = parse_hyperparams_from_filename(fp)
        if gamma is None and delta is None and steps is None:
            continue
        detectability_pct, avg_gpt4_json, n = load_metrics_from_json(fp, z_threshold=args.z_threshold)
        if detectability_pct is None:
            continue
        # Prefer GPT-4 score from CSV if we have a match (for new-red-green-list generated JSONs)
        gpt4_key = Path(fp).stem + '_gpt4_eval.json'
        if csv_scores and gpt4_key in csv_scores:
            avg_gpt4 = csv_scores[gpt4_key]
        else:
            avg_gpt4 = avg_gpt4_json
        if avg_gpt4 is None:
            print(f"Warning: No GPT-4 scores for {Path(fp).name}; skipping.")
            continue
        rows.append({
            'file': fp,
            'gamma': gamma,
            'delta': delta,
            'steps': steps,
            'detectability_pct': detectability_pct,
            'avg_gpt4': avg_gpt4,
            'n': n,
        })

    if not rows:
        print("No valid files with (gamma, delta, steps) and GPT-4 scores found.")
        return

    print(f"Loaded {len(rows)} configurations from {input_dir}")
    print()

    # For each detectability threshold, find config that maximizes avg GPT-4 score
    print("Optimal hyperparameters (gamma, delta, steps) by detectability threshold:")
    print("  Detectability = percentage of prompts with z-score >= 4")
    print()

    for thresh in sorted(args.thresholds):
        eligible = [r for r in rows if r['detectability_pct'] >= thresh]
        if not eligible:
            best = None
            msg = f"No configuration with detectability >= {thresh}%"
        else:
            best = max(eligible, key=lambda r: r['avg_gpt4'])
            msg = (
                f"gamma={best['gamma']}, delta={best['delta']}, steps={best['steps']}  "
                f"(detectability={best['detectability_pct']:.1f}%, avg GPT-4 score={best['avg_gpt4']:.4f}, n={best['n']})"
            )
        print(f"  At {thresh}% detectability threshold: {msg}")
        if best:
            print(f"    File: {Path(best['file']).name}")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
