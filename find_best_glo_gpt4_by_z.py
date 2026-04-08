#!/usr/bin/env python3
"""
1) Scan all WaterBench JSON files in glo-outputs and count prompts where
   watermark_metrics.binomial_z_score >= z_threshold (default: 4).

2) Merge that count into glo-outputs_gpt4_summary.csv (match on ``file``).

3) Write a merged CSV.

4) For each target percentage (default: 99, 95, 90, 85) of prompts, using a
   fixed cohort size (default: 100), print the file that maximizes
   average_gpt_score among rows with count >= ceil(n_prompts * pct / 100).

Run from the LLaDA directory (or pass absolute paths).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _float_or_none(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_or_none(s: str | None) -> int | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def count_binomial_z_ge(
    json_path: Path, z_threshold: float
) -> tuple[int, int]:
    """
    Returns (n_ge_threshold, n_results) for one eval JSON.
    Only counts prompts with a numeric binomial_z_score.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    n_ge = 0
    for r in results:
        wm = r.get("watermark_metrics") or {}
        z = wm.get("binomial_z_score")
        if z is None:
            continue
        try:
            zf = float(z)
        except (TypeError, ValueError):
            continue
        if zf >= z_threshold:
            n_ge += 1
    return n_ge, len(results)


def scan_folder(
    json_dir: Path, z_threshold: float
) -> dict[str, tuple[int, int]]:
    """Map filename -> (n_ge, n_results)."""
    out: dict[str, tuple[int, int]] = {}
    for p in sorted(json_dir.glob("*.json")):
        try:
            n_ge, n_tot = count_binomial_z_ge(p, z_threshold)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skip {p.name}: {e}")
            continue
        out[p.name] = (n_ge, n_tot)
    return out


def best_gpt_at_pct(
    rows: list[dict[str, str]],
    n_prompts: int,
    pct: float,
    gpt_col: str,
) -> tuple[float | None, dict[str, str] | None, int]:
    """Maximize gpt among rows with n_prompts_binomial_z_ge >= required."""
    required = math.ceil(n_prompts * pct / 100.0)
    best_gpt: float | None = None
    best_row: dict[str, str] | None = None
    for row in rows:
        gpt = _float_or_none(row.get(gpt_col, ""))
        n_ge = _int_or_none(row.get("n_prompts_binomial_z_ge"))
        if gpt is None or n_ge is None:
            continue
        if n_ge < required:
            continue
        if best_gpt is None or gpt > best_gpt:
            best_gpt = gpt
            best_row = row
    return best_gpt, best_row, required


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("water-bench-results/json-outputs/glo-outputs"),
        help="Folder with per-run eval JSON files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "water-bench-results/json-outputs/glo-outputs_gpt4_summary.csv"
        ),
        help="GPT-4 summary CSV to merge into.",
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=Path(
            "water-bench-results/json-outputs/glo-outputs_gpt4_summary_merged.csv"
        ),
        help="Output path for merged CSV.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=4.0,
        help="Count prompts with binomial_z_score >= this (default: 4).",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=100,
        help="Cohort size for percentage cutoffs (default: 100).",
    )
    parser.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        default=[99.0, 95.0, 90.0, 85.0],
        help="Required %% of prompts with z >= z-threshold (default: 99 95 90 85).",
    )
    parser.add_argument(
        "--gpt-column",
        type=str,
        default="average_gpt_score",
        help="Column to maximize.",
    )
    args = parser.parse_args()

    if not args.json_dir.is_dir():
        raise SystemExit(f"JSON directory not found: {args.json_dir}")
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    counts = scan_folder(args.json_dir, args.z_threshold)
    print(
        f"Scanned {len(counts)} JSON files in {args.json_dir} "
        f"(binomial_z_score >= {args.z_threshold})."
    )

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        base_fieldnames = list(reader.fieldnames or [])
        csv_rows = list(reader)

    extra_cols = ["n_prompts_binomial_z_ge", "n_results_in_json"]
    for row in csv_rows:
        fname = row.get("file", "").strip()
        if fname in counts:
            n_ge, n_tot = counts[fname]
            row["n_prompts_binomial_z_ge"] = str(n_ge)
            row["n_results_in_json"] = str(n_tot)
        else:
            row["n_prompts_binomial_z_ge"] = ""
            row["n_results_in_json"] = ""

    fieldnames = list(base_fieldnames)
    for c in extra_cols:
        if c not in fieldnames:
            fieldnames.append(c)

    args.merged_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.merged_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote merged CSV: {args.merged_csv}")
    print(
        f"\nBest {args.gpt_column} at each detection rate "
        f"(need >= ceil({args.n_prompts} * pct/100) prompts with "
        f"binomial_z >= {args.z_threshold}):\n"
    )
    for pct in args.percentages:
        gpt, row, need = best_gpt_at_pct(
            csv_rows, args.n_prompts, pct, args.gpt_column
        )
        print(f"  >= {pct:g}%  (>= {need} / {args.n_prompts} prompts): ", end="")
        if row is None or gpt is None:
            print("no qualifying file")
        else:
            print(f"{args.gpt_column}={gpt}  file={row.get('file', '')}")


if __name__ == "__main__":
    main()
