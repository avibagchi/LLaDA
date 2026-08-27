#!/usr/bin/env python3
"""
Edit-type x epsilon ablation for the Gumbel-max watermark's robustness to
post-generation edits, built on top of run_mixed_comparison.py's edit/scoring
primitives.

Unlike calling run_mixed_comparison.py once per (edit_type, eps) combination
(which would reload the model and regenerate 2*n_samples sequences from
scratch every time), this script generates the shared WM/UW corpus ONCE and
reuses it across the whole ablation grid -- only the edit+scoring step is
repeated per combination.

Ablation grid (isolate one edit type at a time, then a combined "mixed" case
matching run_mixed_comparison.py's default where all three fire together):
    edit_type in {del, ins, sub, mixed}
    eps       in {0.05, 0.10, 0.20, 0.30}
  -> 16 combinations, each scored on n_samples prompts.

For every combination this prints the full 6-distribution table (raw score +
det@1.19, and Algorithm-2-calibrated p-value + det@p<0.05), matching
run_mixed_comparison.py's own output format. At the end it prints one
consolidated table across all combinations, emphasizing:
  - UW+mixed (DP) --  false positive rate (should be tiny; raw/det@1.19 is
    KNOWN to be broken here per run_gumbel_dp_calibrated.py's own docstring --
    printing both makes that failure mode visible across the whole grid)
  - WM+mixed (DP) --  completeness (recovered detection rate after edits)

Usage:
    conda run -n llada --no-capture-output python run_edit_ablation_efficient.py --device cuda:1
    conda run -n llada --no-capture-output python run_edit_ablation_efficient.py --device cuda:1 \\
        --n_samples 100 --n_perms 50 --eps_values 0.05,0.1,0.2,0.3 --edit_types del,ins,sub,mixed
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import math
import json
import random
import argparse
import datetime
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from generate import generate, get_special_token_ids
from run_mixed_comparison import (
    load_jsonl, format_prompt, trim_eos,
    apply_mixed_edits, naive_score, score_matrix, dp_score,
    permutation_pvalue_naive, permutation_pvalue_dp,
)

DEFAULT_EPS_VALUES = [0.05, 0.10, 0.20, 0.30]
DEFAULT_EDIT_TYPES = ["del", "ins", "sub", "mixed"]
TAU = 1.19
ALPHA = 0.05


def eps_for_edit_type(edit_type, eps):
    """Map an ablation cell to (eps_sub, eps_del, eps_ins)."""
    if edit_type == "del":
        return 0.0, eps, 0.0
    if edit_type == "ins":
        return 0.0, 0.0, eps
    if edit_type == "sub":
        return eps, 0.0, 0.0
    if edit_type == "mixed":
        return eps, eps, eps
    raise ValueError(f"Unknown edit_type: {edit_type}")


def stats(s):
    v = [x for x in s if math.isfinite(x)]
    return (sum(v) / len(v), sorted(v)[len(v) // 2], len(v)) if v else (float("nan"), float("nan"), 0)


def score_combo(wm_samples, uw_samples, eps_sub, eps_del, eps_ins,
                watermark_seed, vocab_size, mask_id, edit_seed, n_perms, device):
    rng = random.Random(edit_seed)
    n = min(len(wm_samples), len(uw_samples))

    out = {k: [] for k in [
        "scores_uw", "scores_wm",
        "scores_uw_mix_naive", "scores_uw_mix_dp",
        "scores_wm_mix_naive", "scores_wm_mix_dp",
        "pvalues_uw", "pvalues_wm",
        "pvalues_uw_mix_naive", "pvalues_uw_mix_dp",
        "pvalues_wm_mix_naive", "pvalues_wm_mix_dp",
    ]}

    for i in range(n):
        prompt_len = wm_samples[i]["prompt_len"]
        wm_tok = wm_samples[i]["tokens"]
        uw_tok = uw_samples[i]["tokens"]

        out["scores_uw"].append(naive_score(uw_tok, watermark_seed, prompt_len, vocab_size, device))
        out["scores_wm"].append(naive_score(wm_tok, watermark_seed, prompt_len, vocab_size, device))

        uw_mix, n_del_uw, n_ins_uw, _ = apply_mixed_edits(uw_tok, eps_sub, eps_del, eps_ins, rng, vocab_size, mask_id)
        wm_mix, n_del_wm, n_ins_wm, _ = apply_mixed_edits(wm_tok, eps_sub, eps_del, eps_ins, rng, vocab_size, mask_id)

        L_uw_orig = len(uw_tok)
        L_wm_orig = len(wm_tok)

        out["scores_uw_mix_naive"].append(naive_score(uw_mix, watermark_seed, prompt_len, vocab_size, device))
        out["scores_wm_mix_naive"].append(naive_score(wm_mix, watermark_seed, prompt_len, vocab_size, device))

        k_uw = n_del_uw + n_ins_uw
        if L_uw_orig > 0 and len(uw_mix) > 0 and k_uw >= 0:
            mat = score_matrix(uw_mix, L_uw_orig, watermark_seed, prompt_len, vocab_size, device)
            tk = dp_score(mat, L_uw_orig, len(uw_mix), k_uw)
            out["scores_uw_mix_dp"].append(tk.get(k_uw, float("nan")))
        else:
            out["scores_uw_mix_dp"].append(float("nan"))

        k_wm = n_del_wm + n_ins_wm
        if L_wm_orig > 0 and len(wm_mix) > 0 and k_wm >= 0:
            mat = score_matrix(wm_mix, L_wm_orig, watermark_seed, prompt_len, vocab_size, device)
            tk = dp_score(mat, L_wm_orig, len(wm_mix), k_wm)
            out["scores_wm_mix_dp"].append(tk.get(k_wm, float("nan")))
        else:
            out["scores_wm_mix_dp"].append(float("nan"))

        out["pvalues_uw"].append(permutation_pvalue_naive(out["scores_uw"][-1], len(uw_tok), n_perms))
        out["pvalues_wm"].append(permutation_pvalue_naive(out["scores_wm"][-1], len(wm_tok), n_perms))
        out["pvalues_uw_mix_naive"].append(permutation_pvalue_naive(out["scores_uw_mix_naive"][-1], len(uw_mix), n_perms))
        out["pvalues_wm_mix_naive"].append(permutation_pvalue_naive(out["scores_wm_mix_naive"][-1], len(wm_mix), n_perms))
        out["pvalues_uw_mix_dp"].append(permutation_pvalue_dp(out["scores_uw_mix_dp"][-1], L_uw_orig, len(uw_mix), k_uw, n_perms))
        out["pvalues_wm_mix_dp"].append(permutation_pvalue_dp(out["scores_wm_mix_dp"][-1], L_wm_orig, len(wm_mix), k_wm, n_perms))

    return out


def print_combo_tables(tag, out):
    print(f"\n=== Summary (raw scores) -- {tag} ===")
    print(f"{'Distribution':<35} {'Mean':>8} {'Median':>8} {'N':>5}  det@1.19")
    rows_raw = [
        ("1. UW clean", out["scores_uw"]),
        ("2. WM clean", out["scores_wm"]),
        ("3. UW+mixed (naive)", out["scores_uw_mix_naive"]),
        ("4. UW+mixed (DP)", out["scores_uw_mix_dp"]),
        ("5. WM+mixed (naive)", out["scores_wm_mix_naive"]),
        ("6. WM+mixed (DP)", out["scores_wm_mix_dp"]),
    ]
    for name, sc in rows_raw:
        m, med, n_ = stats(sc)
        v = [x for x in sc if math.isfinite(x)]
        det = sum(x > TAU for x in v) / len(v) if v else float("nan")
        marker = "  <-- FPR" if name.startswith("4.") else ("  <-- completeness" if name.startswith("6.") else "")
        print(f"  {name:<33} {m:>8.4f} {med:>8.4f} {n_:>5}  {det:>7.1%}{marker}")

    print(f"\n=== Summary (Algorithm 2 p-values) -- {tag} ===")
    print(f"{'Distribution':<35} {'Mean p':>8} {'Med p':>8} {'N':>5}  det@p<0.05")
    rows_p = [
        ("1. UW clean", out["pvalues_uw"]),
        ("2. WM clean", out["pvalues_wm"]),
        ("3. UW+mixed (naive)", out["pvalues_uw_mix_naive"]),
        ("4. UW+mixed (DP)", out["pvalues_uw_mix_dp"]),
        ("5. WM+mixed (naive)", out["pvalues_wm_mix_naive"]),
        ("6. WM+mixed (DP)", out["pvalues_wm_mix_dp"]),
    ]
    for name, pvals in rows_p:
        m, med, n_ = stats(pvals)
        v = [x for x in pvals if math.isfinite(x)]
        det = sum(x < ALPHA for x in v) / len(v) if v else float("nan")
        marker = "  <-- FPR" if name.startswith("4.") else ("  <-- completeness" if name.startswith("6.") else "")
        print(f"  {name:<33} {m:>8.4f} {med:>8.4f} {n_:>5}  {det:>7.1%}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", default="water-bench-sampled_100_seed43.jsonl")
    parser.add_argument("--output_dir", default="water-bench-results/json-outputs")
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--watermark_seed", type=int, default=42)
    parser.add_argument("--gen_length", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--block_length", type=int, default=25)
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--edit_seed", type=int, default=99)
    parser.add_argument("--n_perms", type=int, default=50,
                        help="Permutation trials for Algorithm 2 p-values (kept lower than the "
                             "single-combo script's default of 200 since this multiplies across "
                             "the whole ablation grid)")
    parser.add_argument("--eps_values", type=str, default=",".join(str(e) for e in DEFAULT_EPS_VALUES),
                        help="Comma-separated epsilon values to sweep, e.g. 0.05,0.1,0.2,0.3")
    parser.add_argument("--edit_types", type=str, default=",".join(DEFAULT_EDIT_TYPES),
                        help="Comma-separated edit types to sweep: del,ins,sub,mixed")
    parser.add_argument("--resume", action="store_true",
                        help="Skip combos whose result is already in the output JSON")
    args = parser.parse_args()

    eps_values = [float(x) for x in args.eps_values.split(",") if x.strip()]
    edit_types = [x.strip() for x in args.edit_types.split(",") if x.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "edit_ablation_results.json"

    all_results = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f).get("combos", {})
        print(f"Resume mode: {len(all_results)} combos already present")

    print(f"Loading LLaDA on {args.device}...")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    special_token_ids = get_special_token_ids(tokenizer)

    entries = load_jsonl(args.jsonl_file, args.n_samples)
    prompt_tokens_list = []
    for entry in entries:
        text = format_prompt(entry.get("context", ""), entry.get("input", ""), tokenizer)
        prompt_tokens_list.append(tokenizer(text)["input_ids"] if text else None)
    print(f"Loaded {len(entries)} prompts")

    # -----------------------------------------------------------------------
    # Generate the shared WM/UW corpus ONCE -- reused across every (edit_type,
    # eps) combination below.
    # -----------------------------------------------------------------------
    print(f"\n=== Generating {len(entries)} watermarked sequences (shared corpus) ===")
    wm_samples = []
    for idx, (entry, ptoks) in enumerate(zip(entries, prompt_tokens_list)):
        if ptoks is None:
            continue
        with torch.no_grad():
            out_tok = generate(
                model=model, prompt=torch.tensor([ptoks]).to(args.device),
                steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
                temperature=args.temperature, remasking="low_confidence", mask_id=args.mask_id,
                watermark_type="aaronson", vocab_size=args.vocab_size,
                special_token_ids=special_token_ids, aaronson_seed=args.watermark_seed,
                watermark_steps=args.steps,
            )
        gen = trim_eos(out_tok[0, len(ptoks):].tolist())
        wm_samples.append({"prompt_len": len(ptoks), "tokens": gen})
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(entries)}")
    print(f"Generated {len(wm_samples)} watermarked samples")

    print(f"\n=== Generating {len(entries)} unwatermarked sequences (shared corpus) ===")
    uw_samples = []
    for idx, (entry, ptoks) in enumerate(zip(entries, prompt_tokens_list)):
        if ptoks is None:
            continue
        with torch.no_grad():
            out_tok = generate(
                model=model, prompt=torch.tensor([ptoks]).to(args.device),
                steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
                temperature=args.temperature, remasking="low_confidence", mask_id=args.mask_id,
                watermark_type=None, vocab_size=args.vocab_size,
                special_token_ids=special_token_ids,
            )
        gen = trim_eos(out_tok[0, len(ptoks):].tolist())
        uw_samples.append({"prompt_len": len(ptoks), "tokens": gen})
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(entries)}")
    print(f"Generated {len(uw_samples)} unwatermarked samples")

    # -----------------------------------------------------------------------
    # Ablation grid
    # -----------------------------------------------------------------------
    combos = [(et, eps) for et in edit_types for eps in eps_values]
    print(f"\nAblation grid: {len(combos)} combinations "
          f"(edit_types={edit_types}, eps_values={eps_values}, n_perms={args.n_perms})")

    summary_rows = []
    for combo_idx, (edit_type, eps) in enumerate(combos):
        tag = f"{edit_type}_eps{eps:.2f}"
        if tag in all_results:
            print(f"\n[{combo_idx+1}/{len(combos)}] {tag} -- skipped (resume)")
            out = all_results[tag]
        else:
            eps_sub, eps_del, eps_ins = eps_for_edit_type(edit_type, eps)
            print(f"\n[{combo_idx+1}/{len(combos)}] {tag}  "
                  f"(eps_sub={eps_sub:.2f}, eps_del={eps_del:.2f}, eps_ins={eps_ins:.2f})")
            out = score_combo(
                wm_samples, uw_samples, eps_sub, eps_del, eps_ins,
                args.watermark_seed, args.vocab_size, args.mask_id,
                args.edit_seed, args.n_perms, args.device,
            )
            all_results[tag] = out
            with open(out_path, "w") as f:
                json.dump({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "config": vars(args),
                    "combos": all_results,
                }, f, indent=2)

        print_combo_tables(tag, out)

        fpr_raw = sum(x > TAU for x in out["scores_uw_mix_dp"] if math.isfinite(x)) / max(1, sum(math.isfinite(x) for x in out["scores_uw_mix_dp"]))
        comp_raw = sum(x > TAU for x in out["scores_wm_mix_dp"] if math.isfinite(x)) / max(1, sum(math.isfinite(x) for x in out["scores_wm_mix_dp"]))
        fpr_cal = sum(x < ALPHA for x in out["pvalues_uw_mix_dp"] if math.isfinite(x)) / max(1, sum(math.isfinite(x) for x in out["pvalues_uw_mix_dp"]))
        comp_cal = sum(x < ALPHA for x in out["pvalues_wm_mix_dp"] if math.isfinite(x)) / max(1, sum(math.isfinite(x) for x in out["pvalues_wm_mix_dp"]))
        summary_rows.append((edit_type, eps, fpr_raw, comp_raw, fpr_cal, comp_cal))

    # -----------------------------------------------------------------------
    # Consolidated summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("CONSOLIDATED SUMMARY -- UW+mixed(DP) false positive rate & WM+mixed(DP) completeness")
    print("=" * 88)
    print(f"{'edit_type':<10} {'eps':>6}   {'FPR@tau=1.19':>13} {'Complete@tau=1.19':>18}   "
          f"{'FPR@p<0.05':>11} {'Complete@p<0.05':>16}")
    for edit_type, eps, fpr_raw, comp_raw, fpr_cal, comp_cal in summary_rows:
        print(f"{edit_type:<10} {eps:>6.2f}   {fpr_raw:>12.1%} {comp_raw:>17.1%}   "
              f"{fpr_cal:>10.1%} {comp_cal:>15.1%}")

    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "config": vars(args),
            "combos": all_results,
            "summary": [
                {"edit_type": et, "eps": eps, "fpr_raw_tau1.19": fr, "completeness_raw_tau1.19": cr,
                 "fpr_calibrated_p0.05": fc, "completeness_calibrated_p0.05": cc}
                for et, eps, fr, cr, fc, cc in summary_rows
            ],
        }, f, indent=2)
    print(f"\nSaved full results -> {out_path}")


if __name__ == "__main__":
    main()
