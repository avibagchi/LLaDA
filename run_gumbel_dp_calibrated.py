#!/usr/bin/env python3
"""
DP-robust watermark detection ablation for Gumbel-max watermarking, with
PER-CELL null calibration.

run_gumbel_dp_robustness.py uses one fixed threshold tau=1.19 (calibrated for
the zero-edit-budget statistic) across every edit budget k. That is invalid:
letting the Kuditipudi DP search over an edit-budget-k alignment space
inflates the max-score statistic upward regardless of whether a watermark is
present (a multiple-comparisons effect) -- verified empirically: unwatermarked
text scored with this DP hits 100% "detection" at fixed tau=1.19 once k>=4.

This script instead:
  1. Generates n_samples Gumbel-max WATERMARKED sequences (Phase 1).
  2. Generates n_null_samples genuinely UNWATERMARKED sequences from the same
     model (Phase 1b) -- these anchor the null hypothesis.
  3. For every (edit_type, epsilon, k_mult) cell, scores the watermarked
     samples under bounded edits exactly as before (Phase 2).
  4. Calibrates tau_k for that SAME cell from the null samples: each null
     sample is re-edited with n_null_edit_draws independent random draws of
     the same (edit_type, epsilon), scored with the same k, and the
     (1 - fpr_target) percentile of that pooled null distribution becomes
     tau_k (Phase 3).
  5. Reports, per cell: the watermarked mean score, the null mean score, the
     calibrated tau_k, and the detection rate under tau_k -- so completeness
     is judged against a threshold that actually controls the false-positive
     rate at this edit budget, instead of against a threshold that was never
     calibrated for it.

Usage:
  python run_gumbel_dp_calibrated.py --device cuda:0
  python run_gumbel_dp_calibrated.py --device cuda:0 --n_samples 30 --n_null_samples 30
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import json
import math
import random
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from generate import generate, get_special_token_ids
from run_gumbel_dp_robustness import (
    load_jsonl, format_prompt, apply_edits, compute_score_matrix, dp_detect,
    EPSILON_VALUES, EDIT_TYPES, K_MULTIPLIERS, DETECTION_THRESHOLD,
)

EOS_IDS = {50256, 2, 126081}


# ---------------------------------------------------------------------------
# Generation (shared for watermarked + null passes)
# ---------------------------------------------------------------------------

def generate_samples(model, prompt_tokens_list, entries, args, watermark_type, desc):
    samples = []
    for idx, (entry, prompt_toks) in enumerate(zip(entries, prompt_tokens_list)):
        if prompt_toks is None:
            continue
        prompt_tensor = torch.tensor([prompt_toks]).to(args.device)
        with torch.no_grad():
            kwargs = dict(
                model=model,
                prompt=prompt_tensor,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                remasking="low_confidence",
                mask_id=args.mask_id,
                vocab_size=args.vocab_size,
                special_token_ids=args.special_token_ids,
            )
            if watermark_type == "aaronson":
                kwargs.update(
                    watermark_type="aaronson",
                    aaronson_seed=args.watermark_seed,
                    watermark_steps=args.watermark_steps,
                )
            else:
                kwargs.update(watermark_type="none")
            out = generate(**kwargs)
        gen_toks = out[0, len(prompt_toks):]
        actual_len = gen_toks.shape[0]
        for j, t in enumerate(gen_toks):
            if t.item() in EOS_IDS:
                actual_len = j
                break
        samples.append({
            "sample_id": idx,
            "prompt_length": len(prompt_toks),
            "gen_tokens": gen_toks[:actual_len].tolist(),
        })
        if (idx + 1) % 10 == 0:
            print(f"  [{desc}] {idx + 1}/{len(entries)}")
    return samples


# ---------------------------------------------------------------------------
# Null calibration
# ---------------------------------------------------------------------------

def null_cell_scores_all_mults(null_samples, edit_type, epsilon, k_multipliers, vocab_size,
                                watermark_seed, n_draws, rng):
    """
    Pool DP scores for (edit_type, epsilon) across all null samples and all
    k_multipliers in one pass, re-editing each null sample n_draws times (1
    draw if epsilon==0, since apply_edits is deterministic with no edits).
    Shares one score_matrix/dp_detect call across all k_multipliers per draw,
    exactly like the observed-score computation in Phase 2.

    Returns {mult: [scores]}.
    """
    scores_by_mult = {mult: [] for mult in k_multipliers}
    draws = 1 if epsilon == 0 else n_draws
    for sample in null_samples:
        orig_tokens = sample["gen_tokens"]
        prompt_len = sample["prompt_length"]
        L_key = len(orig_tokens)
        if L_key == 0:
            continue
        for _ in range(draws):
            edited, _, n_del, n_ins = apply_edits(orig_tokens, epsilon, edit_type, vocab_size, rng)
            L_text = len(edited)
            n_shift = n_del + n_ins
            k_max_budget = int(math.ceil(2.0 * max(n_shift, 1)))

            score_mat = compute_score_matrix(edited, L_key, watermark_seed, prompt_len, vocab_size)
            tk_all = dp_detect(score_mat, L_key, L_text, k_max_budget)

            for mult in k_multipliers:
                k_val = min(int(round(mult * n_shift)), k_max_budget)
                s = tk_all.get(k_val, float("-inf"))
                if s > -1e30:
                    scores_by_mult[mult].append(s)
    return scores_by_mult


def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(math.ceil(pct / 100.0 * len(s))) - 1))
    return s[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", default="water-bench-sampled_100_seed43.jsonl")
    parser.add_argument("--output_dir", default="water-bench-results/json-outputs")
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--watermark_seed", type=int, default=42)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--watermark_steps", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=30,
                         help="Number of watermarked samples to test detection on.")
    parser.add_argument("--n_null_samples", type=int, default=30,
                         help="Number of genuinely unwatermarked samples used to build the null.")
    parser.add_argument("--n_null_edit_draws", type=int, default=10,
                         help="Independent random edit re-draws per null sample per cell.")
    parser.add_argument("--fpr_target", type=float, default=0.01,
                         help="Target false-positive rate for calibrating tau_k.")
    parser.add_argument("--edit_seed", type=int, default=99)
    parser.add_argument("--null_edit_seed", type=int, default=1337)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gumbel_dp_calibrated.json"

    print(f"Loading LLaDA from {args.model_path} on {args.device}...")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    args.special_token_ids = get_special_token_ids(tokenizer)

    entries_all = load_jsonl(args.jsonl_file)
    n_needed = args.n_samples + args.n_null_samples
    if len(entries_all) < n_needed:
        raise ValueError(
            f"{args.jsonl_file} has only {len(entries_all)} prompts; need {n_needed} "
            f"(--n_samples + --n_null_samples). Reduce sample counts or use a larger jsonl."
        )
    wm_entries = entries_all[:args.n_samples]
    null_entries = entries_all[args.n_samples:n_needed]

    def to_prompt_tokens(entries):
        out = []
        for entry in entries:
            text = format_prompt(entry.get("context", ""), entry.get("input", ""), tokenizer)
            out.append(tokenizer(text)["input_ids"] if text else None)
        return out

    wm_prompt_tokens = to_prompt_tokens(wm_entries)
    null_prompt_tokens = to_prompt_tokens(null_entries)

    print("\n=== Phase 1: Generating Gumbel-max WATERMARKED sequences ===")
    wm_samples = generate_samples(model, wm_prompt_tokens, wm_entries, args, "aaronson", "watermarked")
    print(f"Generated {len(wm_samples)} watermarked samples")

    print("\n=== Phase 1b: Generating genuinely UNWATERMARKED sequences (null) ===")
    null_samples = generate_samples(model, null_prompt_tokens, null_entries, args, "none", "null")
    print(f"Generated {len(null_samples)} null samples\n")

    # ------------------------------------------------------------------
    # Phase 2: score watermarked samples under bounded edits (same grid as
    # run_gumbel_dp_robustness.py)
    # ------------------------------------------------------------------
    print("=== Phase 2: scoring watermarked samples under bounded edits ===")
    edit_rng = random.Random(args.edit_seed)
    wm_results = []  # per-sample, per-cell observed dp_score

    for sample in tqdm(wm_samples, desc="Watermarked samples"):
        orig_tokens = sample["gen_tokens"]
        prompt_len = sample["prompt_length"]
        L_key = len(orig_tokens)
        rec = {"sample_id": sample["sample_id"], "L_key": L_key, "cells": {}}

        for edit_type in EDIT_TYPES:
            for epsilon in EPSILON_VALUES:
                edited, n_sub, n_del, n_ins = apply_edits(
                    orig_tokens, epsilon, edit_type, args.vocab_size, edit_rng
                )
                L_text = len(edited)
                n_shift = n_del + n_ins
                k_max_budget = int(math.ceil(2.0 * max(n_shift, 1)))

                score_mat = compute_score_matrix(edited, L_key, args.watermark_seed, prompt_len, args.vocab_size)
                tk_all = dp_detect(score_mat, L_key, L_text, k_max_budget)

                for mult in K_MULTIPLIERS:
                    k_val = min(int(round(mult * n_shift)), k_max_budget)
                    score = tk_all.get(k_val, float("-inf"))
                    key = (edit_type, epsilon, mult)
                    rec["cells"][f"{edit_type}|{epsilon:.2f}|{mult:.1f}"] = {
                        "n_shift": n_shift,
                        "k": k_val,
                        "score": round(score, 6) if score > -1e30 else None,
                    }
        wm_results.append(rec)

    # ------------------------------------------------------------------
    # Phase 3: calibrate tau_k per cell from the null pool
    # ------------------------------------------------------------------
    print("\n=== Phase 3: calibrating tau per (edit_type, epsilon, k_mult) cell from null pool ===")
    null_rng = random.Random(args.null_edit_seed)
    null_cache = {}  # (edit_type, epsilon, mult) -> scores list

    eps_type_pairs = [(et, eps) for et in EDIT_TYPES for eps in EPSILON_VALUES]
    for edit_type, epsilon in tqdm(eps_type_pairs, desc="Calibrating null"):
        scores_by_mult = null_cell_scores_all_mults(
            null_samples, edit_type, epsilon, K_MULTIPLIERS, args.vocab_size,
            args.watermark_seed, args.n_null_edit_draws, null_rng
        )
        for mult in K_MULTIPLIERS:
            null_cache[(edit_type, epsilon, mult)] = scores_by_mult[mult]

    # ------------------------------------------------------------------
    # Phase 4: combine + report
    # ------------------------------------------------------------------
    print("\n=== Summary: watermarked vs. null, fixed tau=1.19 vs. calibrated tau_k ===")
    summary = {}
    for edit_type in EDIT_TYPES:
        summary[edit_type] = {}
        for epsilon in EPSILON_VALUES:
            summary[edit_type][f"eps={epsilon:.2f}"] = {}
            for mult in K_MULTIPLIERS:
                wm_scores = []
                for rec in wm_results:
                    c = rec["cells"].get(f"{edit_type}|{epsilon:.2f}|{mult:.1f}")
                    if c and c["score"] is not None:
                        wm_scores.append(c["score"])

                null_scores = null_cache[(edit_type, epsilon, mult)]
                tau_k = percentile(null_scores, 100 * (1 - args.fpr_target))
                null_mean = sum(null_scores) / len(null_scores) if null_scores else None
                wm_mean = sum(wm_scores) / len(wm_scores) if wm_scores else None

                det_fixed = (sum(1 for s in wm_scores if s > DETECTION_THRESHOLD) / len(wm_scores)
                             if wm_scores else None)
                det_calibrated = (sum(1 for s in wm_scores if tau_k is not None and s > tau_k) / len(wm_scores)
                                   if wm_scores and tau_k is not None else None)

                mk = f"k_mult={mult:.1f}"
                eps_key = f"eps={epsilon:.2f}"
                summary[edit_type][eps_key][mk] = {
                    "wm_mean": round(wm_mean, 4) if wm_mean is not None else None,
                    "null_mean": round(null_mean, 4) if null_mean is not None else None,
                    "tau_fixed": DETECTION_THRESHOLD,
                    "tau_calibrated": round(tau_k, 4) if tau_k is not None else None,
                    "det_rate_fixed": round(det_fixed, 4) if det_fixed is not None else None,
                    "det_rate_calibrated": round(det_calibrated, 4) if det_calibrated is not None else None,
                    "n_null_trials": len(null_scores),
                }
                print(
                    f"  {edit_type:6s} {eps_key} {mk}  "
                    f"wm_mean={wm_mean:.3f}" if wm_mean is not None else f"  {edit_type:6s} {eps_key} {mk}  wm_mean=NA",
                    end="  "
                )
                print(
                    f"null_mean={null_mean:.3f}  tau_fixed=1.19  tau_calib={tau_k:.3f}  "
                    f"det_fixed={(det_fixed or 0)*100:.0f}%  det_calib={(det_calibrated or 0)*100:.0f}%"
                    if null_mean is not None and tau_k is not None else "null_mean=NA"
                )

    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {k: v for k, v in vars(args).items() if k != "special_token_ids"},
        "epsilon_values": EPSILON_VALUES,
        "edit_types": EDIT_TYPES,
        "k_multipliers": K_MULTIPLIERS,
        "fpr_target": args.fpr_target,
        "summary": summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
