#!/usr/bin/env python3
"""
Non-adaptive robustness ablation for CDMark, DMark, and LR-DWM -- following
each baseline's OWN paper methodology, not the Kuditipudi DP-alignment
recovery used for the Gumbel-max scheme (see run_edit_ablation_efficient.py).

None of CDMark (Table 1, cdmark.pdf), DMark (Table 2, dmark.pdf), or LR-DWM
(Table 2, lr-dwm.pdf) use any alignment search after edits -- their detectors
are inherently position-agnostic:
  - CDMark sums per-token signal vectors order-independently (z = ||sum v_xi||^2),
    so deletions/insertions/substitutions just change which/how-many vectors
    are summed -- no realignment needed.
  - DMark/LR-DWM re-derive each token's green-list membership from whichever
    neighbor token is ACTUALLY adjacent to it in the (possibly edited) text --
    the detector doesn't care what the "original" position was.
So robustness here is evaluated the same way the papers do it: apply the
edit, re-score the edited token sequence directly with the SAME unmodified
detector used for clean text, and check the same z>=4 threshold used
throughout this repo's pipeline.

Each baseline is evaluated at its OWN optimal hyperparameters (gamma, delta,
t_end), one operating point per method, matching how each paper fixes a
single strong configuration before running attacks:
  - CDMark:  gamma=0.90, delta=2, t_end=40
  - DMark:   gamma=0.10, delta=4, t_end=300
  - LR-DWM:  gamma=0.50, delta=8, t_end=160

Attack grid: deletion / insertion / substitution / mixed (sub+del+ins fired
together, matching run_edit_ablation_efficient.py's convention) at
eps in {0.05, 0.10, 0.20, 0.30}.

Detection uses each method's normal fixed z>=4 threshold (not a p-value
calibration -- that's specific to the Gumbel-max scheme's DP-alignment
recovery, see run_edit_ablation_efficient.py):
  - Soundness    = fraction of NO-WATERMARK samples with z >= 4
                   (should stay near 0%)
  - Completeness = fraction of WATERMARKED samples with z >= 4
                   (recall after the edit)

Usage:
    conda run -n llada --no-capture-output python run_baseline_robustness_ablation.py --device cuda:1 --n_samples 100
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import math
import json
import random
import argparse
import datetime
import torch
from pathlib import Path

_DLM_WM_SRC = Path(__file__).resolve().parent.parent / "diffusion-lm-watermark" / "src"
if _DLM_WM_SRC.is_dir() and str(_DLM_WM_SRC) not in sys.path:
    sys.path.insert(0, str(_DLM_WM_SRC))
from transformers import AutoTokenizer, AutoModel

from generate import (
    generate, get_special_token_ids,
    calculate_cdmark_score, calculate_dmark_score, calculate_lrdwm_score,
)
from run_mixed_comparison import load_jsonl, format_prompt, trim_eos, apply_mixed_edits

BEST_CONFIGS = {
    "cdmark":   dict(gamma=0.90, delta=2.0,  tend=40),
    "dmark":    dict(gamma=0.10, delta=4.0,  tend=300),
    "lrdwm":    dict(gamma=0.50, delta=8.0,  tend=160),
    "gloaguen": dict(gamma=0.10, delta=8.0,  tend=300),
}
EDIT_TYPES = ["del", "ins", "sub", "mixed"]
EDIT_TYPE_LABELS = {"del": "Deletion", "ins": "Insertion", "sub": "Substitution", "mixed": "Mixed"}
EPS_VALUES = [0.05, 0.10, 0.20, 0.30]
Z_THRESH = 4.0


def eps_for_edit_type(edit_type, eps):
    """Map an ablation cell to (eps_sub, eps_del, eps_ins) for apply_mixed_edits."""
    if edit_type == "del":
        return 0.0, eps, 0.0
    if edit_type == "ins":
        return 0.0, 0.0, eps
    if edit_type == "sub":
        return eps, 0.0, 0.0
    if edit_type == "mixed":
        return eps, eps, eps
    raise ValueError(f"Unknown edit_type: {edit_type}")


def compute_score(method, gamma, tokens_list, seed, vocab_size, mask_id, device,
                  gloaguen_wm=None):
    if not tokens_list:
        return 0.0, 0
    t = torch.tensor(tokens_list, dtype=torch.long, device=device).unsqueeze(0)
    if method == "cdmark":
        z, n = calculate_cdmark_score(t, secret_key=seed, vocab_size=vocab_size, m=1, mask_id=mask_id)
    elif method == "dmark":
        z, n = calculate_dmark_score(t, secret_key=seed, gamma=gamma, vocab_size=vocab_size,
                                      variant="predictive_bidirectional", mask_id=mask_id)
    elif method == "lrdwm":
        z, n = calculate_lrdwm_score(t, secret_key=seed, gamma=gamma, vocab_size=vocab_size, mask_id=mask_id)
    elif method == "gloaguen":
        result = gloaguen_wm.detect(t[0])
        z = float(result.get("binomial_z_score", result.get("z_score", 0.0)))
        n = len(tokens_list)
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(z), int(n)


def generate_corpus(model, tokenizer, entries, prompt_tokens_list, args, special_token_ids,
                     watermark_type, gamma=None, delta=None, tend=None, seed=None, label="",
                     gloaguen_wm=None):
    print(f"\n=== Generating {len(entries)} '{label}' sequences ===")
    samples = []
    for idx, (entry, ptoks) in enumerate(zip(entries, prompt_tokens_list)):
        if ptoks is None:
            continue
        prompt_tensor = torch.tensor([ptoks]).to(args.device)
        gen_kwargs = dict(
            model=model, prompt=prompt_tensor,
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
            temperature=args.temperature, remasking="low_confidence", mask_id=args.mask_id,
            vocab_size=args.vocab_size, special_token_ids=special_token_ids,
        )
        if watermark_type == "cdmark":
            gen_kwargs.update(watermark_type="cdmark", amplification=delta,
                               cdmark_seed=seed, cdmark_m=1, watermark_steps=tend)
        elif watermark_type == "dmark":
            gen_kwargs.update(watermark_type="dmark", gamma=gamma, amplification=delta,
                               dmark_variant="predictive_bidirectional", dmark_seed=seed,
                               watermark_steps=tend)
        elif watermark_type == "lrdwm":
            gen_kwargs.update(watermark_type="lrdwm", gamma=gamma, amplification=delta,
                               lrdwm_seed=seed, watermark_steps=tend)
        elif watermark_type == "gloaguen":
            gen_kwargs.update(watermark_type="gloaguen", watermark_steps=tend,
                               gloaguen_watermark=gloaguen_wm)
        elif watermark_type is None:
            gen_kwargs.update(watermark_type=None)
        with torch.no_grad():
            out = generate(**gen_kwargs)
        gen = trim_eos(out[0, len(ptoks):].tolist())
        samples.append({"prompt_len": len(ptoks), "tokens": gen})
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(entries)}")
    print(f"Generated {len(samples)} '{label}' samples")
    return samples


def run_grid_for_method(method, wm_samples, uw_samples, gamma, seed, vocab_size, mask_id,
                         edit_seed, device, gloaguen_wm=None):
    n = min(len(wm_samples), len(uw_samples))
    rows = []
    for edit_type in EDIT_TYPES:
        for eps in EPS_VALUES:
            rng = random.Random(edit_seed)
            eps_sub, eps_del, eps_ins = eps_for_edit_type(edit_type, eps)
            wm_dets, uw_dets = [], []
            for i in range(n):
                wm_tok = wm_samples[i]["tokens"]
                uw_tok = uw_samples[i]["tokens"]
                wm_edit, *_ = apply_mixed_edits(wm_tok, eps_sub, eps_del, eps_ins, rng, vocab_size, mask_id)
                uw_edit, *_ = apply_mixed_edits(uw_tok, eps_sub, eps_del, eps_ins, rng, vocab_size, mask_id)
                z_wm, _ = compute_score(method, gamma, wm_edit, seed, vocab_size, mask_id, device,
                                        gloaguen_wm=gloaguen_wm)
                z_uw, _ = compute_score(method, gamma, uw_edit, seed, vocab_size, mask_id, device,
                                        gloaguen_wm=gloaguen_wm)
                wm_dets.append(z_wm >= Z_THRESH)
                uw_dets.append(z_uw >= Z_THRESH)
            completeness = sum(wm_dets) / len(wm_dets) if wm_dets else float("nan")
            soundness = sum(uw_dets) / len(uw_dets) if uw_dets else float("nan")
            rows.append({
                "edit_type": edit_type, "eps": eps,
                "completeness": completeness, "soundness": soundness, "n": n,
            })
    return rows


def print_method_table(method, rows):
    print(f"\n=== {method.upper()} robustness (z>={Z_THRESH} threshold) ===")
    print(f"{'Edit Type':<14} {'eps':>6}   {'Soundness':>13} {'Completeness':>13}")
    for r in rows:
        label = EDIT_TYPE_LABELS[r["edit_type"]]
        print(f"{label:<14} {r['eps']:>6.2f}   {r['soundness']:>12.1%} {r['completeness']:>12.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", default="water-bench-sampled_100_seed43.jsonl")
    parser.add_argument("--output_dir", default="water-bench-results/json-outputs")
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen_length", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--block_length", type=int, default=25)
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--edit_seed", type=int, default=99)
    parser.add_argument("--methods", nargs="+", default=list(BEST_CONFIGS.keys()),
                        choices=list(BEST_CONFIGS.keys()),
                        help="Which methods to run (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_robustness_ablation.json"

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

    gloaguen_wm = None
    if "gloaguen" in args.methods:
        from dlm_watermark.watermarks.diffusion_watermark import OurWatermark
        gloaguen_cfg = BEST_CONFIGS["gloaguen"]
        gloaguen_wm = OurWatermark(
            delta=gloaguen_cfg["delta"],
            enforce_kl=True,
            topk=100,
            n_iter=1,
            seeding_scheme="sumhash",
            tokenizer=tokenizer,
            device=args.device,
        )
        print(f"Instantiated Gloaguen watermark (delta={gloaguen_cfg['delta']})")

    uw_samples = generate_corpus(model, tokenizer, entries, prompt_tokens_list, args,
                                  special_token_ids, watermark_type=None, label="no-watermark")

    wm_corpora = {}
    for method, cfg in BEST_CONFIGS.items():
        if method not in args.methods:
            continue
        wm_corpora[method] = generate_corpus(
            model, tokenizer, entries, prompt_tokens_list, args, special_token_ids,
            watermark_type=method, gamma=cfg["gamma"], delta=cfg["delta"], tend=cfg["tend"],
            seed=args.seed, label=f"{method} (gamma={cfg['gamma']} delta={cfg['delta']} tend={cfg['tend']})",
            gloaguen_wm=gloaguen_wm,
        )

    all_results = {}
    for method, cfg in BEST_CONFIGS.items():
        if method not in args.methods:
            continue
        rows = run_grid_for_method(
            method, wm_corpora[method], uw_samples, cfg["gamma"], args.seed,
            args.vocab_size, args.mask_id, args.edit_seed, args.device,
            gloaguen_wm=gloaguen_wm,
        )
        all_results[method] = {"config": cfg, "rows": rows}
        print_method_table(method, rows)

    print("\n" + "=" * 78)
    print(f"CONSOLIDATED SUMMARY -- Soundness/Completeness per method / attack / eps (z>={Z_THRESH})")
    print("=" * 78)
    print(f"{'method':<8} {'Edit Type':<14} {'eps':>6}   {'Soundness':>13} {'Completeness':>13}")
    for method, data in all_results.items():
        for r in data["rows"]:
            label = EDIT_TYPE_LABELS[r["edit_type"]]
            print(f"{method:<8} {label:<14} {r['eps']:>6.2f}   {r['soundness']:>12.1%} {r['completeness']:>12.1%}")

    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "config": vars(args),
            "best_configs": BEST_CONFIGS,
            "z_thresh": Z_THRESH,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved full results -> {out_path}")


if __name__ == "__main__":
    main()
