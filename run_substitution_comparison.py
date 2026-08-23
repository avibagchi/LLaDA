#!/usr/bin/env python3
"""
4-distribution comparison of Gumbel-max watermark detection scores under substitution attack.

Substitutions replace tokens in-place, preserving positional alignment — so DP alignment
recovery does NOT apply (k=0). The naive aligned scorer is the correct and only scorer.

Distributions:
  1. Unwatermarked clean      → naive score
  2. Watermarked clean        → naive score
  3. Unwatermarked + subst'd  → naive score (alignment preserved, signal unchanged)
  4. Watermarked + subst'd    → naive score (alignment preserved, signal diluted)

Key comparison: box 3 (UW+sub) vs box 4 (WM+sub) — gap shows how much watermark
signal survives as substitution rate increases.

Usage:
    conda run -n llada python run_substitution_comparison.py --device cuda:0 --eps 0.2
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from generate import generate, get_special_token_ids

EOS_IDS = {50256, 2, 126081}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path, n):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if len(data) >= n:
                    break
    return data


def format_prompt(context, input_text, tokenizer):
    if context and input_text:
        body = f"{context}\n{input_text}"
    elif context:
        body = context
    elif input_text:
        body = input_text
    else:
        return None
    text = f"You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n\n{body}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def trim_eos(tokens):
    for i, t in enumerate(tokens):
        if t in EOS_IDS:
            return tokens[:i]
    return tokens


def apply_substitution(tokens, epsilon, rng, vocab_size, mask_id):
    """Randomly replace epsilon-fraction of tokens with random tokens."""
    n_sub = max(1, round(epsilon * len(tokens))) if epsilon > 0 and len(tokens) > 0 else 0
    if n_sub == 0:
        return list(tokens), 0
    positions = rng.sample(range(len(tokens)), min(n_sub, len(tokens)))
    result = list(tokens)
    for p in positions:
        new_tok = rng.randint(0, vocab_size - 1)
        while new_tok == mask_id:
            new_tok = rng.randint(0, vocab_size - 1)
        result[p] = new_tok
    return result, len(positions)


# ---------------------------------------------------------------------------
# Scoring (naive only — substitutions preserve alignment so DP is not needed)
# ---------------------------------------------------------------------------

def to_z_naive(score, L):
    """Analytical z-score: mu=1.0, sigma=1/sqrt(L) under H0."""
    if not math.isfinite(score) or L <= 0:
        return float("nan")
    return (score - 1.0) * math.sqrt(L)


def naive_score(tokens, watermark_seed, position_offset, vocab_size, device="cpu"):
    """
    Standard aligned score: seed position i, score token i.
    Score = (1/L) * sum_i -log(1 - r_i[token_i])
    device must match the device used during generation.
    """
    if not tokens:
        return float("nan")
    total = 0.0
    for i, tok in enumerate(tokens):
        g = torch.Generator(device=device)
        g.manual_seed(int(watermark_seed + position_offset + i))
        r_i = torch.rand(vocab_size, generator=g, device=device).clamp_(1e-8, 1 - 1e-8)
        total += -torch.log1p(-r_i[tok]).item()
    return total / len(tokens)


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
    parser.add_argument("--gen_length", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--block_length", type=int, default=25)
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.2,
                        help="Fraction of tokens to substitute (0.0-1.0)")
    parser.add_argument("--edit_seed", type=int, default=99)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    # Generate watermarked sequences
    # -----------------------------------------------------------------------
    print(f"\n=== Generating {len(entries)} watermarked sequences ===")
    wm_samples = []
    for idx, (entry, ptoks) in enumerate(zip(entries, prompt_tokens_list)):
        if ptoks is None:
            continue
        with torch.no_grad():
            out = generate(
                model=model, prompt=torch.tensor([ptoks]).to(args.device),
                steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
                temperature=args.temperature, remasking="low_confidence", mask_id=args.mask_id,
                watermark_type="aaronson", vocab_size=args.vocab_size,
                special_token_ids=special_token_ids, aaronson_seed=args.watermark_seed,
                watermark_steps=args.steps,
            )
        gen = trim_eos(out[0, len(ptoks):].tolist())
        wm_samples.append({"prompt_len": len(ptoks), "tokens": gen})
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(entries)}")
    print(f"Generated {len(wm_samples)} watermarked samples")

    # -----------------------------------------------------------------------
    # Generate unwatermarked sequences
    # -----------------------------------------------------------------------
    print(f"\n=== Generating {len(entries)} unwatermarked sequences ===")
    uw_samples = []
    for idx, (entry, ptoks) in enumerate(zip(entries, prompt_tokens_list)):
        if ptoks is None:
            continue
        with torch.no_grad():
            out = generate(
                model=model, prompt=torch.tensor([ptoks]).to(args.device),
                steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
                temperature=args.temperature, remasking="low_confidence", mask_id=args.mask_id,
                watermark_type=None, vocab_size=args.vocab_size,
                special_token_ids=special_token_ids,
            )
        gen = trim_eos(out[0, len(ptoks):].tolist())
        uw_samples.append({"prompt_len": len(ptoks), "tokens": gen})
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(entries)}")
    print(f"Generated {len(uw_samples)} unwatermarked samples")

    # -----------------------------------------------------------------------
    # Score all four distributions
    # -----------------------------------------------------------------------
    print(f"\n=== Scoring (eps={args.eps:.0%} substitution) ===")
    rng = random.Random(args.edit_seed)

    scores_uw         = []  # 1. unwatermarked clean
    scores_wm         = []  # 2. watermarked clean
    scores_uw_sub     = []  # 3. unwatermarked + substituted
    scores_wm_sub     = []  # 4. watermarked + substituted

    zscores_uw    = []
    zscores_wm    = []
    zscores_uw_sub = []
    zscores_wm_sub = []

    n = min(len(wm_samples), len(uw_samples))
    dev = args.device

    for i in range(n):
        prompt_len = wm_samples[i]["prompt_len"]
        wm_tok = wm_samples[i]["tokens"]
        uw_tok = uw_samples[i]["tokens"]

        scores_uw.append(naive_score(uw_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))
        scores_wm.append(naive_score(wm_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))

        uw_sub_tok, _ = apply_substitution(uw_tok, args.eps, rng, args.vocab_size, args.mask_id)
        wm_sub_tok, _ = apply_substitution(wm_tok, args.eps, rng, args.vocab_size, args.mask_id)

        scores_uw_sub.append(naive_score(uw_sub_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))
        scores_wm_sub.append(naive_score(wm_sub_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))

        zscores_uw.append(to_z_naive(scores_uw[-1], len(uw_tok)))
        zscores_wm.append(to_z_naive(scores_wm[-1], len(wm_tok)))
        zscores_uw_sub.append(to_z_naive(scores_uw_sub[-1], len(uw_sub_tok)))
        zscores_wm_sub.append(to_z_naive(scores_wm_sub[-1], len(wm_sub_tok)))

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    def stats(s):
        v = [x for x in s if math.isfinite(x)]
        return (sum(v)/len(v), sorted(v)[len(v)//2], len(v)) if v else (float("nan"), float("nan"), 0)

    print("\n=== Summary (raw scores) ===")
    print(f"{'Distribution':<30} {'Mean':>8} {'Median':>8} {'N':>5}  det@1.19")
    for name, sc in [
        ("1. UW clean",      scores_uw),
        ("2. WM clean",      scores_wm),
        ("3. UW+substituted", scores_uw_sub),
        ("4. WM+substituted", scores_wm_sub),
    ]:
        m, med, n_ = stats(sc)
        v = [x for x in sc if math.isfinite(x)]
        det = sum(x > 1.19 for x in v) / len(v) if v else float("nan")
        print(f"  {name:<28} {m:>8.4f} {med:>8.4f} {n_:>5}  {det:>7.1%}")

    print("\n=== Summary (z-scores, analytical: μ=1, σ=1/√L) ===")
    print(f"{'Distribution':<30} {'Mean z':>8} {'Med z':>8} {'N':>5}  det@z=2")
    for name, zsc in [
        ("1. UW clean",      zscores_uw),
        ("2. WM clean",      zscores_wm),
        ("3. UW+substituted", zscores_uw_sub),
        ("4. WM+substituted", zscores_wm_sub),
    ]:
        m, med, n_ = stats(zsc)
        v = [x for x in zsc if math.isfinite(x)]
        det = sum(x > 2.0 for x in v) / len(v) if v else float("nan")
        print(f"  {name:<28} {m:>8.3f} {med:>8.3f} {n_:>5}  {det:>7.1%}")

    # -----------------------------------------------------------------------
    # Save raw results
    # -----------------------------------------------------------------------
    out_path = out_dir / f"substitution_comparison_eps={args.eps:.2f}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "config": vars(args),
            "scores_uw":    scores_uw,
            "scores_wm":    scores_wm,
            "scores_uw_sub": scores_uw_sub,
            "scores_wm_sub": scores_wm_sub,
            "zscores_uw":    zscores_uw,
            "zscores_wm":    zscores_wm,
            "zscores_uw_sub": zscores_uw_sub,
            "zscores_wm_sub": zscores_wm_sub,
        }, f, indent=2)
    print(f"\nSaved → {out_path}")

    # -----------------------------------------------------------------------
    # Box plot
    # -----------------------------------------------------------------------
    labels = [
        "UW\n(clean)",
        "WM\n(clean)",
        f"UW+sub\n({args.eps:.0%})",
        f"WM+sub\n({args.eps:.0%})",
    ]
    colors = ["#4C72B0", "#DD8452", "#8ea9d8", "#8B2500"]

    def clean(s):
        return [x for x in s if math.isfinite(x)]

    data = [clean(scores_uw), clean(scores_wm), clean(scores_uw_sub), clean(scores_wm_sub)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.axvline(2.5, color="gray", linestyle=":", linewidth=0.8)
    ymax = ax.get_ylim()[1]
    ax.text(1.5, ymax * 0.97, "No edit", ha="center", fontsize=8, color="gray")
    ax.text(3.5, ymax * 0.97, f"{args.eps:.0%} substitution", ha="center", fontsize=8, color="gray")

    ax.axhline(1.19, color="green", linestyle="--", linewidth=1.2, label="τ* = 1.19")
    ax.set_ylabel("Detection Score")
    ax.set_title(
        f"Gumbel-max Watermark Detection Scores Under Substitution\n"
        f"(alignment preserved — no DP needed, gen_length={args.gen_length}, ε={args.eps:.0%}, n={n})"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig_path = out_dir / f"substitution_comparison_eps={args.eps:.2f}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {fig_path}")

    # -----------------------------------------------------------------------
    # Z-score box plot (analytical null, no calibration needed)
    # -----------------------------------------------------------------------
    zlabels = [
        "UW\n(clean)",
        "WM\n(clean)",
        f"UW+sub\n({args.eps:.0%})",
        f"WM+sub\n({args.eps:.0%})",
    ]

    def clean(s):
        return [x for x in s if math.isfinite(x)]

    zdata = [clean(zscores_uw), clean(zscores_wm), clean(zscores_uw_sub), clean(zscores_wm_sub)]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bp2 = ax2.boxplot(zdata, tick_labels=zlabels, patch_artist=True, widths=0.55,
                      flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp2["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax2.axvline(2.5, color="gray", linestyle=":", linewidth=0.8)
    ymax2 = ax2.get_ylim()[1]
    ax2.text(1.5, ymax2 * 0.97, "No edit", ha="center", fontsize=8, color="gray")
    ax2.text(3.5, ymax2 * 0.97, f"{args.eps:.0%} substitution", ha="center", fontsize=8, color="gray")

    ax2.axhline(2.0, color="green", linestyle="--", linewidth=1.2, label="z* = 2.0  (~2.3% FPR)")
    ax2.axhline(0.0, color="black", linestyle="-",  linewidth=0.6, alpha=0.3)
    ax2.set_ylabel("Z-score")
    ax2.set_title(
        f"Gumbel-max Watermark Z-scores Under Substitution\n"
        f"(analytical null: μ=1, σ=1/√L, gen_length={args.gen_length}, ε={args.eps:.0%}, n={n})"
    )
    ax2.legend(fontsize=9)
    fig2.tight_layout()

    zfig_path = out_dir / f"substitution_zscores_eps={args.eps:.2f}.pdf"
    fig2.savefig(zfig_path, bbox_inches="tight")
    fig2.savefig(str(zfig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved z-score figure → {zfig_path}")


if __name__ == "__main__":
    main()
