#!/usr/bin/env python3
"""
6-distribution comparison of Gumbel-max watermark detection scores under insertion attack.
Insertions shift alignment (like deletions), so DP T_k recovery applies.

Distributions:
  1. Unwatermarked clean          → naive score
  2. Watermarked clean            → naive score
  3. Unwatermarked + inserted     → naive score  (broken alignment, before DP)
  4. Unwatermarked + inserted     → DP T_k score, k=n_ins (after DP)
  5. Watermarked + inserted       → naive score  (broken alignment, before DP)
  6. Watermarked + inserted       → DP T_k score, k=n_ins (after DP)

Key comparison: box 4 (UW+ins DP) vs box 6 (WM+ins DP) — gap shows watermark survives.

Usage:
    conda run -n llada python run_insertion_comparison.py --device cuda:0 --eps 0.2
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


def apply_insertion(tokens, epsilon, rng, vocab_size, mask_id):
    """Randomly insert epsilon-fraction random tokens at random positions."""
    n_ins = max(1, round(epsilon * len(tokens))) if epsilon > 0 and len(tokens) > 0 else 0
    if n_ins == 0:
        return list(tokens), 0
    result = list(tokens)
    for _ in range(n_ins):
        pos = rng.randint(0, len(result))
        new_tok = rng.randint(0, vocab_size - 1)
        while new_tok == mask_id:
            new_tok = rng.randint(0, vocab_size - 1)
        result.insert(pos, new_tok)
    return result, n_ins


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def naive_score(tokens, watermark_seed, position_offset, vocab_size, device="cpu"):
    """
    Standard aligned score: seed position i, score token i.
    Score = (1/L) * sum_i -log(1 - r_i[token_i])
    After insertions this is broken (positions misaligned).
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


def score_matrix(text_tokens, L_key, watermark_seed, position_offset, vocab_size, device="cpu"):
    """
    Build (L_key x L_text) score matrix.
    L_key = original (pre-edit) length. L_text = post-insertion length.
    device must match the device used during generation.
    """
    L_text = len(text_tokens)
    t = torch.tensor(text_tokens, dtype=torch.long, device=device)
    mat = torch.empty(L_key, L_text, dtype=torch.float32)
    for i in range(L_key):
        g = torch.Generator(device=device)
        g.manual_seed(int(watermark_seed + position_offset + i))
        r_i = torch.rand(vocab_size, generator=g, device=device).clamp_(1e-8, 1 - 1e-8)
        mat[i] = -torch.log1p(-r_i[t]).cpu()
    return mat


def dp_score(mat, L_key, L_text, k_max):
    """Kuditipudi-style DP alignment. Returns best T_k for k in 0..k_max."""
    if L_key == 0 or L_text == 0 or k_max < 0:
        return {k: float("-inf") for k in range(max(k_max, 0) + 1)}
    NEG_INF = float("-inf")
    D = torch.full((L_key + 1, L_text + 1, k_max + 1), NEG_INF, dtype=torch.float32)
    D[0, 0, 0] = 0.0
    for j in range(1, min(L_text + 1, k_max + 1)):
        D[0, j, j] = 0.0
    for d in range(1, L_key + L_text + 1):
        i_lo = max(1, d - L_text)
        i_hi = min(L_key, d)
        if i_lo > i_hi:
            continue
        is_ = torch.arange(i_lo, i_hi + 1)
        js_ = d - is_
        valid_m = js_ >= 1
        if valid_m.any():
            im, jm = is_[valid_m], js_[valid_m]
            s = mat[im - 1, jm - 1]
            D[im, jm, :] = torch.maximum(D[im, jm, :], D[im - 1, jm - 1, :] + s.unsqueeze(1))
        if k_max >= 1:
            D[is_, js_, 1:] = torch.maximum(D[is_, js_, 1:], D[is_ - 1, js_, :-1])
        if k_max >= 1:
            valid_i = js_ >= 1
            if valid_i.any():
                ii, ji = is_[valid_i], js_[valid_i]
                D[ii, ji, 1:] = torch.maximum(D[ii, ji, 1:], D[ii, ji - 1, :-1])
    results = {}
    for k in range(k_max + 1):
        best = NEG_INF
        for e in range(k + 1):
            total_len = L_key + L_text - e
            if total_len <= 0 or total_len % 2 != 0:
                continue
            q = total_len // 2
            val = D[L_key, L_text, e].item()
            if val > -1e30:
                cand = val / q
                if cand > best:
                    best = cand
        results[k] = best
    return results


# ---------------------------------------------------------------------------
# Z-score calibration
# ---------------------------------------------------------------------------

def calibrate_null_dp(L_key, L_text, k, n_trials=500):
    """Monte Carlo (mu, sigma) of dp_score under H0 (i.i.d. Exp(1) score matrix)."""
    scores = []
    for _ in range(n_trials):
        mat = -torch.log1p(-torch.rand(L_key, L_text).clamp(1e-8, 1 - 1e-8))
        s = dp_score(mat, L_key, L_text, k).get(k, float("-inf"))
        if math.isfinite(s):
            scores.append(s)
    if len(scores) < 2:
        return float("nan"), float("nan")
    mu = sum(scores) / len(scores)
    sigma = (sum((x - mu) ** 2 for x in scores) / (len(scores) - 1)) ** 0.5
    return mu, sigma


def to_z_naive(score, L):
    """Analytical z-score for naive aligned score: mu=1.0, sigma=1/sqrt(L)."""
    if not math.isfinite(score) or L <= 0:
        return float("nan")
    return (score - 1.0) * math.sqrt(L)


def to_z_dp(score, mu, sigma):
    """Z-score for DP score given calibrated null (mu, sigma)."""
    if not math.isfinite(score) or not math.isfinite(mu) or sigma <= 0:
        return float("nan")
    return (score - mu) / sigma


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
                        help="Fraction of tokens to insert (0.0-1.0)")
    parser.add_argument("--edit_seed", type=int, default=99)
    parser.add_argument("--n_cal", type=int, default=500,
                        help="Monte Carlo trials for DP null calibration")
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
    # Calibrate DP null distribution
    # -----------------------------------------------------------------------
    mean_len = round(sum(len(s["tokens"]) for s in wm_samples) / max(len(wm_samples), 1))
    k_cal = max(1, round(args.eps * mean_len))
    L_text_cal = mean_len + k_cal  # insertions make text longer
    print(f"\nCalibrating DP null (L_key={mean_len}, k={k_cal}, trials={args.n_cal})...")
    dp_mu, dp_sigma = calibrate_null_dp(mean_len, L_text_cal, k_cal, n_trials=args.n_cal)
    print(f"  DP null: μ={dp_mu:.4f}, σ={dp_sigma:.4f}")

    # -----------------------------------------------------------------------
    # Score all six distributions
    # -----------------------------------------------------------------------
    print(f"\n=== Scoring (eps={args.eps:.0%} insertion) ===")
    rng = random.Random(args.edit_seed)

    scores_uw           = []
    scores_wm           = []
    scores_uw_ins_naive = []
    scores_uw_ins_dp    = []
    scores_wm_ins_naive = []
    scores_wm_ins_dp    = []

    zscores_uw           = []
    zscores_wm           = []
    zscores_uw_ins_naive = []
    zscores_uw_ins_dp    = []
    zscores_wm_ins_naive = []
    zscores_wm_ins_dp    = []

    n = min(len(wm_samples), len(uw_samples))
    dev = args.device

    for i in range(n):
        prompt_len = wm_samples[i]["prompt_len"]
        wm_tok = wm_samples[i]["tokens"]
        uw_tok = uw_samples[i]["tokens"]

        scores_uw.append(naive_score(uw_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))
        scores_wm.append(naive_score(wm_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))

        uw_ins_tok, n_ins_uw = apply_insertion(uw_tok, args.eps, rng, args.vocab_size, args.mask_id)
        wm_ins_tok, n_ins_wm = apply_insertion(wm_tok, args.eps, rng, args.vocab_size, args.mask_id)
        L_uw_orig = len(uw_tok)
        L_wm_orig = len(wm_tok)

        # 3. UW+inserted naive
        scores_uw_ins_naive.append(naive_score(uw_ins_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))

        # 4. UW+inserted DP (k=n_ins, L_key=L_uw_orig)
        if L_uw_orig > 0 and len(uw_ins_tok) > 0:
            mat = score_matrix(uw_ins_tok, L_uw_orig, args.watermark_seed, prompt_len, args.vocab_size, dev)
            tk = dp_score(mat, L_uw_orig, len(uw_ins_tok), n_ins_uw)
            scores_uw_ins_dp.append(tk.get(n_ins_uw, float("nan")))
        else:
            scores_uw_ins_dp.append(float("nan"))

        # 5. WM+inserted naive
        scores_wm_ins_naive.append(naive_score(wm_ins_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))

        # 6. WM+inserted DP (k=n_ins, L_key=L_wm_orig)
        if L_wm_orig > 0 and len(wm_ins_tok) > 0:
            mat = score_matrix(wm_ins_tok, L_wm_orig, args.watermark_seed, prompt_len, args.vocab_size, dev)
            tk = dp_score(mat, L_wm_orig, len(wm_ins_tok), n_ins_wm)
            scores_wm_ins_dp.append(tk.get(n_ins_wm, float("nan")))
        else:
            scores_wm_ins_dp.append(float("nan"))

        zscores_uw.append(to_z_naive(scores_uw[-1], len(uw_tok)))
        zscores_wm.append(to_z_naive(scores_wm[-1], len(wm_tok)))
        zscores_uw_ins_naive.append(to_z_naive(scores_uw_ins_naive[-1], len(uw_ins_tok)))
        zscores_wm_ins_naive.append(to_z_naive(scores_wm_ins_naive[-1], len(wm_ins_tok)))
        zscores_uw_ins_dp.append(to_z_dp(scores_uw_ins_dp[-1], dp_mu, dp_sigma))
        zscores_wm_ins_dp.append(to_z_dp(scores_wm_ins_dp[-1], dp_mu, dp_sigma))

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    def stats(s):
        v = [x for x in s if math.isfinite(x)]
        return (sum(v)/len(v), sorted(v)[len(v)//2], len(v)) if v else (float("nan"), float("nan"), 0)

    print("\n=== Summary (raw scores) ===")
    print(f"{'Distribution':<35} {'Mean':>8} {'Median':>8} {'N':>5}  det@1.19")
    for name, sc in [
        ("1. UW clean",             scores_uw),
        ("2. WM clean",             scores_wm),
        ("3. UW+inserted (naive)",  scores_uw_ins_naive),
        ("4. UW+inserted (DP)",     scores_uw_ins_dp),
        ("5. WM+inserted (naive)",  scores_wm_ins_naive),
        ("6. WM+inserted (DP)",     scores_wm_ins_dp),
    ]:
        m, med, n_ = stats(sc)
        v = [x for x in sc if math.isfinite(x)]
        det = sum(x > 1.19 for x in v) / len(v) if v else float("nan")
        print(f"  {name:<33} {m:>8.4f} {med:>8.4f} {n_:>5}  {det:>7.1%}")

    print(f"\n=== Summary (z-scores, μ_DP={dp_mu:.3f}, σ_DP={dp_sigma:.3f}) ===")
    print(f"{'Distribution':<35} {'Mean z':>8} {'Med z':>8} {'N':>5}  det@z=2")
    for name, zsc in [
        ("1. UW clean",             zscores_uw),
        ("2. WM clean",             zscores_wm),
        ("3. UW+inserted (naive)",  zscores_uw_ins_naive),
        ("4. UW+inserted (DP)",     zscores_uw_ins_dp),
        ("5. WM+inserted (naive)",  zscores_wm_ins_naive),
        ("6. WM+inserted (DP)",     zscores_wm_ins_dp),
    ]:
        m, med, n_ = stats(zsc)
        v = [x for x in zsc if math.isfinite(x)]
        det = sum(x > 2.0 for x in v) / len(v) if v else float("nan")
        print(f"  {name:<33} {m:>8.3f} {med:>8.3f} {n_:>5}  {det:>7.1%}")

    # -----------------------------------------------------------------------
    # Save raw results
    # -----------------------------------------------------------------------
    out_path = out_dir / f"insertion_comparison_eps={args.eps:.2f}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "config": vars(args),
            "dp_null": {"mu": dp_mu, "sigma": dp_sigma},
            "scores_uw":           scores_uw,
            "scores_wm":           scores_wm,
            "scores_uw_ins_naive": scores_uw_ins_naive,
            "scores_uw_ins_dp":    scores_uw_ins_dp,
            "scores_wm_ins_naive": scores_wm_ins_naive,
            "scores_wm_ins_dp":    scores_wm_ins_dp,
            "zscores_uw":           zscores_uw,
            "zscores_wm":           zscores_wm,
            "zscores_uw_ins_naive": zscores_uw_ins_naive,
            "zscores_uw_ins_dp":    zscores_uw_ins_dp,
            "zscores_wm_ins_naive": zscores_wm_ins_naive,
            "zscores_wm_ins_dp":    zscores_wm_ins_dp,
        }, f, indent=2)
    print(f"\nSaved → {out_path}")

    # -----------------------------------------------------------------------
    # Box plot (raw scores)
    # -----------------------------------------------------------------------
    labels = [
        "UW\n(clean)",
        "WM\n(clean)",
        "UW+ins\n(naive)",
        "UW+ins\n(DP T_k)",
        "WM+ins\n(naive)",
        "WM+ins\n(DP T_k)",
    ]
    colors = ["#4C72B0", "#DD8452", "#8ea9d8", "#2b4f8a", "#e8a07a", "#8B2500"]

    def clean(s):
        return [x for x in s if math.isfinite(x)]

    data = [
        clean(scores_uw), clean(scores_wm),
        clean(scores_uw_ins_naive), clean(scores_uw_ins_dp),
        clean(scores_wm_ins_naive), clean(scores_wm_ins_dp),
    ]

    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.axvline(2.5, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(4.5, color="gray", linestyle=":", linewidth=0.8)
    ymax = ax.get_ylim()[1]
    ax.text(1.5, ymax * 0.97, "No edit", ha="center", fontsize=8, color="gray")
    ax.text(3.5, ymax * 0.97, f"{args.eps:.0%} insertion — UW", ha="center", fontsize=8, color="gray")
    ax.text(5.5, ymax * 0.97, f"{args.eps:.0%} insertion — WM", ha="center", fontsize=8, color="gray")

    ax.axhline(1.19, color="green", linestyle="--", linewidth=1.2, label="τ* = 1.19")
    ax.set_ylabel("Detection Score")
    ax.set_title(
        f"Gumbel-max Watermark Detection Scores: Naive vs DP Alignment\n"
        f"(gen_length={args.gen_length}, ε={args.eps:.0%} insertion, n={n})"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig_path = out_dir / f"insertion_comparison_eps={args.eps:.2f}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {fig_path}")

    # -----------------------------------------------------------------------
    # Z-score box plot
    # -----------------------------------------------------------------------
    zdata = [
        clean(zscores_uw), clean(zscores_wm),
        clean(zscores_uw_ins_naive), clean(zscores_uw_ins_dp),
        clean(zscores_wm_ins_naive), clean(zscores_wm_ins_dp),
    ]

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    bp2 = ax2.boxplot(zdata, tick_labels=labels, patch_artist=True, widths=0.55,
                      flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp2["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax2.axvline(2.5, color="gray", linestyle=":", linewidth=0.8)
    ax2.axvline(4.5, color="gray", linestyle=":", linewidth=0.8)
    ymax2 = ax2.get_ylim()[1]
    ax2.text(1.5, ymax2 * 0.97, "No edit", ha="center", fontsize=8, color="gray")
    ax2.text(3.5, ymax2 * 0.97, f"{args.eps:.0%} insertion — UW", ha="center", fontsize=8, color="gray")
    ax2.text(5.5, ymax2 * 0.97, f"{args.eps:.0%} insertion — WM", ha="center", fontsize=8, color="gray")

    ax2.axhline(2.0, color="green", linestyle="--", linewidth=1.2, label="z* = 2.0  (~2.3% FPR)")
    ax2.axhline(0.0, color="black", linestyle="-",  linewidth=0.6, alpha=0.3)
    ax2.set_ylabel("Z-score")
    ax2.set_title(
        f"Gumbel-max Watermark Z-scores: Naive vs DP Alignment\n"
        f"(gen_length={args.gen_length}, ε={args.eps:.0%} insertion, n={n}, "
        f"DP null: μ={dp_mu:.3f}, σ={dp_sigma:.3f})"
    )
    ax2.legend(fontsize=9)
    fig2.tight_layout()

    zfig_path = out_dir / f"insertion_zscores_eps={args.eps:.2f}.pdf"
    fig2.savefig(zfig_path, bbox_inches="tight")
    fig2.savefig(str(zfig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved z-score figure → {zfig_path}")


if __name__ == "__main__":
    main()
