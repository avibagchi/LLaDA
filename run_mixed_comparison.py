#!/usr/bin/env python3
"""
6-distribution comparison under a mix of deletion + insertion + substitution attacks.

Edit pipeline (applied in order):
  1. Substitute eps_sub fraction of tokens in-place (preserves alignment)
  2. Delete eps_del fraction of tokens (shortens text, shifts alignment)
  3. Insert eps_ins fraction of random tokens (lengthens text, shifts alignment)

Scoring:
  - Naive: score post-edit text at positions 0,1,...,L_edit-1 (alignment broken)
  - DP T_k: Kuditipudi alignment recovery with budget k = n_del + n_ins
    (substitutions don't shift alignment so they don't count toward k)

Distributions:
  1. UW clean   → naive
  2. WM clean   → naive
  3. UW+mixed   → naive
  4. UW+mixed   → DP T_k
  5. WM+mixed   → naive
  6. WM+mixed   → DP T_k

Usage:
    conda run -n llada python run_mixed_comparison.py --device cuda:0 \\
        --eps_del 0.1 --eps_ins 0.1 --eps_sub 0.1
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


# ---------------------------------------------------------------------------
# Edit functions
# ---------------------------------------------------------------------------

def apply_substitution(tokens, epsilon, rng, vocab_size, mask_id):
    """Replace epsilon-fraction of tokens in-place with random tokens."""
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


def apply_deletion(tokens, epsilon, rng):
    """Randomly delete epsilon-fraction of tokens."""
    n_del = max(1, round(epsilon * len(tokens))) if epsilon > 0 and len(tokens) > 0 else 0
    if n_del == 0:
        return list(tokens), 0
    positions = sorted(rng.sample(range(len(tokens)), min(n_del, len(tokens))), reverse=True)
    result = list(tokens)
    for p in positions:
        result.pop(p)
    return result, len(positions)


def apply_insertion(tokens, epsilon, rng, vocab_size, mask_id):
    """Insert epsilon-fraction random tokens at random positions."""
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


def apply_mixed_edits(tokens, eps_sub, eps_del, eps_ins, rng, vocab_size, mask_id):
    """
    Apply substitution, then deletion, then insertion.
    Returns (edited_tokens, n_del, n_ins, n_sub).
    DP budget = n_del + n_ins (substitutions don't shift alignment).
    """
    t, n_sub = apply_substitution(tokens, eps_sub, rng, vocab_size, mask_id)
    t, n_del = apply_deletion(t, eps_del, rng)
    t, n_ins = apply_insertion(t, eps_ins, rng, vocab_size, mask_id)
    return t, n_del, n_ins, n_sub


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def naive_score(tokens, watermark_seed, position_offset, vocab_size, device="cpu"):
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
    """Build (L_key x L_text) score matrix. L_key = original pre-edit length."""
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
    """
    Kuditipudi-style DP alignment. Returns best T_k for k in 0..k_max.
    T_k = max_{e=0..k} D(L_key, L_text, e) / q,  q = (L_key + L_text - e) / 2
    """
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
# Kuditipudi Algorithm 2: permutation test p-values
# ---------------------------------------------------------------------------

def permutation_pvalue_naive(actual_score, L_text, n_perms=200):
    """
    Algorithm 2 permutation test for naive score.
    Under H0, each -log(1-U) ~ Exp(1), so naive score ~ mean of L iid Exp(1).
    p-value = (count of null >= actual + 1) / (n_perms + 1)  [Laplace smoothed]
    """
    if not math.isfinite(actual_score) or L_text <= 0:
        return float("nan")
    count_geq = 0
    for _ in range(n_perms):
        null = -torch.log1p(-torch.rand(L_text).clamp(1e-8, 1 - 1e-8)).mean().item()
        if null >= actual_score:
            count_geq += 1
    return (count_geq + 1) / (n_perms + 1)


def permutation_pvalue_dp(actual_score, L_key, L_text, k_budget, n_perms=200):
    """
    Algorithm 2 permutation test for dp_score.
    Under H0, score matrix ~ iid Exp(1); sample null matrices directly.
    p-value = (count of null T_k >= actual + 1) / (n_perms + 1)  [Laplace smoothed]
    """
    if not math.isfinite(actual_score) or L_key <= 0 or L_text <= 0:
        return float("nan")
    count_geq = 0
    for _ in range(n_perms):
        null_mat = -torch.log1p(-torch.rand(L_key, L_text).clamp(1e-8, 1 - 1e-8))
        null_tk = dp_score(null_mat, L_key, L_text, k_budget).get(k_budget, float("-inf"))
        if null_tk >= actual_score:
            count_geq += 1
    return (count_geq + 1) / (n_perms + 1)


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
    parser.add_argument("--eps_del", type=float, default=0.1,
                        help="Fraction of tokens to delete")
    parser.add_argument("--eps_ins", type=float, default=0.1,
                        help="Fraction of tokens to insert")
    parser.add_argument("--eps_sub", type=float, default=0.1,
                        help="Fraction of tokens to substitute")
    parser.add_argument("--edit_seed", type=int, default=99)
    parser.add_argument("--n_perms", type=int, default=200,
                        help="Permutation trials for Algorithm 2 p-values")
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

    print(f"\nUsing Algorithm 2 permutation test (n_perms={args.n_perms}) for per-sample p-values.")

    # -----------------------------------------------------------------------
    # Score all six distributions
    # -----------------------------------------------------------------------
    eps_tag = f"del{args.eps_del:.2f}_ins{args.eps_ins:.2f}_sub{args.eps_sub:.2f}"
    print(f"\n=== Scoring (mixed edits: {eps_tag}) ===")
    rng = random.Random(args.edit_seed)

    scores_uw            = []
    scores_wm            = []
    scores_uw_mix_naive  = []
    scores_uw_mix_dp     = []
    scores_wm_mix_naive  = []
    scores_wm_mix_dp     = []

    pvalues_uw            = []
    pvalues_wm            = []
    pvalues_uw_mix_naive  = []
    pvalues_uw_mix_dp     = []
    pvalues_wm_mix_naive  = []
    pvalues_wm_mix_dp     = []

    n = min(len(wm_samples), len(uw_samples))
    dev = args.device

    for i in range(n):
        prompt_len = wm_samples[i]["prompt_len"]
        wm_tok = wm_samples[i]["tokens"]
        uw_tok = uw_samples[i]["tokens"]

        scores_uw.append(naive_score(uw_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))
        scores_wm.append(naive_score(wm_tok, args.watermark_seed, prompt_len, args.vocab_size, dev))

        uw_mix, n_del_uw, n_ins_uw, _ = apply_mixed_edits(
            uw_tok, args.eps_sub, args.eps_del, args.eps_ins, rng, args.vocab_size, args.mask_id)
        wm_mix, n_del_wm, n_ins_wm, _ = apply_mixed_edits(
            wm_tok, args.eps_sub, args.eps_del, args.eps_ins, rng, args.vocab_size, args.mask_id)

        L_uw_orig = len(uw_tok)
        L_wm_orig = len(wm_tok)

        scores_uw_mix_naive.append(
            naive_score(uw_mix, args.watermark_seed, prompt_len, args.vocab_size, dev))
        scores_wm_mix_naive.append(
            naive_score(wm_mix, args.watermark_seed, prompt_len, args.vocab_size, dev))

        # DP budget = n_del + n_ins (substitutions preserve alignment order)
        k_uw = n_del_uw + n_ins_uw
        if L_uw_orig > 0 and len(uw_mix) > 0 and k_uw >= 0:
            mat = score_matrix(uw_mix, L_uw_orig, args.watermark_seed, prompt_len, args.vocab_size, dev)
            tk = dp_score(mat, L_uw_orig, len(uw_mix), k_uw)
            scores_uw_mix_dp.append(tk.get(k_uw, float("nan")))
        else:
            scores_uw_mix_dp.append(float("nan"))

        k_wm = n_del_wm + n_ins_wm
        if L_wm_orig > 0 and len(wm_mix) > 0 and k_wm >= 0:
            mat = score_matrix(wm_mix, L_wm_orig, args.watermark_seed, prompt_len, args.vocab_size, dev)
            tk = dp_score(mat, L_wm_orig, len(wm_mix), k_wm)
            scores_wm_mix_dp.append(tk.get(k_wm, float("nan")))
        else:
            scores_wm_mix_dp.append(float("nan"))

        pvalues_uw.append(permutation_pvalue_naive(scores_uw[-1], len(uw_tok), args.n_perms))
        pvalues_wm.append(permutation_pvalue_naive(scores_wm[-1], len(wm_tok), args.n_perms))
        pvalues_uw_mix_naive.append(permutation_pvalue_naive(scores_uw_mix_naive[-1], len(uw_mix), args.n_perms))
        pvalues_wm_mix_naive.append(permutation_pvalue_naive(scores_wm_mix_naive[-1], len(wm_mix), args.n_perms))
        pvalues_uw_mix_dp.append(permutation_pvalue_dp(scores_uw_mix_dp[-1], L_uw_orig, len(uw_mix), k_uw, args.n_perms))
        pvalues_wm_mix_dp.append(permutation_pvalue_dp(scores_wm_mix_dp[-1], L_wm_orig, len(wm_mix), k_wm, args.n_perms))

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
        ("1. UW clean",          scores_uw),
        ("2. WM clean",          scores_wm),
        ("3. UW+mixed (naive)",  scores_uw_mix_naive),
        ("4. UW+mixed (DP)",     scores_uw_mix_dp),
        ("5. WM+mixed (naive)",  scores_wm_mix_naive),
        ("6. WM+mixed (DP)",     scores_wm_mix_dp),
    ]:
        m, med, n_ = stats(sc)
        v = [x for x in sc if math.isfinite(x)]
        det = sum(x > 1.19 for x in v) / len(v) if v else float("nan")
        print(f"  {name:<33} {m:>8.4f} {med:>8.4f} {n_:>5}  {det:>7.1%}")

    print(f"\n=== Summary (Algorithm 2 p-values, n_perms={args.n_perms}) ===")
    print(f"{'Distribution':<35} {'Mean p':>8} {'Med p':>8} {'N':>5}  det@p<0.05")
    for name, pvals in [
        ("1. UW clean",          pvalues_uw),
        ("2. WM clean",          pvalues_wm),
        ("3. UW+mixed (naive)",  pvalues_uw_mix_naive),
        ("4. UW+mixed (DP)",     pvalues_uw_mix_dp),
        ("5. WM+mixed (naive)",  pvalues_wm_mix_naive),
        ("6. WM+mixed (DP)",     pvalues_wm_mix_dp),
    ]:
        m, med, n_ = stats(pvals)
        v = [x for x in pvals if math.isfinite(x)]
        det = sum(x < 0.05 for x in v) / len(v) if v else float("nan")
        print(f"  {name:<33} {m:>8.4f} {med:>8.4f} {n_:>5}  {det:>7.1%}")

    # -----------------------------------------------------------------------
    # Save raw results
    # -----------------------------------------------------------------------
    out_path = out_dir / f"mixed_comparison_{eps_tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "config": vars(args),
            "scores_uw":             scores_uw,
            "scores_wm":             scores_wm,
            "scores_uw_mix_naive":   scores_uw_mix_naive,
            "scores_uw_mix_dp":      scores_uw_mix_dp,
            "scores_wm_mix_naive":   scores_wm_mix_naive,
            "scores_wm_mix_dp":      scores_wm_mix_dp,
            "pvalues_uw":            pvalues_uw,
            "pvalues_wm":            pvalues_wm,
            "pvalues_uw_mix_naive":  pvalues_uw_mix_naive,
            "pvalues_uw_mix_dp":     pvalues_uw_mix_dp,
            "pvalues_wm_mix_naive":  pvalues_wm_mix_naive,
            "pvalues_wm_mix_dp":     pvalues_wm_mix_dp,
        }, f, indent=2)
    print(f"\nSaved → {out_path}")

    # -----------------------------------------------------------------------
    # Box plot (raw scores)
    # -----------------------------------------------------------------------
    labels = [
        "UW\n(clean)",
        "WM\n(clean)",
        "UW+mix\n(naive)",
        "UW+mix\n(DP T_k)",
        "WM+mix\n(naive)",
        "WM+mix\n(DP T_k)",
    ]
    colors = ["#4C72B0", "#DD8452", "#8ea9d8", "#2b4f8a", "#e8a07a", "#8B2500"]

    def clean(s):
        return [x for x in s if math.isfinite(x)]

    data = [
        clean(scores_uw),
        clean(scores_wm),
        clean(scores_uw_mix_naive),
        clean(scores_uw_mix_dp),
        clean(scores_wm_mix_naive),
        clean(scores_wm_mix_dp),
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
    ax.text(1.5, ymax * 0.97, "No edit",     ha="center", fontsize=8, color="gray")
    ax.text(3.5, ymax * 0.97, "Mixed — UW",  ha="center", fontsize=8, color="gray")
    ax.text(5.5, ymax * 0.97, "Mixed — WM",  ha="center", fontsize=8, color="gray")

    ax.axhline(1.19, color="green", linestyle="--", linewidth=1.2, label="τ* = 1.19")
    ax.set_ylabel("Detection Score")
    ax.set_title(
        f"Gumbel-max Watermark Detection: Mixed Edits (sub={args.eps_sub:.0%}, "
        f"del={args.eps_del:.0%}, ins={args.eps_ins:.0%})\n"
        f"(gen_length={args.gen_length}, n={n}, DP budget=n_del+n_ins)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig_path = out_dir / f"mixed_comparison_{eps_tag}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {fig_path}")

    # -----------------------------------------------------------------------
    # P-value box plot (Algorithm 2)
    # -----------------------------------------------------------------------
    pdata = [
        clean(pvalues_uw),
        clean(pvalues_wm),
        clean(pvalues_uw_mix_naive),
        clean(pvalues_uw_mix_dp),
        clean(pvalues_wm_mix_naive),
        clean(pvalues_wm_mix_dp),
    ]

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    bp2 = ax2.boxplot(pdata, tick_labels=labels, patch_artist=True, widths=0.55,
                      flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp2["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax2.axvline(2.5, color="gray", linestyle=":", linewidth=0.8)
    ax2.axvline(4.5, color="gray", linestyle=":", linewidth=0.8)
    ax2.text(1.5, 0.97, "No edit",     ha="center", fontsize=8, color="gray", transform=ax2.get_xaxis_transform())
    ax2.text(3.5, 0.97, "Mixed — UW",  ha="center", fontsize=8, color="gray", transform=ax2.get_xaxis_transform())
    ax2.text(5.5, 0.97, "Mixed — WM",  ha="center", fontsize=8, color="gray", transform=ax2.get_xaxis_transform())

    ax2.axhline(0.05, color="green", linestyle="--", linewidth=1.2, label="α = 0.05  (5% FPR)")
    ax2.set_ylabel("p-value (Algorithm 2)")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title(
        f"Gumbel-max Watermark p-values: Mixed Edits (sub={args.eps_sub:.0%}, "
        f"del={args.eps_del:.0%}, ins={args.eps_ins:.0%})\n"
        f"(gen_length={args.gen_length}, n={n}, n_perms={args.n_perms})"
    )
    ax2.legend(fontsize=9)
    fig2.tight_layout()

    pfig_path = out_dir / f"mixed_pvalues_{eps_tag}.pdf"
    fig2.savefig(pfig_path, bbox_inches="tight")
    fig2.savefig(str(pfig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved p-value figure → {pfig_path}")


if __name__ == "__main__":
    main()
