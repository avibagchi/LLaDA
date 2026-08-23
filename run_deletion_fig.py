#!/usr/bin/env python3
"""
Reproduces Figures 1 and 2 from the paper, for the deletion robustness case.

Compares:
  - Watermarked text with eps-fraction tokens deleted → scored with DP T_k (k=n_del)
  - Unwatermarked text with eps-fraction tokens deleted → scored with DP T_k (k=n_del)

This is a fair comparison (same k budget, same edit applied to both sides).

Produces:
  - deletion_scores_fig1.pdf : box plot of T_k distributions (Fig 1 analogue)
  - deletion_scores_fig2.pdf : P(T_k > tau) vs tau (Fig 2 analogue)
  - deletion_scores.json     : raw results for further analysis

Usage:
    conda run -n llada python run_deletion_fig.py --device cuda:3
    conda run -n llada python run_deletion_fig.py --device cuda:3 --n_samples 200
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

EPS_VALUES = [0.0, 0.10, 0.20, 0.30]
TAU_RANGE = np.linspace(0.7, 2.2, 120)
EOS_IDS = {50256, 2, 126081}


# ---------------------------------------------------------------------------
# Helpers
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
        text = f"You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n\n{context}\n{input_text}"
    elif context:
        text = f"You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n\n{context}"
    elif input_text:
        text = f"You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n\n{input_text}"
    else:
        return None
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def trim_eos(tokens):
    for i, t in enumerate(tokens):
        if t in EOS_IDS:
            return tokens[:i]
    return tokens


def apply_deletion(tokens, epsilon, rng):
    """Randomly delete epsilon-fraction of tokens. Returns (edited, n_del)."""
    n = max(1, round(epsilon * len(tokens))) if epsilon > 0 else 0
    if n == 0:
        return list(tokens), 0
    positions = sorted(rng.sample(range(len(tokens)), min(n, len(tokens))), reverse=True)
    result = list(tokens)
    for p in positions:
        result.pop(p)
    return result, len(positions)


# ---------------------------------------------------------------------------
# Score matrix + DP (identical logic to run_gumbel_dp_robustness.py)
# ---------------------------------------------------------------------------

def compute_score_matrix(text_tokens, L_key, watermark_seed, position_offset, vocab_size):
    L_text = len(text_tokens)
    text_t = torch.tensor(text_tokens, dtype=torch.long)
    score_matrix = torch.empty(L_key, L_text, dtype=torch.float32)
    for i in range(L_key):
        g = torch.Generator()
        g.manual_seed(watermark_seed + position_offset + i)
        r_i = torch.rand(vocab_size, generator=g).clamp_(1e-8, 1.0 - 1e-8)
        score_matrix[i] = -torch.log1p(-r_i[text_t])
    return score_matrix


def dp_score(score_matrix, L_key, L_text, k_max):
    """Returns T_k for k in 0..k_max."""
    if L_key == 0 or L_text == 0:
        return {k: float("-inf") for k in range(k_max + 1)}
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
            s = score_matrix[im - 1, jm - 1]
            D[im, jm, :] = torch.maximum(D[im, jm, :], D[im - 1, jm - 1, :] + s.unsqueeze(1))
        if k_max >= 1:
            D[is_, js_, 1:] = torch.maximum(D[is_, js_, 1:], D[is_ - 1, js_, :-1])
            valid_i = js_ >= 1
            if valid_i.any():
                ii, ji = is_[valid_i], js_[valid_i]
                D[ii, ji, 1:] = torch.maximum(D[ii, ji, 1:], D[ii, ji - 1, :-1])
    results = {}
    for k in range(k_max + 1):
        best = NEG_INF
        for e in range(k + 1):
            total = L_key + L_text - e
            if total <= 0 or total % 2 != 0:
                continue
            q = total // 2
            val = D[L_key, L_text, e].item()
            if val > -1e30:
                cand = val / q
                if cand > best:
                    best = cand
        results[k] = best
    return results


def score_tokens(tokens, prompt_len, watermark_seed, vocab_size, n_del):
    """Score token list under DP with budget k = n_del."""
    if len(tokens) == 0:
        return float("-inf")
    L = len(tokens)
    k_max = max(n_del, 0)
    mat = compute_score_matrix(tokens, L, watermark_seed, prompt_len, vocab_size)
    tk = dp_score(mat, L, L, k_max)
    return tk.get(k_max, float("-inf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", default="water-bench-sampled_100_seed43.jsonl")
    parser.add_argument("--output_dir", default="water-bench-results/json-outputs")
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--watermark_seed", type=int, default=42)
    parser.add_argument("--gen_length", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--block_length", type=int, default=25)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--edit_seed", type=int, default=99)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
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
    # Phase 1: generate watermarked sequences
    # -----------------------------------------------------------------------
    print("\n=== Phase 1: Generating watermarked sequences ===")
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
    # Phase 2: generate unwatermarked sequences
    # -----------------------------------------------------------------------
    print("\n=== Phase 2: Generating unwatermarked sequences ===")
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
    # Phase 3: Score under deletion + DP alignment
    # -----------------------------------------------------------------------
    print("\n=== Phase 3: Scoring under deletion DP ===")
    rng = random.Random(args.edit_seed)

    # scores[eps] = {"wm": [...], "uw": [...]}
    all_scores = {}

    for eps in EPS_VALUES:
        wm_scores, uw_scores = [], []
        for i in range(len(wm_samples)):
            wm_tok = wm_samples[i]["tokens"]
            uw_tok = uw_samples[i]["tokens"]
            prompt_len = wm_samples[i]["prompt_len"]

            wm_del, n_del_wm = apply_deletion(wm_tok, eps, rng)
            uw_del, n_del_uw = apply_deletion(uw_tok, eps, rng)

            # Score with k = actual deletions applied to that sequence
            s_wm = score_tokens(wm_del, prompt_len, args.watermark_seed, args.vocab_size, n_del_wm)
            s_uw = score_tokens(uw_del, prompt_len, args.watermark_seed, args.vocab_size, n_del_uw)

            wm_scores.append(s_wm)
            uw_scores.append(s_uw)

        wm_valid = [s for s in wm_scores if s > -1e30]
        uw_valid = [s for s in uw_scores if s > -1e30]
        print(f"  eps={eps:.2f}: wm_mean={sum(wm_valid)/len(wm_valid):.4f}  "
              f"uw_mean={sum(uw_valid)/len(uw_valid):.4f}  "
              f"n={len(wm_valid)}")
        all_scores[eps] = {"wm": wm_scores, "uw": uw_scores}

    # -----------------------------------------------------------------------
    # Save raw results
    # -----------------------------------------------------------------------
    results_path = out_dir / "deletion_scores.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "config": vars(args),
            "eps_values": EPS_VALUES,
            "scores": {str(eps): v for eps, v in all_scores.items()},
        }, f, indent=2)
    print(f"\nSaved raw scores to {results_path}")

    # -----------------------------------------------------------------------
    # Figure 1 analogue: box plots of T_k distributions
    # -----------------------------------------------------------------------
    n_eps = len(EPS_VALUES)
    fig, axes = plt.subplots(1, n_eps, figsize=(4 * n_eps, 5), sharey=False)
    if n_eps == 1:
        axes = [axes]

    for ax, eps in zip(axes, EPS_VALUES):
        wm_v = [s for s in all_scores[eps]["wm"] if s > -1e30]
        uw_v = [s for s in all_scores[eps]["uw"] if s > -1e30]
        bp = ax.boxplot([uw_v, wm_v], labels=["No Watermark", "Watermarked"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        for median in bp["medians"]:
            median.set_color("red")
        ax.set_title(f"Deletion ε={eps:.0%}")
        ax.set_ylabel("DP Score $T_k$")
        ax.axhline(1.19, color="green", linestyle="--", linewidth=1, label="τ*=1.19")
        if eps == EPS_VALUES[0]:
            ax.legend(fontsize=8)

    fig.suptitle("DP Score Distributions Under Deletion Attack\n(Fair comparison: both sides deleted at same ε, scored at k=n_del)",
                 fontsize=10)
    fig.tight_layout()
    fig1_path = out_dir / "deletion_scores_fig1.pdf"
    fig.savefig(fig1_path, bbox_inches="tight")
    fig.savefig(str(fig1_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Fig 1 → {fig1_path}")

    # -----------------------------------------------------------------------
    # Figure 2 analogue: P(T_k > tau) vs tau
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, n_eps, figsize=(4 * n_eps, 4), sharey=True)
    if n_eps == 1:
        axes = [axes]

    for ax, eps in zip(axes, EPS_VALUES):
        wm_v = [s for s in all_scores[eps]["wm"] if s > -1e30]
        uw_v = [s for s in all_scores[eps]["uw"] if s > -1e30]
        N_wm = len(wm_v)
        N_uw = len(uw_v)
        wm_curve = [100 * sum(s > tau for s in wm_v) / N_wm for tau in TAU_RANGE]
        uw_curve = [100 * sum(s > tau for s in uw_v) / N_uw for tau in TAU_RANGE]
        ax.plot(TAU_RANGE, wm_curve, "r-s", markersize=2, label="Watermarked")
        ax.plot(TAU_RANGE, uw_curve, "b-o", markersize=2, label="No Watermark")
        ax.axvline(1.19, color="green", linestyle="--", linewidth=1, label="τ*=1.19")
        ax.set_xlabel("Detection Threshold τ")
        ax.set_title(f"Deletion ε={eps:.0%}")
        if eps == EPS_VALUES[0]:
            ax.set_ylabel("% Prompts with $T_k$ > τ")
            ax.legend(fontsize=8)
        ax.set_ylim(-2, 102)

    fig.suptitle("Completeness & Soundness Under Deletion Attack\n(Detector given budget k = actual deletions)",
                 fontsize=10)
    fig.tight_layout()
    fig2_path = out_dir / "deletion_scores_fig2.pdf"
    fig.savefig(fig2_path, bbox_inches="tight")
    fig.savefig(str(fig2_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Fig 2 → {fig2_path}")

    # -----------------------------------------------------------------------
    # Quick text summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"{'eps':>5} {'wm_mean':>9} {'uw_mean':>9} {'gap':>7}  "
          f"TPR@1.19  FPR@1.19")
    for eps in EPS_VALUES:
        wm_v = [s for s in all_scores[eps]["wm"] if s > -1e30]
        uw_v = [s for s in all_scores[eps]["uw"] if s > -1e30]
        wm_mean = sum(wm_v) / len(wm_v)
        uw_mean = sum(uw_v) / len(uw_v)
        tpr = sum(s > 1.19 for s in wm_v) / len(wm_v)
        fpr = sum(s > 1.19 for s in uw_v) / len(uw_v)
        print(f"  {eps:.2f}  {wm_mean:>9.4f}  {uw_mean:>9.4f}  {wm_mean-uw_mean:>7.4f}  "
              f"{tpr:>7.1%}  {fpr:>7.1%}")


if __name__ == "__main__":
    main()
