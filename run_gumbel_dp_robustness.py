#!/usr/bin/env python3
"""
DP-robust watermark detection ablation for Gumbel-max watermarking.

Generates Gumbel-max watermarked text, applies bounded edits (substitutions,
deletions, insertions, or mixed), then detects using Kuditipudi-style 4D DP
alignment reduced to a 3D recurrence via the identity q = (L + L' - e) / 2.

DP recurrence (from image, Kuditipudi et al. 2024):
  D(i, j, e) = max {
    D(i-1, j-1, e) + score[i-1, j-1],   # match key pos i with text pos j
    D(i-1, j, e-1),                        # deletion  (skip key pos, 1 edit)
    D(i, j-1, e-1),                        # insertion (skip text pos, 1 edit)
  }
  D(0, 0, 0) = 0;  all others = -inf

  T_k(y) = max_{e=0..k, (L+L'-e) even, q=(L+L'-e)/2>0} D(L, L', e) / q
  Detected if T_k(y) > tau = 1.19

Note: substitutions in the text appear as "bad matches" in the DP (they don't
consume edit budget but reduce the score). Insertions/deletions shift alignment
and require edit budget k to compensate.

Ablation axes:
  edit_type ∈ {sub, del, ins, mixed}
  epsilon   ∈ {0.0, 0.10, 0.20, 0.30}   (fraction of tokens edited)
  k         extracted at multipliers {0, 0.5, 1.0, 1.5, 2.0} × actual_shift_edits

Usage:
  python run_gumbel_dp_robustness.py --device cuda:0
  python run_gumbel_dp_robustness.py --device cuda:0 --gen_length 128 --n_samples 50
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

DETECTION_THRESHOLD = 1.19
EPSILON_VALUES = [0.0, 0.10, 0.20, 0.30]
EDIT_TYPES = ["sub", "del", "ins", "mixed"]
K_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_prompt(context, input_text, tokenizer):
    if context and input_text:
        user_content = (
            "You are a helpful assistant, please answer the following question "
            f"with financial knowledge within 300 words:\n\n{context}\n{input_text}"
        )
    elif context:
        user_content = (
            "You are a helpful assistant, please answer the following question "
            f"with financial knowledge within 300 words:\n\n{context}"
        )
    elif input_text:
        user_content = (
            "You are a helpful assistant, please answer the following question "
            f"with financial knowledge within 300 words:\n\n{input_text}"
        )
    else:
        return None
    msgs = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Edit simulation
# ---------------------------------------------------------------------------

def apply_edits(tokens, epsilon, edit_type, vocab_size, rng):
    """
    Apply random bounded edits to a token list.

    Returns (edited_tokens, n_sub, n_del, n_ins).
    Substitutions don't shift alignment; only deletions + insertions consume
    DP edit budget k.
    """
    tokens = list(tokens)
    L = len(tokens)
    n_target = max(1, round(epsilon * L)) if epsilon > 0 else 0

    n_sub = n_del = n_ins = 0
    if n_target == 0:
        return tokens, 0, 0, 0

    if edit_type == "sub":
        n = min(n_target, L)
        for p in rng.sample(range(L), n):
            tokens[p] = rng.randint(0, vocab_size - 1)
            n_sub += 1

    elif edit_type == "del":
        n = min(n_target, len(tokens))
        for p in sorted(rng.sample(range(len(tokens)), n), reverse=True):
            tokens.pop(p)
            n_del += 1

    elif edit_type == "ins":
        for _ in range(n_target):
            pos = rng.randint(0, len(tokens))
            tokens.insert(pos, rng.randint(0, vocab_size - 1))
            n_ins += 1

    elif edit_type == "mixed":
        third = max(1, n_target // 3)
        # Substitutions
        n = min(third, len(tokens))
        for p in rng.sample(range(len(tokens)), n):
            tokens[p] = rng.randint(0, vocab_size - 1)
            n_sub += 1
        # Deletions
        n = min(third, len(tokens))
        for p in sorted(rng.sample(range(len(tokens)), n), reverse=True):
            tokens.pop(p)
            n_del += 1
        # Insertions
        for _ in range(third):
            pos = rng.randint(0, len(tokens))
            tokens.insert(pos, rng.randint(0, vocab_size - 1))
            n_ins += 1

    return tokens, n_sub, n_del, n_ins


# ---------------------------------------------------------------------------
# Score matrix
# ---------------------------------------------------------------------------

def compute_score_matrix(text_tokens, L_key, watermark_seed, position_offset, vocab_size):
    """
    Build score_matrix[i, j] = -log(1 - r_{i, text_tokens[j]}).

    r_{i, v} ~ Unif(0,1) with seed = watermark_seed + position_offset + i,
    matching the seeding used in apply_aaronson_gumbel_watermark (position_offset=0
    there, but absolute position = position_offset + i passed in generate()).
    """
    L_text = len(text_tokens)
    text_t = torch.tensor(text_tokens, dtype=torch.long)
    score_matrix = torch.empty(L_key, L_text, dtype=torch.float32)

    for i in range(L_key):
        g = torch.Generator()
        g.manual_seed(watermark_seed + position_offset + i)
        r_i = torch.rand(vocab_size, generator=g).clamp_(1e-8, 1.0 - 1e-8)
        score_matrix[i] = -torch.log1p(-r_i[text_t])

    return score_matrix


# ---------------------------------------------------------------------------
# DP detection (diagonal sweep, vectorised over k dimension)
# ---------------------------------------------------------------------------

def dp_detect(score_matrix, L_key, L_text, k_max):
    """
    Run the Kuditipudi DP with budget k_max and return a dict
    {k: T_k} for k in 0..k_max (evaluated by restricting the max-over-e).

    q = (L_key + L_text - e) / 2  is the implicit match count.
    D is stored as (L_key+1) × (L_text+1) × (k_max+1) float32 tensor.
    Diagonal sweep ensures all source cells are computed before use.
    """
    if L_key == 0 or L_text == 0:
        return {k: float("-inf") for k in range(k_max + 1)}

    NEG_INF = float("-inf")
    D = torch.full((L_key + 1, L_text + 1, k_max + 1), NEG_INF, dtype=torch.float32)
    D[0, 0, 0] = 0.0

    # Boundary paths that start with insertions (consume text before any key match).
    # The diagonal sweep skips i=0, so we initialize these explicitly.
    # D[0, j, j] = 0 means j insertions used, 0 matches, score = 0.
    for j in range(1, min(L_text + 1, k_max + 1)):
        D[0, j, j] = 0.0

    for d in range(1, L_key + L_text + 1):
        i_lo = max(1, d - L_text)
        i_hi = min(L_key, d)
        if i_lo > i_hi:
            continue

        is_ = torch.arange(i_lo, i_hi + 1)   # shape [N]
        js_ = d - is_                          # j = d - i

        # --- Match: D[i,j,e] = max(D[i,j,e], D[i-1,j-1,e] + s[i-1,j-1]) ---
        valid_m = js_ >= 1
        if valid_m.any():
            im = is_[valid_m]
            jm = js_[valid_m]
            s = score_matrix[im - 1, jm - 1]          # [Nm]
            prev = D[im - 1, jm - 1, :]               # [Nm, k+1]
            D[im, jm, :] = torch.maximum(D[im, jm, :], prev + s.unsqueeze(1))

        # --- Deletion: D[i,j,e] = max(D[i,j,e], D[i-1,j,e-1])  (e >= 1) ---
        if k_max >= 1:
            prev_del = D[is_ - 1, js_, :-1]           # [N, k_max]
            D[is_, js_, 1:] = torch.maximum(D[is_, js_, 1:], prev_del)

        # --- Insertion: D[i,j,e] = max(D[i,j,e], D[i,j-1,e-1])  (e>=1, j>=1) ---
        if k_max >= 1:
            valid_i = js_ >= 1
            if valid_i.any():
                ii = is_[valid_i]
                ji = js_[valid_i]
                prev_ins = D[ii, ji - 1, :-1]         # [Ni, k_max]
                D[ii, ji, 1:] = torch.maximum(D[ii, ji, 1:], prev_ins)

    # Extract T_k for all k in 0..k_max
    tk_scores = {}
    for k in range(k_max + 1):
        best = NEG_INF
        for e in range(k + 1):
            total = L_key + L_text - e
            if total <= 0 or total % 2 != 0:
                continue
            q = total // 2
            if q <= 0:
                continue
            val = D[L_key, L_text, e].item()
            if val > -1e30:
                candidate = val / q
                if candidate > best:
                    best = candidate
        tk_scores[k] = best

    return tk_scores


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
    parser.add_argument("--gen_length", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--block_length", type=int, default=25)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--watermark_steps", type=int, default=200,
                        help="Number of diffusion steps to apply watermark (t_end)")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--edit_seed", type=int, default=99)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gumbel_dp_robustness.json"

    # Load model
    print(f"Loading LLaDA from {args.model_path} on {args.device}...")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    special_token_ids = get_special_token_ids(tokenizer)
    eos_ids = {50256, 2, 126081}

    # Load prompts
    entries = load_jsonl(args.jsonl_file)[: args.n_samples]
    prompt_tokens_list = []
    for entry in entries:
        text = format_prompt(entry.get("context", ""), entry.get("input", ""), tokenizer)
        prompt_tokens_list.append(tokenizer(text)["input_ids"] if text else None)
    print(f"Loaded {len(entries)} prompts")

    # ------------------------------------------------------------------
    # Phase 1: generate watermarked token sequences
    # ------------------------------------------------------------------
    print("\n=== Phase 1: Generating Gumbel-max watermarked sequences ===")
    samples = []  # list of {sample_id, prompt_length, gen_tokens}

    for idx, (entry, prompt_toks) in enumerate(zip(entries, prompt_tokens_list)):
        if prompt_toks is None:
            continue
        prompt_tensor = torch.tensor([prompt_toks]).to(args.device)
        with torch.no_grad():
            out = generate(
                model=model,
                prompt=prompt_tensor,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                remasking="low_confidence",
                mask_id=args.mask_id,
                watermark_type="aaronson",
                vocab_size=args.vocab_size,
                special_token_ids=special_token_ids,
                aaronson_seed=args.watermark_seed,
                watermark_steps=args.watermark_steps,
            )
        gen_toks = out[0, len(prompt_toks):]
        # Trim at EOS
        actual_len = gen_toks.shape[0]
        for j, t in enumerate(gen_toks):
            if t.item() in eos_ids:
                actual_len = j
                break
        samples.append({
            "sample_id": idx,
            "prompt_length": len(prompt_toks),
            "gen_tokens": gen_toks[:actual_len].tolist(),
        })
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(entries)}")

    print(f"Generated {len(samples)} samples\n")

    # ------------------------------------------------------------------
    # Phase 2: DP robustness ablation
    # ------------------------------------------------------------------
    print("=== Phase 2: DP robustness ablation ===")
    rng = random.Random(args.edit_seed)
    all_results = []

    for sample in tqdm(samples, desc="Samples"):
        sid = sample["sample_id"]
        orig_tokens = sample["gen_tokens"]
        prompt_len = sample["prompt_length"]
        # Key positions 0..L_key-1 use seeds watermark_seed + prompt_len + i
        # (position_offset = prompt_len matches generation, which uses absolute pos)
        L_key = len(orig_tokens)

        sample_rec = {
            "sample_id": sid,
            "original_length": L_key,
            "edits": {},
        }

        for edit_type in EDIT_TYPES:
            sample_rec["edits"][edit_type] = {}

            for epsilon in EPSILON_VALUES:
                # Apply edits once per (edit_type, epsilon) combo
                edited, n_sub, n_del, n_ins = apply_edits(
                    orig_tokens, epsilon, edit_type, args.vocab_size, rng
                )
                L_text = len(edited)
                n_shift = n_del + n_ins  # alignment-shifting edits (consume DP budget)

                # k_max = 2 * max_k_multiplier * n_shift (enough to cover all multipliers)
                k_max_budget = int(math.ceil(2.0 * max(n_shift, 1)))
                # For sub-only, n_shift=0 → k_max=2; DP with k=0 still gives the right score.

                # Build score matrix once (expensive: L_key × vocab_size random numbers)
                score_mat = compute_score_matrix(
                    edited, L_key, args.watermark_seed, prompt_len, args.vocab_size
                )

                # Run DP once, get T_k for k = 0..k_max
                tk_all = dp_detect(score_mat, L_key, L_text, k_max_budget)

                # Extract results for each K_MULTIPLIER × n_shift
                k_results = {}
                for mult in K_MULTIPLIERS:
                    k_val = int(round(mult * n_shift))
                    k_clamped = min(k_val, k_max_budget)
                    score = tk_all.get(k_clamped, float("-inf"))
                    k_results[f"k_mult={mult:.1f}"] = {
                        "k_budget": k_clamped,
                        "n_shift_edits": n_shift,
                        "dp_score": round(score, 6) if score > -1e30 else None,
                        "detected": score > DETECTION_THRESHOLD if score > -1e30 else False,
                    }

                eps_key = f"eps={epsilon:.2f}"
                sample_rec["edits"][edit_type][eps_key] = {
                    "n_sub": n_sub,
                    "n_del": n_del,
                    "n_ins": n_ins,
                    "edited_length": L_text,
                    "k_results": k_results,
                }

        all_results.append(sample_rec)

    # ------------------------------------------------------------------
    # Summarize
    # ------------------------------------------------------------------
    print("\n=== Summary (detection rate @ tau=1.19) ===")
    summary = {}
    for edit_type in EDIT_TYPES:
        summary[edit_type] = {}
        for epsilon in EPSILON_VALUES:
            eps_key = f"eps={epsilon:.2f}"
            summary[edit_type][eps_key] = {}
            for mult in K_MULTIPLIERS:
                mk = f"k_mult={mult:.1f}"
                detected = 0
                scores = []
                for r in all_results:
                    rec = r["edits"].get(edit_type, {}).get(eps_key, {}).get("k_results", {}).get(mk, {})
                    if rec.get("detected", False):
                        detected += 1
                    if rec.get("dp_score") is not None:
                        scores.append(rec["dp_score"])
                det_rate = detected / len(all_results) if all_results else 0
                avg_sc = sum(scores) / len(scores) if scores else None
                summary[edit_type][eps_key][mk] = {
                    "detection_rate": round(det_rate, 4),
                    "avg_score": round(avg_sc, 4) if avg_sc is not None else None,
                    "n_detected": detected,
                    "n_total": len(all_results),
                }
                print(
                    f"  {edit_type:6s}  {eps_key}  {mk}  "
                    f"det={detected}/{len(all_results)} ({det_rate*100:.1f}%)"
                    + (f"  avg_score={avg_sc:.3f}" if avg_sc else "")
                )

    # Save
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": vars(args),
        "detection_threshold": DETECTION_THRESHOLD,
        "epsilon_values": EPSILON_VALUES,
        "edit_types": EDIT_TYPES,
        "k_multipliers": K_MULTIPLIERS,
        "summary": summary,
        "per_sample_results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
