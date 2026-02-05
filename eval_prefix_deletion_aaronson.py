#!/usr/bin/env python3
"""
Evaluate Aaronson watermark robustness to deletions (prefix, suffix, random, middle).

Loads water-bench result JSON (from eval_waterbench.py with watermark_type=aaronson),
applies deletions to each generated text, and computes the Aaronson watermark score
on the truncated sequence. When aaronson_wm_param_m was used at generation time,
detection tries all shifts s ∈ {0, ..., m-1} (Algorithm 2) so the watermark can
still be detected after prefix deletion.

Usage:
  python eval_prefix_deletion_aaronson.py --results_json path/to/2000_aaronson.json [options]
"""
import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from generate import calculate_aaronson_watermark_score, get_special_token_ids


def apply_deletion(tokens, k, deletion_type):
    """
    Apply deletion of k tokens. tokens: [1, L]. Returns [1, L'] with L' = L - k (or 0 if k >= L).
    """
    if k <= 0:
        return tokens
    L = tokens.shape[1]
    if k >= L:
        return tokens[:, :0]  # empty sequence
    if deletion_type == "prefix":
        return tokens[:, k:]
    if deletion_type == "suffix":
        return tokens[:, :-k]
    if deletion_type == "random":
        indices = list(range(L))
        drop = set(random.sample(indices, k))
        keep = [i for i in indices if i not in drop]
        return tokens[:, keep]
    if deletion_type == "middle":
        start = random.randint(0, L - k)
        return torch.cat([tokens[:, :start], tokens[:, start + k :]], dim=1)
    raise ValueError(f"Unknown deletion_type: {deletion_type}")


def load_results(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Aaronson score after random prefix deletions on water-bench results"
    )
    parser.add_argument(
        "--results_json",
        type=str,
        required=True,
        help="Path to water-bench result JSON (aaronson watermark)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path for tokenizer (default: use config from JSON)",
    )
    parser.add_argument(
        "--deletion_type",
        type=str,
        default="prefix",
        choices=["prefix", "suffix", "random", "middle"],
        help="Type of deletion: prefix (drop first k), suffix (drop last k), random (drop k random tokens), middle (drop k contiguous tokens from middle). Default: prefix",
    )
    parser.add_argument(
        "--max_prefix_delete",
        type=str,
        default="0.5",
        help="Max tokens to delete: int (absolute) or float in (0,1] (fraction of length). Used for all deletion_type. Default 0.5",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=3,
        help="Number of deletion trials per sample (default: 3)",
    )
    parser.add_argument(
        "--score_aggregation",
        type=str,
        default="mean",
        choices=["mean", "best"],
        help="Per-sample score: mean over trials (reflects deletion severity) or best over trials (default: mean)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for prefix deletion (default: 123)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Normalized score threshold for detection (default: 1.0)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of result samples (default: all)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save per-sample stats JSON",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=126464,
        help="Vocabulary size (default: 126464)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_results(args.results_json)
    if data.get("watermark_type") != "aaronson":
        raise ValueError(
            f"Expected watermark_type=aaronson in {args.results_json}, got {data.get('watermark_type')}"
        )

    config = data.get("config", {})
    model_path = args.model_path or config.get("model_path", "GSAI-ML/LLaDA-8B-Instruct")
    aaronson_seed = config.get("aaronson_seed", 42)
    aaronson_wm_param_m = config.get("aaronson_wm_param_m")

    # Parse max_prefix_delete: int or float
    try:
        max_delete = int(args.max_prefix_delete)
        use_fraction = False
    except ValueError:
        max_delete = float(args.max_prefix_delete)
        if not (0 < max_delete <= 1):
            raise ValueError("If float, max_prefix_delete must be in (0, 1]")
        use_fraction = True

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    special_token_ids = get_special_token_ids(tokenizer)

    results = data.get("results", [])
    if args.max_samples is not None:
        results = results[: args.max_samples]

    has_token_ids = sum(1 for r in results if r.get("generated_token_ids") is not None)
    if has_token_ids < len(results):
        print(
            "Warning: {} entries lack 'generated_token_ids'; re-tokenizing from text may change tokenization "
            "and lower scores. Re-run eval_waterbench.py to save token IDs for exact scoring.".format(
                len(results) - has_token_ids
            )
        )

    all_scores_after = []  # normalized score after prefix deletion (best over trials)
    all_detected = []
    per_sample = []

    for entry in tqdm(results, desc="Prefix deletion eval"):
        # Use exact token IDs from generation when available (avoids re-tokenization mismatch)
        token_ids = entry.get("generated_token_ids")
        if token_ids is not None:
            # Ensure list of ints so tensor is (1, L); tolerate single int or nested list
            if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            tokens = torch.tensor([token_ids], dtype=torch.long)
        else:
            generated_text = entry.get("generated_text", "")
            if not generated_text:
                per_sample.append(
                    {"prompt_id": entry.get("prompt_id"), "error": "no generated_text"}
                )
                continue
            tok = tokenizer(generated_text, return_tensors="pt", add_special_tokens=False)
            tokens = tok["input_ids"]
        # Normalize to [1, L]; some tokenizers or malformed data can give 1D or 0D
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() < 2:
            per_sample.append({"prompt_id": entry.get("prompt_id"), "error": "tokens not 2D"})
            continue
        L = tokens.shape[1]
        if L == 0:
            per_sample.append({"prompt_id": entry.get("prompt_id"), "error": "empty tokens"})
            continue

        best_norm_after = -float("inf")
        best_detected = False
        trial_scores = []

        for _ in range(args.num_trials):
            if use_fraction:
                max_k = min(L - 1, int(L * max_delete))
            else:
                max_k = min(max_delete, L - 1)
            k = random.randint(0, max_k) if max_k > 0 else 0
            truncated = apply_deletion(tokens, k, args.deletion_type)
            if truncated.shape[1] == 0:
                continue

            score, actual_length, per_token_scores, best_shift = calculate_aaronson_watermark_score(
                truncated,
                vocab_size=args.vocab_size,
                seed=aaronson_seed,
                special_token_ids=special_token_ids,
                position_offset=0,
                wm_param_m=aaronson_wm_param_m,
            )
            norm = score / actual_length if actual_length > 0 else 0.0
            trial_scores.append(norm)
            if norm > best_norm_after:
                best_norm_after = norm
                best_detected = norm > args.threshold

        # Aggregate per sample: mean or best over trials (best often = no-deletion trial when k=0 is possible)
        if args.score_aggregation == "mean" and trial_scores:
            agg_score = sum(trial_scores) / len(trial_scores)
            agg_detected = agg_score > args.threshold
        else:
            agg_score = best_norm_after if trial_scores else -float("inf")
            agg_detected = best_detected
        if not trial_scores:
            per_sample.append({"prompt_id": entry.get("prompt_id"), "error": "no valid trials"})
            continue
        all_scores_after.append(agg_score)
        all_detected.append(1 if agg_detected else 0)
        per_sample.append({
            "prompt_id": entry.get("prompt_id"),
            "original_length": L,
            "aggregation": args.score_aggregation,
            "normalized_score_after": agg_score,
            "detected": agg_detected,
            "trial_scores": trial_scores,
        })

    n = len(all_scores_after)
    if n == 0:
        print("No valid samples.")
        return

    mean_score = sum(all_scores_after) / n
    detection_rate = sum(all_detected) / n

    print("\n" + "=" * 60)
    print("DELETION ROBUSTNESS EVALUATION (Aaronson watermark)")
    print("=" * 60)
    print(f"Results file: {args.results_json}")
    print(f"deletion_type: {args.deletion_type}")
    print(f"aaronson_seed: {aaronson_seed}")
    print(f"aaronson_wm_param_m: {aaronson_wm_param_m or 'disabled'}")
    print(f"max_delete: {args.max_prefix_delete} ({'fraction' if use_fraction else 'tokens'})")
    print(f"num_trials per sample: {args.num_trials}")
    print(f"score_aggregation: {args.score_aggregation}")
    print(f"threshold (normalized): {args.threshold}")
    print(f"Samples: {n}")
    print(f"Mean normalized score (after {args.deletion_type} deletion): {mean_score:.4f}")
    print(f"Detection rate (score > threshold): {detection_rate:.2%}")
    print("=" * 60 + "\n")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "results_json": args.results_json,
            "deletion_type": args.deletion_type,
            "aaronson_seed": aaronson_seed,
            "aaronson_wm_param_m": aaronson_wm_param_m,
            "max_delete": args.max_prefix_delete,
            "num_trials": args.num_trials,
            "score_aggregation": args.score_aggregation,
            "threshold": args.threshold,
            "num_samples": n,
            "mean_normalized_score_after": mean_score,
            "detection_rate": detection_rate,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
