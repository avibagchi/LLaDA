#!/usr/bin/env python3
"""
Apply prefix deletion to generated text in a water-bench result JSON,
recalculate the normalized Aaronson score, and write a new JSON.

Usage:
  python apply_prefix_deletion_regenerate_json.py \
    --input_json water-bench-results/json-outputs/robust_test_1.json \
    --output_json water-bench-results/json-outputs/robust_test_1_prefix_deleted.json \
    [--prefix_delete 0.5]
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from generate import calculate_aaronson_watermark_score, get_special_token_ids


def get_tokens_from_entry(entry, tokenizer):
    """Get token tensor [1, L] from entry. Use generated_token_ids if valid list, else tokenize."""
    token_ids = entry.get("generated_token_ids")
    if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], (int, float)):
        tokens = torch.tensor([token_ids], dtype=torch.long)
    else:
        generated_text = entry.get("generated_text", "")
        if not generated_text:
            return None
        tok = tokenizer(generated_text, return_tensors="pt", add_special_tokens=False)
        tokens = tok["input_ids"]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.dim() < 2 or tokens.shape[1] == 0:
        return None
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Apply prefix deletion to water-bench results and regenerate JSON with recalculated scores"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="water-bench-results/json-outputs/robust_test_1.json",
        help="Path to input water-bench result JSON",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="water-bench-results/json-outputs/robust_test_1_prefix_deleted.json",
        help="Path to output JSON with prefix-deleted text and new scores",
    )
    parser.add_argument(
        "--prefix_delete",
        type=str,
        default="0.5",
        help="Tokens to delete from prefix: int (absolute) or float in (0,1] (fraction of length). Default: 0.5",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path for tokenizer (default: use config from JSON)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=126464,
        help="Vocabulary size (default: 126464)",
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("watermark_type") != "aaronson":
        raise ValueError(
            f"Expected watermark_type=aaronson in {args.input_json}, got {data.get('watermark_type')}"
        )

    config = data.get("config", {})
    model_path = args.model_path or config.get("model_path", "GSAI-ML/LLaDA-8B-Instruct")
    aaronson_seed = config.get("aaronson_seed", 42)
    aaronson_wm_param_m = config.get("aaronson_wm_param_m")

    try:
        prefix_delete = int(args.prefix_delete)
        use_fraction = False
    except ValueError:
        prefix_delete = float(args.prefix_delete)
        if not (0 < prefix_delete <= 1):
            raise ValueError("If float, prefix_delete must be in (0, 1]")
        use_fraction = True

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    special_token_ids = get_special_token_ids(tokenizer)

    # Check if we have valid token IDs (list of ints); otherwise we'll re-tokenize (scores may be lower)
    n_valid_token_ids = sum(
        1 for r in data.get("results", [])
        if isinstance(r.get("generated_token_ids"), list)
        and len(r.get("generated_token_ids", [])) > 0
        and isinstance(r["generated_token_ids"][0], (int, float))
    )
    n_total = len(data.get("results", []))
    if n_valid_token_ids < n_total:
        print(
            f"Warning: {n_total - n_valid_token_ids}/{n_total} entries have invalid 'generated_token_ids' "
            "(expected list of ints). Re-tokenizing from text may change token IDs and lower scores. "
            "Re-run eval_waterbench.py to produce JSON with proper token IDs."
        )

    new_results = []
    for entry in tqdm(data.get("results", []), desc="Applying prefix deletion"):
        tokens = get_tokens_from_entry(entry, tokenizer)
        if tokens is None:
            new_results.append({**entry, "error": "could not get tokens", "prefix_deleted": None})
            continue

        L = tokens.shape[1]
        if use_fraction:
            k = min(L - 1, int(L * prefix_delete))
        else:
            k = min(prefix_delete, L - 1)
        if k <= 0:
            truncated = tokens
            k = 0
        else:
            truncated = tokens[:, k:]

        if truncated.shape[1] == 0:
            new_results.append({**entry, "error": "empty after deletion", "prefix_deleted": k})
            continue

        score, actual_length, per_token_scores, best_shift = calculate_aaronson_watermark_score(
            truncated,
            vocab_size=args.vocab_size,
            seed=aaronson_seed,
            special_token_ids=special_token_ids,
            position_offset=0,
            wm_param_m=aaronson_wm_param_m,
        )
        normalized_score = score / actual_length if actual_length > 0 else 0.0

        truncated_ids = truncated[0].tolist()
        new_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)

        new_entry = {
            **{k: v for k, v in entry.items() if k not in ("generated_text", "generated_token_ids", "watermark_metrics", "generation_length")},
            "generated_text": new_text,
            "generated_token_ids": truncated_ids,
            "watermark_metrics": {
                "aaronson_score": float(score),
                "normalized_score": float(normalized_score),
                "length": int(actual_length),
                **({"best_shift": int(best_shift)} if best_shift is not None else {}),
            },
            "generation_length": int(actual_length),
            "prefix_deleted": k,
            "original_length": L,
        }
        new_results.append(new_entry)

    output_data = {
        **{k: v for k, v in data.items() if k != "results"},
        "results": new_results,
        "prefix_deletion_applied": True,
        "prefix_delete_config": args.prefix_delete,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for r in new_results if "error" not in r)
    print(f"\nWrote {out_path}")
    print(f"Processed {len(new_results)} results ({n_ok} successful)")


if __name__ == "__main__":
    main()
