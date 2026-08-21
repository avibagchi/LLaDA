#!/usr/bin/env python3
"""
Final 1000-prompt evaluation for all baseline watermarking methods.

Loads LLaDA and GPT-2 ONCE, then runs all optimal configs sequentially.
Produces output files compatible with evaluate_with_gpt4.py.

Table I metrics (computed inline):
  Completeness = P(z >= 4) on watermarked outputs
  Soundness    = P(z <  4) on no-watermark outputs (per method/gamma)

Table II metrics:
  PPL (GPT-2) — computed inline
  Style / Consistency / Accuracy / Ethics / Avg — from GPT-4 judge (separate step)

Dataset: water-bench-sampled_2000_seed42.jsonl (first 1000 prompts)
  Same seed42 as Gumbel-max full-bench run (fair comparison).
  Different from ablation seed43 (100 prompts).

Usage:
    conda run -n llada python run_final_baselines_1000.py --device cuda:3
    conda run -n llada python run_final_baselines_1000.py --device cuda:3 --resume
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import json
import math
import argparse
import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from generate import (
    generate,
    calculate_cdmark_score,
    calculate_dmark_score,
    calculate_lrdwm_score,
    get_special_token_ids,
)

MAX_PROMPTS = 1000

# ---- Optimal configs from ablation hyperparameter selection -----------------
# Each entry: (method, gamma, delta, tend, beta_labels)
WATERMARKED_CONFIGS = [
    # CDMArk
    ("cdmark",  0.9,  2.0,  40,  "b=0.15"),
    ("cdmark",  0.5,  4.0,  20,  "b=0.10/0.05"),
    ("cdmark",  0.25, 4.0,  20,  "b=0.01"),
    # DMark
    ("dmark",   0.1,  4.0, 300,  "b=0.15/0.10"),
    ("dmark",   0.25, 4.0,  80,  "b=0.05"),
    ("dmark",   0.5,  8.0, 160,  "b=0.01"),
    # LR-DWM (all betas share one config)
    ("lrdwm",   0.9,  4.0, 300,  "all-b"),
]

# For soundness: unique (method, gamma) combos needed on no-watermark text
# CDMArk score doesn't use gamma, so only one CDMArk soundness pass needed.
SOUNDNESS_DETECTORS = [
    ("cdmark",  None),   # gamma unused by calculate_cdmark_score
    ("dmark",   0.1),
    ("dmark",   0.25),
    ("dmark",   0.5),
    ("lrdwm",   0.9),
]


def load_jsonl(path, max_prompts=None):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if max_prompts and len(data) >= max_prompts:
                    break
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
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def calc_perplexity(eval_model, eval_tokenizer, text):
    if not text or not text.strip():
        return None
    try:
        ids = eval_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
        if ids.shape[1] < 2:
            return None
        with torch.no_grad():
            out = eval_model(ids, labels=ids)
            logits = out.logits if hasattr(out, "logits") else out[1]
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                ids[:, 1:].contiguous().view(-1),
                reduction="mean",
            )
        return float(torch.exp(loss).item())
    except Exception:
        return None


def compute_z_score(method, gamma, tokens, seed, vocab_size, mask_id):
    """Dispatch z-score computation by method."""
    if method == "cdmark":
        return calculate_cdmark_score(
            tokens.unsqueeze(0), secret_key=seed, vocab_size=vocab_size,
            m=1, mask_id=mask_id,
        )
    elif method == "dmark":
        return calculate_dmark_score(
            tokens.unsqueeze(0), secret_key=seed, gamma=gamma,
            vocab_size=vocab_size, mask_id=mask_id,
        )
    elif method == "lrdwm":
        return calculate_lrdwm_score(
            tokens.unsqueeze(0), secret_key=seed, gamma=gamma,
            vocab_size=vocab_size, mask_id=mask_id,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def out_filename(method, gamma, delta, tend, n=1000):
    return f"{method}_final1000_gamma={gamma}_delta={delta}_tend={tend}_{n}.json"


def run_config(
    method, gamma, delta, tend, beta_labels,
    model, tokenizer, entries, prompt_tokens_list,
    eval_model, eval_tokenizer, special_token_ids,
    args, output_dir,
):
    out_path = output_dir / out_filename(method, gamma, delta, tend)
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return None

    seed = 42
    results = []
    for idx, (entry, prompt_tokens) in enumerate(zip(entries, prompt_tokens_list)):
        if prompt_tokens is None:
            continue
        prompt_tensor = torch.tensor([prompt_tokens]).to(args.device)

        gen_kwargs = dict(
            model=model,
            prompt=prompt_tensor,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            remasking="low_confidence",
            mask_id=args.mask_id,
            watermark_type=method,
            gamma=gamma,
            amplification=delta,
            vocab_size=args.vocab_size,
            special_token_ids=special_token_ids,
            watermark_steps=tend,
        )
        if method == "cdmark":
            gen_kwargs["cdmark_seed"] = seed
            gen_kwargs["cdmark_m"] = 1
        elif method == "dmark":
            gen_kwargs["dmark_seed"] = seed
        elif method == "lrdwm":
            gen_kwargs["lrdwm_seed"] = seed

        with torch.no_grad():
            generated = generate(**gen_kwargs)

        gen_tokens = generated[0, len(prompt_tokens):]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        perplexity = calc_perplexity(eval_model, eval_tokenizer, gen_text)
        z_score, valid_len = compute_z_score(method, gamma, gen_tokens, seed, args.vocab_size, args.mask_id)

        results.append({
            "prompt_id": idx + 1,
            "context": entry.get("context", ""),
            "input": entry.get("input", ""),
            "generated_text": gen_text,
            "expected_outputs": entry.get("outputs", []),
            "perplexity": perplexity,
            "watermark_type": method,
            "watermark_metrics": {"z_score": float(z_score), "length": int(valid_len)},
            "generation_length": len(gen_tokens.tolist()),
            "dataset": entry.get("dataset", ""),
            "_id": entry.get("_id", ""),
        })

        if (idx + 1) % 100 == 0:
            detected = sum(1 for r in results if r["watermark_metrics"]["z_score"] >= 4.0)
            print(f"    {idx+1}/{len(entries)}: detected={detected}/{len(results)}")

    ppls = [r["perplexity"] for r in results if r["perplexity"] is not None]
    detected = sum(1 for r in results if r["watermark_metrics"]["z_score"] >= 4.0)
    completeness = detected / len(results) if results else 0.0
    avg_ppl = sum(ppls) / len(ppls) if ppls else None

    output_data = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "watermark_type": method,
        "config": {
            "model_path": args.model_path,
            "gen_length": args.gen_length,
            "steps": args.steps,
            "temperature": args.temperature,
            "block_length": args.block_length,
            "seed": seed,
            "gamma": gamma,
            "delta": delta,
            "watermark_steps": tend,
            "beta_labels": beta_labels,
        },
        "total_prompts": len(results),
        "completeness": completeness,
        "average_perplexity": avg_ppl,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  completeness={completeness:.3f}  avg_ppl={avg_ppl:.1f if avg_ppl else 'N/A'}")
    return output_data


def run_no_watermark(
    model, tokenizer, entries, prompt_tokens_list,
    eval_model, eval_tokenizer, special_token_ids,
    args, output_dir,
):
    out_path = output_dir / f"nowatermark_final1000_{len(entries)}.json"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        with open(out_path) as f:
            return json.load(f)

    results = []
    for idx, (entry, prompt_tokens) in enumerate(zip(entries, prompt_tokens_list)):
        if prompt_tokens is None:
            continue
        prompt_tensor = torch.tensor([prompt_tokens]).to(args.device)

        with torch.no_grad():
            generated = generate(
                model=model,
                prompt=prompt_tensor,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                remasking="low_confidence",
                mask_id=args.mask_id,
                watermark_type="green_list",
                gamma=0.5,
                amplification=0.0,
                vocab_size=args.vocab_size,
                special_token_ids=special_token_ids,
            )

        gen_tokens = generated[0, len(prompt_tokens):]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        perplexity = calc_perplexity(eval_model, eval_tokenizer, gen_text)

        # Compute z-scores for all soundness detectors
        seed = 42
        z_scores = {}
        for det_method, det_gamma in SOUNDNESS_DETECTORS:
            key = f"{det_method}_g{det_gamma}" if det_gamma is not None else det_method
            try:
                z, _ = compute_z_score(det_method, det_gamma, gen_tokens, seed, args.vocab_size, args.mask_id)
                z_scores[key] = float(z)
            except Exception:
                z_scores[key] = None

        results.append({
            "prompt_id": idx + 1,
            "context": entry.get("context", ""),
            "input": entry.get("input", ""),
            "generated_text": gen_text,
            "expected_outputs": entry.get("outputs", []),
            "perplexity": perplexity,
            "watermark_type": "none",
            "z_scores_soundness": z_scores,
            "generation_length": len(gen_tokens.tolist()),
            "dataset": entry.get("dataset", ""),
            "_id": entry.get("_id", ""),
        })

        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{len(entries)}")

    ppls = [r["perplexity"] for r in results if r["perplexity"] is not None]
    avg_ppl = sum(ppls) / len(ppls) if ppls else None

    # Compute soundness per detector
    soundness = {}
    for det_method, det_gamma in SOUNDNESS_DETECTORS:
        key = f"{det_method}_g{det_gamma}" if det_gamma is not None else det_method
        zs = [r["z_scores_soundness"][key] for r in results if r["z_scores_soundness"].get(key) is not None]
        soundness[key] = sum(1 for z in zs if z < 4.0) / len(zs) if zs else None

    output_data = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "watermark_type": "none",
        "config": {
            "model_path": args.model_path,
            "gen_length": args.gen_length,
            "steps": args.steps,
            "temperature": args.temperature,
            "block_length": args.block_length,
        },
        "total_prompts": len(results),
        "average_perplexity": avg_ppl,
        "soundness_per_detector": soundness,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  avg_ppl={avg_ppl:.1f if avg_ppl else 'N/A'}")
    print("  Soundness (P(z<4) on clean text):")
    for k, v in soundness.items():
        print(f"    {k}: {v:.4f}" if v is not None else f"    {k}: N/A")
    return output_data


def print_summary(wm_results, nw_data, output_dir):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Table I metrics")
    print("=" * 70)
    print(f"{'Config':<40} {'Completeness':>12} {'Avg PPL':>8}")
    print("-" * 62)

    for method, gamma, delta, tend, beta_labels in WATERMARKED_CONFIGS:
        out_path = output_dir / out_filename(method, gamma, delta, tend)
        if not out_path.exists():
            print(f"  {method} g={gamma} d={delta} t={tend} ({beta_labels}): MISSING")
            continue
        with open(out_path) as f:
            d = json.load(f)
        label = f"{method} γ={gamma} δ={delta} t={tend}"
        comp = d.get("completeness", float("nan"))
        ppl = d.get("average_perplexity", float("nan"))
        ppl_str = f"{ppl:.1f}" if ppl and not math.isnan(ppl) else "N/A"
        print(f"  {label:<38} {comp:>12.3f} {ppl_str:>8}")

    if nw_data:
        print()
        print("Soundness (P(z<4) on no-watermark text):")
        for k, v in nw_data.get("soundness_per_detector", {}).items():
            vs = f"{v:.4f}" if v is not None else "N/A"
            print(f"  {k}: {vs}")

    print("\n" + "=" * 70)
    print("Next: run GPT-4 judge for Table II metrics")
    gpt4_files = [out_filename(m, g, d, t) for m, g, d, t, _ in WATERMARKED_CONFIGS]
    gpt4_files.append(f"nowatermark_final1000_1000.json")
    print("  conda run -n llada python run_gpt4_eval_1000.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, default="water-bench-sampled_2000_seed42.jsonl")
    parser.add_argument("--max_prompts", type=int, default=MAX_PROMPTS)
    parser.add_argument("--output_dir", type=str, default="water-bench-results/json-outputs")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--gen_length", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--block_length", type=int, default=25)
    parser.add_argument("--mask_id", type=int, default=126336)
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--resume", action="store_true", help="Skip existing output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LLaDA from {args.model_path} on {args.device}...")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    special_token_ids = get_special_token_ids(tokenizer)

    print("Loading GPT-2 for perplexity (CPU)...")
    eval_model = AutoModelForCausalLM.from_pretrained("gpt2").cpu().eval()
    eval_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

    print(f"Loading up to {args.max_prompts} prompts from {args.jsonl_file}...")
    entries = load_jsonl(args.jsonl_file, max_prompts=args.max_prompts)
    prompt_tokens_list = []
    for entry in entries:
        text = format_prompt(entry.get("context", ""), entry.get("input", ""), tokenizer)
        prompt_tokens_list.append(tokenizer(text)["input_ids"] if text else None)
    print(f"  {len(entries)} prompts loaded\n")

    total_configs = len(WATERMARKED_CONFIGS) + 1  # +1 for no-watermark
    print(f"Running {total_configs} configs ({len(WATERMARKED_CONFIGS)} watermarked + 1 no-watermark)")
    print(f"Output dir: {output_dir}\n")

    wm_results = []
    for config_idx, (method, gamma, delta, tend, beta_labels) in enumerate(WATERMARKED_CONFIGS, 1):
        print(f"[{config_idx}/{total_configs}] {method} γ={gamma} δ={delta} tend={tend} ({beta_labels})")
        result = run_config(
            method, gamma, delta, tend, beta_labels,
            model, tokenizer, entries, prompt_tokens_list,
            eval_model, eval_tokenizer, special_token_ids,
            args, output_dir,
        )
        wm_results.append(result)

    print(f"[{total_configs}/{total_configs}] No-watermark baseline")
    nw_data = run_no_watermark(
        model, tokenizer, entries, prompt_tokens_list,
        eval_model, eval_tokenizer, special_token_ids,
        args, output_dir,
    )

    print_summary(wm_results, nw_data, output_dir)


if __name__ == "__main__":
    main()
