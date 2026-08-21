#!/usr/bin/env python3
"""
CDMArk ablation sweep — loads LLaDA and GPT-2 ONCE, then iterates over all
(gamma, delta, tend) combinations without reloading.

CDMArk uses signal vectors V ∈ R^{vocab_size × m}; for zero-bit (m=1) the
logit bias is delta * V[w,0] and detection is z = sum(V[x_i,0]) / sqrt(N).
gamma controls the detection threshold indirectly (not used in generation for
CDMArk, but kept in the grid for consistent ablation structure; it is passed
through as a config field only).

Usage:
    python run_cdmark_ablation_efficient.py --device cuda:3
    python run_cdmark_ablation_efficient.py --device cuda:3 --resume
    python run_cdmark_ablation_efficient.py --device cuda:3 --cdmark_m 1
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import json
import math
import argparse
import datetime
import sys
from pathlib import Path
from itertools import product
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from generate import generate, calculate_cdmark_score, get_special_token_ids

GAMMA_VALUES = [0.1, 0.25, 0.5, 0.75, 0.9]
DELTA_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]
TEND_VALUES  = [5, 10, 20, 40, 80, 160, 300]


def load_jsonl(path):
    data = []
    with open(path, encoding='utf-8') as f:
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
            logits = out.logits if hasattr(out, 'logits') else out[1]
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                ids[:, 1:].contiguous().view(-1),
                reduction='mean'
            )
        return float(torch.exp(loss).item())
    except Exception:
        return None


def run_combination(model, tokenizer, entries, prompt_tokens_list,
                    eval_model, eval_tokenizer, special_token_ids,
                    gamma, delta, tend, cdmark_seed, cdmark_m,
                    gen_length, steps, temperature, block_length,
                    mask_id, vocab_size, device):
    results = []
    for idx, (entry, prompt_tokens) in enumerate(zip(entries, prompt_tokens_list)):
        if prompt_tokens is None:
            continue
        prompt_tensor = torch.tensor([prompt_tokens]).to(device)
        with torch.no_grad():
            generated = generate(
                model=model,
                prompt=prompt_tensor,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                remasking='low_confidence',
                mask_id=mask_id,
                watermark_type='cdmark',
                amplification=delta,
                vocab_size=vocab_size,
                special_token_ids=special_token_ids,
                cdmark_seed=cdmark_seed,
                cdmark_m=cdmark_m,
                watermark_steps=tend,
            )
        gen_tokens = generated[0, len(prompt_tokens):]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        perplexity = calc_perplexity(eval_model, eval_tokenizer, gen_text)
        z_score, valid_len = calculate_cdmark_score(
            gen_tokens.unsqueeze(0),
            secret_key=cdmark_seed,
            vocab_size=vocab_size,
            m=cdmark_m,
            mask_id=mask_id,
        )
        results.append({
            "prompt_id": idx + 1,
            "context": entry.get('context', ''),
            "input": entry.get('input', ''),
            "generated_text": gen_text,
            "expected_outputs": entry.get('outputs', []),
            "perplexity": perplexity,
            "watermark_type": "cdmark",
            "watermark_metrics": {"z_score": float(z_score), "length": int(valid_len)},
            "generation_length": len(gen_tokens.tolist()),
            "dataset": entry.get('dataset', ''),
            "_id": entry.get('_id', ''),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_file', type=str, default='water-bench-sampled_100_seed43.jsonl')
    parser.add_argument('--output_dir', type=str, default='water-bench-results/json-outputs')
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--cdmark_seed', type=int, default=42)
    parser.add_argument('--cdmark_m', type=int, default=1)
    parser.add_argument('--gen_length', type=int, default=300)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--block_length', type=int, default=25)
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--vocab_size', type=int, default=126464)
    parser.add_argument('--resume', action='store_true')
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

    print(f"Loading prompts from {args.jsonl_file}...")
    entries = load_jsonl(args.jsonl_file)
    prompt_tokens_list = []
    for entry in entries:
        text = format_prompt(entry.get('context', ''), entry.get('input', ''), tokenizer)
        prompt_tokens_list.append(tokenizer(text)["input_ids"] if text else None)
    print(f"  {len(entries)} prompts loaded\n")

    combos = list(product(GAMMA_VALUES, DELTA_VALUES, TEND_VALUES))
    if args.resume:
        combos = [
            (g, d, t) for g, d, t in combos
            if not (output_dir / f"cdmark_m{args.cdmark_m}_gamma={g}_delta={d}_tend={t}_sampled_100.json").exists()
        ]
        print(f"Resume mode: {len(combos)} combinations remaining")
    else:
        print(f"Running {len(combos)} combinations (m={args.cdmark_m})")

    total = len(combos)
    for combo_idx, (gamma, delta, tend) in enumerate(combos):
        out_name = f"cdmark_m{args.cdmark_m}_gamma={gamma}_delta={delta}_tend={tend}_sampled_100.json"
        out_path = output_dir / out_name
        print(f"\n[{combo_idx+1}/{total}] γ={gamma} δ={delta} t_end={tend} → {out_name}")

        results = run_combination(
            model, tokenizer, entries, prompt_tokens_list,
            eval_model, eval_tokenizer, special_token_ids,
            gamma=gamma, delta=delta, tend=tend,
            cdmark_seed=args.cdmark_seed, cdmark_m=args.cdmark_m,
            gen_length=args.gen_length, steps=args.steps,
            temperature=args.temperature, block_length=args.block_length,
            mask_id=args.mask_id, vocab_size=args.vocab_size,
            device=args.device,
        )

        perplexities = [r['perplexity'] for r in results if r['perplexity'] is not None]
        detected = sum(1 for r in results if r['watermark_metrics']['z_score'] >= 4.0)
        print(f"  detected={detected}/{len(results)}  avg_ppl={sum(perplexities)/len(perplexities):.1f}")

        output_data = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "watermark_type": "cdmark",
            "config": {
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "steps": args.steps,
                "temperature": args.temperature,
                "block_length": args.block_length,
                "cdmark_seed": args.cdmark_seed,
                "cdmark_m": args.cdmark_m,
                "cdmark_gamma": gamma,
                "cdmark_delta": delta,
                "cdmark_watermark_steps": tend,
            },
            "total_prompts": len(results),
            "average_perplexity": sum(perplexities)/len(perplexities) if perplexities else None,
            "results": results,
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results in {output_dir}/")
    print(f"  python find_optimal_red_green_hyperparams.py {output_dir}/")


if __name__ == '__main__':
    main()
