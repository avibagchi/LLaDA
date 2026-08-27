#!/usr/bin/env python3
"""
Gumbel-max (Aaronson) t_end ablation sweep — loads LLaDA and GPT-2 ONCE, then
iterates over all watermarking intervals S_W = {t_start=1, t_end} without
reloading.

Unlike the green-list baselines (Kirchenbauer/Gloaguen/DMark/etc.), the
Gumbel-max scheme has no gamma/delta — it is unbiased at every step
(Theorem 1), so the only thing being ablated here is which prefix of
diffusion steps [1, t_end] gets watermarked. Per the paper (Fig. 3), judge
score should be roughly flat across t_end since there's no bias-accumulation
tradeoff; this script lets you verify completeness/quality at each t_end
directly.

Detection follows Algorithm 2 / eval_waterbench.py convention:
    normalized_score = aaronson_score / actual_length
    detected         = normalized_score > tau  (tau* = 1.19 from the paper)

Usage:
    python run_gumbel_tend_ablation_efficient.py --device cuda:3
    python run_gumbel_tend_ablation_efficient.py --device cuda:3 --resume
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import json
import argparse
import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from generate import generate, calculate_aaronson_watermark_score, get_special_token_ids

TEND_VALUES = [5, 10, 20, 40, 80, 160, 300]
TAU = 1.19


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
                    tend, aaronson_seed,
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
                watermark_type='aaronson',
                vocab_size=vocab_size,
                special_token_ids=special_token_ids,
                aaronson_seed=aaronson_seed,
                watermark_steps=tend,
            )
        gen_tokens = generated[0, len(prompt_tokens):]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        perplexity = calc_perplexity(eval_model, eval_tokenizer, gen_text)

        score, actual_length, _ = calculate_aaronson_watermark_score(
            gen_tokens.unsqueeze(0),
            vocab_size=vocab_size,
            seed=aaronson_seed,
            special_token_ids=special_token_ids,
            position_offset=len(prompt_tokens),
        )
        normalized_score = float(score / actual_length) if actual_length > 0 else 0.0

        results.append({
            "prompt_id": idx + 1,
            "context": entry.get('context', ''),
            "input": entry.get('input', ''),
            "generated_text": gen_text,
            "expected_outputs": entry.get('outputs', []),
            "perplexity": perplexity,
            "watermark_type": "aaronson",
            "watermark_metrics": {
                "normalized_score": normalized_score,
                "length": int(actual_length),
            },
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
    parser.add_argument('--aaronson_seed', type=int, default=42)
    parser.add_argument('--gen_length', type=int, default=300)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--block_length', type=int, default=25)
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--vocab_size', type=int, default=126464)
    parser.add_argument('--tau', type=float, default=TAU)
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

    tend_values = TEND_VALUES
    if args.resume:
        tend_values = [
            t for t in TEND_VALUES
            if not (output_dir / f"aaronson_tend={t}_sampled_100.json").exists()
        ]
        print(f"Resume mode: {len(tend_values)} configs remaining")
    else:
        print(f"Running {len(tend_values)} t_end configs: {TEND_VALUES}")

    for combo_idx, tend in enumerate(tend_values):
        out_name = f"aaronson_tend={tend}_sampled_100.json"
        out_path = output_dir / out_name
        print(f"\n[{combo_idx+1}/{len(tend_values)}] t_end={tend} (S_W=[1,{tend}]) -> {out_name}")

        results = run_combination(
            model, tokenizer, entries, prompt_tokens_list,
            eval_model, eval_tokenizer, special_token_ids,
            tend=tend, aaronson_seed=args.aaronson_seed,
            gen_length=args.gen_length, steps=args.steps,
            temperature=args.temperature, block_length=args.block_length,
            mask_id=args.mask_id, vocab_size=args.vocab_size,
            device=args.device,
        )

        perplexities = [r['perplexity'] for r in results if r['perplexity'] is not None]
        detected = sum(1 for r in results if r['watermark_metrics']['normalized_score'] > args.tau)
        completeness = detected / len(results) if results else None
        avg_ppl = sum(perplexities) / len(perplexities) if perplexities else None
        print(f"  completeness={completeness:.3f}  avg_ppl={avg_ppl:.1f}" if completeness is not None and avg_ppl else
              f"  detected={detected}/{len(results)}")

        output_data = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "watermark_type": "aaronson",
            "config": {
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "steps": args.steps,
                "temperature": args.temperature,
                "block_length": args.block_length,
                "aaronson_seed": args.aaronson_seed,
                "watermark_steps": tend,
                "tau": args.tau,
            },
            "total_prompts": len(results),
            "completeness": completeness,
            "average_perplexity": avg_ppl,
            "results": results,
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results in {output_dir}/")
    print("Next: run GPT-4 judge on each aaronson_tend=*_sampled_100.json to get quality scores")


if __name__ == '__main__':
    main()
