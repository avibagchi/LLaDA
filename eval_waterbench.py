#!/usr/bin/env python3
"""
Evaluate LLaDA model on WaterBench dataset with watermarking support.
Generates JSON results with prompt, context, watermark metrics, perplexity, and expected outputs.
"""
import sys
import torch
import argparse
import json
import datetime
from pathlib import Path

_DLM_WM_SRC = Path(__file__).resolve().parent.parent / "diffusion-lm-watermark" / "src"
if _DLM_WM_SRC.is_dir() and str(_DLM_WM_SRC) not in sys.path:
    sys.path.insert(0, str(_DLM_WM_SRC))
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from generate import generate, calculate_aaronson_watermark_score, calculate_green_matches, get_special_token_ids
import math


def calculate_perplexity_gpt2(eval_model, eval_tokenizer, generated_text):
    """
    Calculate perplexity of generated text using GPT-2 on CPU.
    Runs on CPU to avoid CUDA vocabulary mismatch issues.
    """
    try:
        # Clean text to handle potential encoding issues
        if not generated_text or len(generated_text.strip()) == 0:
            return None
        
        # Convert text to GPT-2 tokens (on CPU)
        gpt2_tokens = eval_tokenizer(generated_text, return_tensors="pt", truncation=True, max_length=1024).input_ids
        
        if gpt2_tokens.shape[1] < 2:
            return None
        
        with torch.no_grad():
            # Get model output (on CPU)
            outputs = eval_model(gpt2_tokens, labels=gpt2_tokens)
            
            # Extract logits (shape: [batch, seq_len, vocab_size])
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = gpt2_tokens[:, 1:].contiguous()
            
            # Calculate perplexity
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                reduction="mean"
            )
            perplexity = torch.exp(loss)
            
        return float(perplexity.item())
    except Exception as e:
        print(f"Warning: Could not calculate perplexity: {e}")
        return None


def load_waterbench_jsonl(jsonl_path):
    """Load WaterBench dataset from JSONL file."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_prompt(context, input_text, tokenizer, use_chat_template=True):
    """Format prompt from context and input fields using chat template (like diffusion-lm-watermark)."""
    # Create user message content (matching diffusion-lm-watermark format)
    if context and input_text:
        user_content = f"You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n\n{context}\n{input_text}"
    elif context:
        user_content = f"You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n\n{context}"
    elif input_text:
        user_content = f"You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n\n{input_text}"
    else:
        return None
    
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        # Use chat template formatting (like diffusion-lm-watermark)
        messages = [{"role": "user", "content": user_content}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    else:
        # Fallback to plain text
        return user_content


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLaDA on WaterBench with watermarking')
    parser.add_argument('--jsonl_file', type=str, required=True, help='Path to WaterBench JSONL file')
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file (auto-generated if not specified)')
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct', help='Model path')
    parser.add_argument('--gen_length', type=int, default=300, help='Number of tokens to generate per prompt')
    parser.add_argument('--steps', type=int, default=300, help='Number of sampling steps')
    parser.add_argument('--block_length', type=int, default=25, help='Block length for generation')
    parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature')
    parser.add_argument('--remasking', type=str, default='random', choices=['random', 'low_confidence'],
                       help='Remasking strategy (random or low_confidence)')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale')
    parser.add_argument('--mask_id', type=int, default=126336, help='Mask token ID')
    parser.add_argument('--vocab_size', type=int, default=126464, help='Vocabulary size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max_prompts', type=int, default=None, help='Limit number of prompts to process')
    
    # Watermarking parameters
    parser.add_argument('--watermark_type', type=str, default='aaronson', 
                       choices=['aaronson', 'green_list', 'gloaguen', 'none'],
                       help='Watermark type (aaronson, green_list, gloaguen, or none)')
    parser.add_argument('--watermark_steps', type=int, default=None, 
                       help='Watermark up to this step (None = all steps)')
    parser.add_argument('--gamma', type=float, default=0.5, 
                       help='Gamma for green_list watermarking (fraction of green tokens)')
    parser.add_argument('--amplification', type=float, default=2.0, 
                       help='Amplification factor for green_list watermarking')
    parser.add_argument('--aaronson_seed', type=int, default=42, 
                       help='Seed for Aaronson watermarking')
    # Gloaguen et al. (OurWatermark / DiffusionKGW optimal Gaussian) — see diffusion-lm-watermark
    parser.add_argument('--gloaguen_delta', type=float, default=2.0, help='KL budget / booster strength')
    parser.add_argument('--gloaguen_enforce_kl', action=argparse.BooleanOptionalAction, default=True,
                       help='Enforce KL constraint in logit booster (default: true)')
    parser.add_argument('--gloaguen_conv_kernel', type=str, default='-1',
                       help='Comma-separated context offsets, e.g. -1 or -2,-1')
    parser.add_argument('--gloaguen_seeding_scheme', type=str, default='sumhash',
                       choices=['sumhash', 'minhash'], help='Hash of diffusion context')
    parser.add_argument('--gloaguen_greenlist_type', type=str, default='bernoulli',
                       choices=['bernoulli', 'gaussian', 'lognormal'], help='Greenlist randomization')
    parser.add_argument('--gloaguen_gamma', type=float, default=0.25,
                       help='Bernoulli greenlist rate (for bernoulli type)')
    parser.add_argument('--gloaguen_topk', type=int, default=100, help='Top-k for energy approximation')
    parser.add_argument('--gloaguen_n_iter', type=int, default=1, help='Fixed-point iterations')
    
    args = parser.parse_args()
    args.gloaguen_conv_kernel = [int(x.strip()) for x in args.gloaguen_conv_kernel.split(',') if x.strip()]
    if not args.gloaguen_conv_kernel:
        args.gloaguen_conv_kernel = [-1]
    
    # Load WaterBench data
    print(f"Loading WaterBench data from {args.jsonl_file}...")
    waterbench_data = load_waterbench_jsonl(args.jsonl_file)
    
    if args.max_prompts:
        waterbench_data = waterbench_data[:args.max_prompts]
    
    print(f"Processing {len(waterbench_data)} prompts")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model = AutoModel.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    special_token_ids = get_special_token_ids(tokenizer)

    gloaguen_wm = None
    if args.watermark_type == 'gloaguen':
        from dlm_watermark.watermarks.diffusion_watermark import OurWatermark
        gparams = {"gamma": args.gloaguen_gamma} if args.gloaguen_greenlist_type == "bernoulli" else {}
        gloaguen_wm = OurWatermark(
            delta=args.gloaguen_delta,
            enforce_kl=args.gloaguen_enforce_kl,
            convolution_kernel=args.gloaguen_conv_kernel,
            greenlist_type=args.gloaguen_greenlist_type,
            greenlist_params=gparams,
            topk=args.gloaguen_topk,
            n_iter=args.gloaguen_n_iter,
            seeding_scheme=args.gloaguen_seeding_scheme,
            tokenizer=tokenizer,
            device=str(model.device),
        )
    
    # Load GPT-2 for perplexity evaluation (external judge)
    print("Loading GPT-2 for perplexity calculation (on CPU)...")
    eval_model = AutoModelForCausalLM.from_pretrained("gpt2").cpu().eval()
    eval_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eval_tokenizer.pad_token = eval_tokenizer.eos_token
    
    # Print configuration
    print(f"\n{'='*60}")
    print("EVALUATION CONFIGURATION:")
    print(f"{'='*60}")
    print(f"WaterBench file: {args.jsonl_file}")
    print(f"Model: {args.model_path}")
    print(f"Generation length: {args.gen_length} tokens")
    print(f"Steps: {args.steps}")
    print(f"Temperature: {args.temperature}")
    print(f"Remasking: {args.remasking}")
    print(f"Block length: {args.block_length}")
    print(f"Watermark type: {args.watermark_type}")
    if args.watermark_type == 'aaronson':
        print(f"  Watermark steps: {args.watermark_steps if args.watermark_steps else 'all'}")
        print(f"  Aaronson seed: {args.aaronson_seed}")
        print(f"  Remasking strategy: original")
    elif args.watermark_type == 'green_list':
        print(f"  Gamma: {args.gamma}")
        print(f"  Amplification: {args.amplification}")
    elif args.watermark_type == 'gloaguen':
        print(f"  delta={args.gloaguen_delta}, enforce_kl={args.gloaguen_enforce_kl}")
        print(f"  conv_kernel={args.gloaguen_conv_kernel}, seeding={args.gloaguen_seeding_scheme}")
        print(f"  greenlist_type={args.gloaguen_greenlist_type}, gamma={args.gloaguen_gamma}")
        print(f"  topk={args.gloaguen_topk}, n_iter={args.gloaguen_n_iter}")
        ws = args.watermark_steps if args.watermark_steps is not None else 'all'
        print(f"  watermark_steps={ws}")
    print(f"{'='*60}\n")
    
    # Process each prompt
    results = []
    
    for idx, entry in enumerate(tqdm(waterbench_data, desc="Generating")):
        # Format prompt from context and input using chat template
        prompt_text = format_prompt(
            entry.get('context', ''), 
            entry.get('input', ''),
            tokenizer=tokenizer,
            use_chat_template=True
        )
        
        if not prompt_text:
            print(f"Warning: Skipping entry {idx} - no context or input")
            continue
        
        # Tokenize prompt
        prompt_tokens = tokenizer(prompt_text)["input_ids"]
        prompt_tensor = torch.tensor([prompt_tokens]).to(args.device)
        
        # Determine watermark parameters
        if args.watermark_type == 'none':
            watermark_type_gen = 'green_list'
            amplification_gen = 0.0
        elif args.watermark_type == 'gloaguen':
            watermark_type_gen = 'gloaguen'
            amplification_gen = None
        else:
            watermark_type_gen = args.watermark_type
            amplification_gen = args.amplification if args.watermark_type == 'green_list' else None
        
        # Generate
        with torch.no_grad():
            generated = generate(
                model=model,
                prompt=prompt_tensor,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                mask_id=args.mask_id,
                watermark_type=watermark_type_gen,
                gamma=args.gamma,
                amplification=amplification_gen,
                aaronson_seed=args.aaronson_seed,
                watermark_steps=args.watermark_steps,
                vocab_size=args.vocab_size,
                special_token_ids=special_token_ids,
                aaronson_remasking_strategy='original',
                gloaguen_watermark=gloaguen_wm,
            )
        
        # Extract generated tokens
        generated_tokens = generated[0, len(prompt_tokens):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate perplexity using GPT-2
        perplexity = calculate_perplexity_gpt2(eval_model, eval_tokenizer, generated_text)
        
        # Calculate watermark metrics
        watermark_metrics = {}
        
        if args.watermark_type == 'aaronson':
            score, actual_length, per_token_scores = calculate_aaronson_watermark_score(
                generated_tokens.unsqueeze(0),
                vocab_size=args.vocab_size,
                seed=args.aaronson_seed,
                special_token_ids=special_token_ids,
                position_offset=len(prompt_tokens)
            )
            watermark_metrics = {
                "aaronson_score": float(score),
                "normalized_score": float(score / actual_length) if actual_length > 0 else 0.0,
                "length": int(actual_length)
            }
        elif args.watermark_type == 'green_list':
            max_match_percent, actual_length, max_num_matches, best_start, match_arr = calculate_green_matches(
                generated_tokens.unsqueeze(0), 
                gamma=args.gamma,
                vocab_size=args.vocab_size
            )
            # Calculate Z-score for green list
            true_num_green = args.gamma * actual_length
            if math.sqrt(true_num_green * (1 - args.gamma)) == 0:
                z_score = 0
            else:
                z_score = (max_num_matches - true_num_green) / math.sqrt(true_num_green * (1 - args.gamma))
            
            watermark_metrics = {
                "green_token_matches": f"{max_num_matches}/{actual_length} ({max_match_percent:.2%})",
                "z_score": float(z_score),
                "match_percentage": float(max_match_percent),
                "length": int(actual_length)
            }
        elif args.watermark_type == 'gloaguen' and gloaguen_wm is not None:
            det = gloaguen_wm.detect(generated[0])
            if det:
                watermark_metrics = {
                    "z_score": det.get("z_score"),
                    "binomial_z_score": det.get("binomial_z_score"),
                    "p_value": det.get("p_value"),
                    "scheme": "DiffusionKGW_Optimal_Gaussian",
                }
            else:
                watermark_metrics = {
                    "z_score": None,
                    "binomial_z_score": None,
                    "p_value": None,
                    "note": "detect skipped (sequence too short for context window)",
                }
        elif args.watermark_type == 'none':
            # Calculate Aaronson score even when no watermark was applied
            score, actual_length, per_token_scores = calculate_aaronson_watermark_score(
                generated_tokens.unsqueeze(0),
                vocab_size=args.vocab_size,
                seed=args.aaronson_seed,
                special_token_ids=special_token_ids,
                position_offset=len(prompt_tokens)
            )
            watermark_metrics = {
                "aaronson_score": float(score),
                "normalized_score": float(score / actual_length) if actual_length > 0 else 0.0,
                "length": int(actual_length)
            }
        
        # Store result
        result = {
            "prompt_id": idx + 1,
            "context": entry.get('context', ''),
            "input": entry.get('input', ''),
            "prompt": prompt_text,
            "generated_text": generated_text,
            "expected_outputs": entry.get('outputs', []),
            "perplexity": perplexity,
            "watermark_type": args.watermark_type,
            "watermark_metrics": watermark_metrics,
            "generation_length": len(generated_tokens.tolist()),
            "dataset": entry.get('dataset', ''),
            "_id": entry.get('_id', '')
        }
        results.append(result)
    
    # Create output directory if it doesn't exist
    output_dir = Path("water-bench-results/json-outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename if not specified
    if args.output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_basename = Path(args.jsonl_file).stem
        args.output_file = f"waterbench_{jsonl_basename}_{args.watermark_type}_{timestamp}.json"
    
    # Save results to water-bench-results/json-outputs directory
    output_path = output_dir / Path(args.output_file).name
    print(f"\nSaving results to {output_path}...")
    
    # Calculate statistics
    perplexities = [r['perplexity'] for r in results if r['perplexity'] is not None]
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else None
    
    output_data = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "waterbench_file": args.jsonl_file,
        "watermark_type": args.watermark_type,
        "config": {
            "model_path": args.model_path,
            "gen_length": args.gen_length,
            "steps": args.steps,
            "temperature": args.temperature,
            "remasking": args.remasking,
            "block_length": args.block_length,
            "cfg_scale": args.cfg_scale,
            "watermark_steps": args.watermark_steps,
            "gamma": args.gamma if args.watermark_type == 'green_list' else None,
            "amplification": args.amplification if args.watermark_type == 'green_list' else None,
            "aaronson_seed": args.aaronson_seed if args.watermark_type in ['aaronson', 'none'] else None,
            "gloaguen": (
                gloaguen_wm.get_key_params()
                if args.watermark_type == 'gloaguen' and gloaguen_wm is not None
                else None
            ),
        },
        "total_prompts": len(results),
        "average_perplexity": avg_perplexity,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"Processed {len(results)} prompts")
    if avg_perplexity is not None:
        print(f"Average Perplexity (GPT-2): {avg_perplexity:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
