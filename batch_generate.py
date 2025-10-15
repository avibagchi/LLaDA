#!/usr/bin/env python3
"""
Batch generation script to process multiple prompts from a file with LLaDA.
"""
import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from generate import generate, calculate_aaronson_watermark_score, get_special_token_ids


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


def main():
    parser = argparse.ArgumentParser(description='Batch generate text with LLaDA')
    parser.add_argument('--prompts_file', type=str, required=True, help='File containing prompts (one per line)')
    parser.add_argument('--output_file', type=str, default='generated_outputs.json', help='Output JSON file')
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Base', help='Model path')
    parser.add_argument('--gen_length', type=int, default=512, help='Number of tokens to generate per prompt')
    parser.add_argument('--steps', type=int, default=512, help='Number of sampling steps')
    parser.add_argument('--block_length', type=int, default=512, help='Block length for generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale')
    parser.add_argument('--mask_id', type=int, default=126336, help='Mask token ID')
    parser.add_argument('--vocab_size', type=int, default=126464, help='Vocabulary size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max_prompts', type=int, default=None, help='Limit number of prompts to process')
    
    # Watermarking parameters
    parser.add_argument('--watermark_type', type=str, default='aaronson', choices=['aaronson', 'none'],
                       help='Watermark type (aaronson or none)')
    parser.add_argument('--watermark_steps', type=int, default=None, help='Watermark up to this step (None = all steps)')
    parser.add_argument('--aaronson_seed', type=int, default=42, help='Seed for Aaronson watermarking')
    parser.add_argument('--aaronson_remasking_strategy', type=str, default='original',
                       choices=['original', 'dual_gate', 'blend', 'hard_favor'],
                       help='Remasking strategy')
    parser.add_argument('--aaronson_tau_wm', type=float, default=0.2, help='Watermark confidence threshold for dual_gate')
    parser.add_argument('--aaronson_tau_orig', type=float, default=0.01, help='Original confidence threshold for dual_gate')
    parser.add_argument('--aaronson_lambda', type=float, default=0.7, help='Blending weight for blend strategy')
    
    args = parser.parse_args()
    
    # Read prompts
    print(f"Reading prompts from {args.prompts_file}...")
    with open(args.prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"Processing {len(prompts)} prompts")
    
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
    
    # Load GPT-2 for perplexity evaluation (external judge)
    # Use CPU to avoid CUDA assertion errors from vocabulary mismatch
    print("Loading GPT-2 for perplexity calculation (on CPU)...")
    eval_model = AutoModelForCausalLM.from_pretrained("gpt2").cpu().eval()
    eval_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eval_tokenizer.pad_token = eval_tokenizer.eos_token  # Set pad token
    
    # Print configuration
    print(f"\n{'='*60}")
    print("GENERATION CONFIGURATION:")
    print(f"{'='*60}")
    print(f"Generation length: {args.gen_length} tokens")
    print(f"Steps: {args.steps}")
    print(f"Temperature: {args.temperature}")
    print(f"Perplexity: Calculated using GPT-2 on CPU (external judge)")
    print(f"Watermark type: {args.watermark_type}")
    if args.watermark_type == 'aaronson':
        print(f"Watermark steps: {args.watermark_steps if args.watermark_steps else 'all'}")
        print(f"Aaronson seed: {args.aaronson_seed}")
        print(f"Remasking strategy: {args.aaronson_remasking_strategy}")
        if args.aaronson_remasking_strategy == 'dual_gate':
            print(f"  tau_wm={args.aaronson_tau_wm}, tau_orig={args.aaronson_tau_orig}")
        elif args.aaronson_remasking_strategy == 'blend':
            print(f"  lambda={args.aaronson_lambda}")
    print(f"{'='*60}\n")
    
    # Process each prompt
    results = []
    
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        # Tokenize prompt
        prompt_tokens = tokenizer(prompt)["input_ids"]
        prompt_tensor = torch.tensor([prompt_tokens]).to(args.device)
        
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
                remasking='low_confidence',
                mask_id=args.mask_id,
                watermark_type=args.watermark_type if args.watermark_type != 'none' else 'green_list',
                amplification=0.0 if args.watermark_type == 'none' else None,
                aaronson_seed=args.aaronson_seed,
                watermark_steps=args.watermark_steps,
                vocab_size=args.vocab_size,
                special_token_ids=special_token_ids,
                aaronson_remasking_strategy=args.aaronson_remasking_strategy,
                aaronson_tau_wm=args.aaronson_tau_wm,
                aaronson_tau_orig=args.aaronson_tau_orig,
                aaronson_lambda=args.aaronson_lambda
            )
        
        # Extract generated tokens
        generated_tokens = generated[0, len(prompt_tokens):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Calculate perplexity using GPT-2 (external judge - runs on CPU to avoid CUDA errors)
        perplexity = calculate_perplexity_gpt2(eval_model, eval_tokenizer, generated_text)
        
        # Calculate watermark score if using Aaronson
        watermark_score = None
        normalized_score = None
        if args.watermark_type == 'aaronson':
            score, actual_length, per_token_scores = calculate_aaronson_watermark_score(
                generated_tokens.unsqueeze(0),
                vocab_size=args.vocab_size,
                seed=args.aaronson_seed,
                special_token_ids=special_token_ids,
                position_offset=len(prompt_tokens)
            )
            watermark_score = float(score)
            normalized_score = float(score / actual_length) if actual_length > 0 else 0.0
        
        # Store result
        result = {
            'prompt_id': idx,
            'prompt': prompt,
            'generated_text': generated_text,
            'full_text': full_text,
            'prompt_length': len(prompt_tokens),
            'generated_length': len(generated_tokens.tolist()),
            'perplexity': perplexity,
            'watermark_score': watermark_score,
            'normalized_watermark_score': normalized_score
        }
        results.append(result)
    
    # Save results
    output_path = Path(args.output_file)
    print(f"\nSaving results to {output_path}...")
    
    # Calculate statistics
    perplexities = [r['perplexity'] for r in results if r['perplexity'] is not None]
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else None
    
    with open(output_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'num_prompts': len(prompts),
            'average_perplexity': avg_perplexity,
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Results saved to: {output_path}")
    print(f"Processed {len(results)} prompts")
    if avg_perplexity is not None:
        print(f"Average Perplexity (GPT-2): {avg_perplexity:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

