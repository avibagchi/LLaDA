#!/usr/bin/env python3
"""
Simple script to generate text with LLaDA using Aaronson watermarking.
"""
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from generate import generate, calculate_aaronson_watermark_score, get_special_token_ids


def main():
    parser = argparse.ArgumentParser(description='Generate text with LLaDA and Aaronson watermarking')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt to start generation')
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Base', help='Model path')
    parser.add_argument('--gen_length', type=int, default=256, help='Number of tokens to generate')
    parser.add_argument('--steps', type=int, default=256, help='Number of sampling steps')
    parser.add_argument('--block_length', type=int, default=256, help='Block length for generation')
    parser.add_argument('--temperature', type=float, default=0, help='Sampling temperature (0 for greedy)')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale')
    parser.add_argument('--watermark_steps', type=int, default=None, help='Watermark up to this step (None = all steps)')
    parser.add_argument('--aaronson_seed', type=int, default=42, help='Seed for Aaronson watermarking')
    parser.add_argument('--mask_id', type=int, default=126336, help='Mask token ID')
    parser.add_argument('--vocab_size', type=int, default=126464, help='Vocabulary size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Aaronson remasking strategy parameters
    parser.add_argument('--aaronson_remasking_strategy', type=str, default='original',
                       choices=['original', 'dual_gate', 'blend', 'hard_favor'],
                       help='Remasking strategy: original (best quality), dual_gate (balanced), '
                            'blend (configurable), hard_favor (strongest watermark)')
    parser.add_argument('--aaronson_tau_wm', type=float, default=0.2,
                       help='Watermark confidence threshold for dual_gate (default: 0.2)')
    parser.add_argument('--aaronson_tau_orig', type=float, default=0.01,
                       help='Original confidence threshold for dual_gate (default: 0.01)')
    parser.add_argument('--aaronson_lambda', type=float, default=0.7,
                       help='Blending weight for blend strategy (0-1, default: 0.7). Higher = stronger watermark')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model = AutoModel.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Get special token IDs
    special_token_ids = get_special_token_ids(tokenizer)
    print(f"Special token IDs: {special_token_ids}")
    
    # Tokenize prompt
    prompt_tokens = tokenizer(args.prompt)["input_ids"]
    prompt_tensor = torch.tensor([prompt_tokens]).to(args.device)
    
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Prompt length: {len(prompt_tokens)} tokens")
    print(f"Generation length: {args.gen_length} tokens")
    print(f"Steps: {args.steps}")
    print(f"Watermark steps: {args.watermark_steps if args.watermark_steps else 'all'}")
    print(f"Aaronson seed: {args.aaronson_seed}")
    print(f"Remasking strategy: {args.aaronson_remasking_strategy}")
    if args.aaronson_remasking_strategy == 'dual_gate':
        print(f"  tau_wm={args.aaronson_tau_wm}, tau_orig={args.aaronson_tau_orig}")
    elif args.aaronson_remasking_strategy == 'blend':
        print(f"  lambda={args.aaronson_lambda}")
    print(f"{'='*60}\n")
    
    # Generate
    print("Generating...")
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
            watermark_type='aaronson',
            aaronson_seed=args.aaronson_seed,
            watermark_steps=args.watermark_steps,
            vocab_size=args.vocab_size,
            special_token_ids=special_token_ids,
            aaronson_remasking_strategy=args.aaronson_remasking_strategy,
            aaronson_tau_wm=args.aaronson_tau_wm,
            aaronson_tau_orig=args.aaronson_tau_orig,
            aaronson_lambda=args.aaronson_lambda
        )
    
    # Extract generated tokens (after prompt)
    generated_tokens = generated[0, len(prompt_tokens):]
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # Calculate watermark score (MUST pass position_offset to match generation!)
    score, actual_length, per_token_scores = calculate_aaronson_watermark_score(
        generated_tokens.unsqueeze(0),
        vocab_size=args.vocab_size,
        seed=args.aaronson_seed,
        special_token_ids=special_token_ids,
        position_offset=len(prompt_tokens)  # Critical: match generation offset
    )
    
    normalized_score = score / actual_length if actual_length > 0 else 0
    
    # Print results
    print(f"\n{'='*60}")
    print("FULL TEXT:")
    print(f"{'='*60}")
    print(full_text)
    print(f"\n{'='*60}")
    print("GENERATED TEXT (excluding prompt):")
    print(f"{'='*60}")
    print(generated_text)
    print(f"\n{'='*60}")
    print("WATERMARK DETECTION:")
    print(f"{'='*60}")
    print(f"Aaronson watermark score: {score:.4f}")
    print(f"Normalized score: {normalized_score:.4f}")
    print(f"Analyzed length: {actual_length} tokens")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

