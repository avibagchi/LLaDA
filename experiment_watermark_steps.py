#!/usr/bin/env python3
"""
Experiment: Test Aaronson watermark scores across different prompts and step values.
Generates a matrix of normalized scores for analysis.
"""
import torch
import argparse
import json
import csv
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from generate import generate, calculate_aaronson_watermark_score, get_special_token_ids
from tqdm import tqdm


# 10 diverse test prompts
DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain the concept of machine learning in simple terms.",
    "Write a short story about a robot learning to paint.",
    "Question: If John has 5 apples and buys 3 more, how many does he have? Answer:",
    "Describe the water cycle in nature.",
    "What are the main differences between Python and JavaScript?",
    "Tell me about the history of the Internet.",
    "How does photosynthesis work in plants?",
    "Question: A train travels 60 miles in 1 hour. How far does it travel in 3 hours? Answer:",
    "Explain what DNA is and why it's important."
]


def main():
    parser = argparse.ArgumentParser(description='Experiment with Aaronson watermark across prompts and steps')
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Base', help='Model path')
    parser.add_argument('--gen_length', type=int, default=256, help='Number of tokens to generate')
    parser.add_argument('--steps', type=int, default=256, help='Number of sampling steps')
    parser.add_argument('--block_length', type=int, default=256, help='Block length for generation')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0 for greedy)')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale')
    parser.add_argument('--aaronson_seed', type=int, default=42, help='Seed for Aaronson watermarking')
    parser.add_argument('--mask_id', type=int, default=126336, help='Mask token ID')
    parser.add_argument('--vocab_size', type=int, default=126464, help='Vocabulary size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_json', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV file path')
    
    # Step values to test (10 values from 0 to max steps)
    parser.add_argument('--step_values', type=str, default='0,25,50,75,100,128,150,175,200,256',
                       help='Comma-separated list of watermark_steps values to test')
    
    args = parser.parse_args()
    
    # Parse step values
    step_values = [int(x.strip()) for x in args.step_values.split(',')]
    print(f"Testing {len(step_values)} watermark step values: {step_values}")
    
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
    
    # Results storage
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'config': {
            'model_path': args.model_path,
            'gen_length': args.gen_length,
            'steps': args.steps,
            'block_length': args.block_length,
            'aaronson_seed': args.aaronson_seed,
            'step_values': step_values,
        },
        'data': []
    }
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: Aaronson Watermark Scores")
    print(f"Prompts: {len(DEFAULT_PROMPTS)}")
    print(f"Step values: {len(step_values)}")
    print(f"Total generations: {len(DEFAULT_PROMPTS) * len(step_values)}")
    print(f"{'='*80}\n")
    
    # Run experiments
    total_experiments = len(DEFAULT_PROMPTS) * len(step_values)
    pbar = tqdm(total=total_experiments, desc="Running experiments")
    
    for prompt_idx, prompt in enumerate(DEFAULT_PROMPTS):
        # Tokenize prompt once
        prompt_tokens = tokenizer(prompt)["input_ids"]
        prompt_tensor = torch.tensor([prompt_tokens]).to(args.device)
        
        for watermark_steps in step_values:
            # Generate with watermarking
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
                    watermark_steps=watermark_steps if watermark_steps > 0 else 0,
                    vocab_size=args.vocab_size,
                    special_token_ids=special_token_ids
                )
            
            # Extract generated tokens
            generated_tokens = generated[0, len(prompt_tokens):]
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate watermark score
            score, actual_length, per_token_scores, _ = calculate_aaronson_watermark_score(
                generated_tokens.unsqueeze(0),
                vocab_size=args.vocab_size,
                seed=args.aaronson_seed,
                special_token_ids=special_token_ids,
                position_offset=len(prompt_tokens)
            )
            
            normalized_score = score / actual_length if actual_length > 0 else 0
            
            # Store results
            result_entry = {
                'prompt_idx': prompt_idx + 1,
                'prompt': prompt,
                'watermark_steps': watermark_steps,
                'raw_score': score,
                'normalized_score': normalized_score,
                'length': actual_length,
                'generated_text': generated_text[:200] + '...' if len(generated_text) > 200 else generated_text
            }
            results['data'].append(result_entry)
            
            pbar.set_postfix({
                'prompt': f"{prompt_idx+1}/{len(DEFAULT_PROMPTS)}", 
                'steps': watermark_steps,
                'norm_score': f"{normalized_score:.4f}"
            })
            pbar.update(1)
    
    pbar.close()
    
    # Save results to JSON
    if args.output_json:
        json_file = args.output_json
    else:
        json_file = f"watermark_experiment_{results['timestamp']}.json"
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {json_file}")
    
    # Save results to CSV (matrix format)
    if args.output_csv:
        csv_file = args.output_csv
    else:
        csv_file = f"watermark_experiment_{results['timestamp']}.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row: watermark_steps values
        header = ['Prompt'] + [f"steps={s}" for s in step_values]
        writer.writerow(header)
        
        # Data rows: one per prompt
        for prompt_idx in range(len(DEFAULT_PROMPTS)):
            prompt = DEFAULT_PROMPTS[prompt_idx]
            row = [f"P{prompt_idx+1}: {prompt[:50]}..."]
            
            for watermark_steps in step_values:
                # Find the corresponding result
                entry = next(
                    r for r in results['data'] 
                    if r['prompt_idx'] == prompt_idx + 1 and r['watermark_steps'] == watermark_steps
                )
                row.append(f"{entry['normalized_score']:.4f}")
            
            writer.writerow(row)
    
    print(f"✓ CSV matrix saved to: {csv_file}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    for watermark_steps in step_values:
        scores = [r['normalized_score'] for r in results['data'] if r['watermark_steps'] == watermark_steps]
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        print(f"Watermark steps={watermark_steps:3d}: "
              f"avg={avg_score:.4f}, min={min_score:.4f}, max={max_score:.4f}")
    
    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

