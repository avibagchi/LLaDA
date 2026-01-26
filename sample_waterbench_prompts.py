#!/usr/bin/env python3
"""
Sample random prompts from all WaterBench JSONL files.
Combines all files and randomly samples N prompts.
"""
import json
import argparse
import random
from pathlib import Path
import glob


def load_all_waterbench_files(waterbench_dir="water-bench", max_prompt_tokens=500):
    """
    Load all prompts from all JSONL files in water-bench directory.
    Filters out prompts with contexts longer than max_prompt_tokens.
    
    Args:
        waterbench_dir: Directory containing JSONL files
        max_prompt_tokens: Maximum number of tokens in context+input (default: 500)
                          Prompts exceeding this will be filtered out
    """
    all_prompts = []
    filtered_count = 0
    
    # Find all JSONL files
    jsonl_files = glob.glob(f"{waterbench_dir}/*.jsonl")
    
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {waterbench_dir}/")
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for jsonl_file in sorted(jsonl_files):
        print(f"  - {jsonl_file}")
    
    # Load all prompts
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        # Calculate total prompt length (context + input)
                        context = data.get('context', '')
                        input_text = data.get('input', '')
                        
                        # Use 'length' field if available (total tokens including context and input)
                        # Otherwise estimate from input_length + context
                        if 'length' in data:
                            total_tokens = data['length']
                        elif 'input_length' in data:
                            # input_length is tokenized input, estimate context tokens
                            # Rough estimate: ~1.3 tokens per word (conservative)
                            context_words = len(context.split()) if context else 0
                            context_tokens = int(context_words * 1.3)
                            total_tokens = data['input_length'] + context_tokens
                        else:
                            # Fallback: rough estimate from text length (~4 chars per token)
                            total_text = f"{context}\n{input_text}".strip()
                            total_tokens = len(total_text) // 4
                        
                        # Filter out prompts that are too long
                        if total_tokens > max_prompt_tokens:
                            filtered_count += 1
                            continue
                        
                        # Add source file info
                        data['_source_file'] = Path(jsonl_file).name
                        all_prompts.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON in {jsonl_file} line {line_num}: {e}")
    
    print(f"\nTotal prompts loaded: {len(all_prompts)}")
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} prompts with contexts longer than {max_prompt_tokens} tokens")
    return all_prompts


def sample_prompts(all_prompts, num_samples, seed=None):
    """Randomly sample N prompts from all prompts."""
    if seed is not None:
        random.seed(seed)
    
    if num_samples >= len(all_prompts):
        print(f"Requested {num_samples} prompts, but only {len(all_prompts)} available. Using all prompts.")
        return all_prompts
    
    sampled = random.sample(all_prompts, num_samples)
    print(f"Sampled {len(sampled)} prompts from {len(all_prompts)} total prompts")
    
    # Print distribution by source file
    source_counts = {}
    for prompt in sampled:
        source = prompt.get('_source_file', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\nDistribution by source file:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} prompts")
    
    return sampled


def main():
    parser = argparse.ArgumentParser(description='Sample random prompts from all WaterBench files')
    parser.add_argument('--waterbench_dir', type=str, default='water-bench',
                       help='Directory containing WaterBench JSONL files')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of random prompts to sample (default: 500)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output JSONL file path')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('--max_prompt_tokens', type=int, default=500,
                       help='Maximum tokens in context+input (default: 500). Prompts exceeding this are filtered out.')
    
    args = parser.parse_args()
    
    # Load all prompts (with filtering)
    all_prompts = load_all_waterbench_files(args.waterbench_dir, max_prompt_tokens=args.max_prompt_tokens)
    
    # Sample random prompts
    sampled_prompts = sample_prompts(all_prompts, args.num_samples, args.seed)
    
    # Write to output file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt in sampled_prompts:
            # Remove source file info before writing (optional, keep it for debugging)
            # prompt.pop('_source_file', None)
            f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
    
    print(f"\nSampled prompts saved to: {args.output_file}")
    print(f"Total prompts in output: {len(sampled_prompts)}")


if __name__ == "__main__":
    main()
