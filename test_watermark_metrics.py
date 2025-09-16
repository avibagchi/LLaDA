#!/usr/bin/env python3
"""
Test script for LLaDA watermarking that tracks specific metrics.
"""

import sys
import os
import torch
import torch.nn.functional as F
import math
import csv
import random

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate import generate, calculate_green_matches, get_special_token_ids
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def load_prompts_from_file(filename):
    """Load prompts from a text file."""
    try:
        prompts = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    prompts.append(line)
        return prompts
    except FileNotFoundError:
        print(f"Warning: {filename} not found.")
        return []
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []




def calculate_perplexity(model, tokenizer, generated_tokens, device):
    """Calculate perplexity of generated text using GPT-2."""
    try:
        # Use GPT-2 for perplexity calculation (matching Score-Entropy approach)
        eval_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
        eval_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Convert LLaDA tokens to text and then to GPT-2 tokens
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        gpt2_tokens = eval_tokenizer(generated_text, return_tensors="pt").input_ids.to(device)
        
        if gpt2_tokens.shape[1] < 2:
            return 0.0
        
        with torch.no_grad():
            # Use the same approach as Score-Entropy
            loss, logits = eval_model(gpt2_tokens, labels=gpt2_tokens)[:2]
            logits = logits.transpose(-1, -2)
            perplexity = F.cross_entropy(logits[..., :-1], gpt2_tokens[..., 1:], reduction="none").mean(dim=-1).exp().mean()
            
        return float(perplexity.item())
    except Exception as e:
        print(f"Warning: Could not calculate perplexity: {e}")
        return 0.0


def test_watermarking_metrics():
    """Test watermarking with specific metrics tracking using prompts from prompts.txt."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    # Get special token IDs to exclude from amplification
    special_token_ids = get_special_token_ids(tokenizer)
    print(f"Special token IDs to exclude from amplification: {special_token_ids}")
    
    # Load prompts from prompts.txt
    prompts_file = 'prompts.txt'
    prompts_list = load_prompts_from_file(prompts_file)
    
    if not prompts_list:
        print(f"No prompts found in {prompts_file}")
        return []
    
    print(f"Loaded {len(prompts_list)} prompts from {prompts_file}")
    
    # Test parameters
    gamma_list = [0.025] # [0.1, 0.25, 0.5, 0.75, 0.9]
    amp_list = [4] # [0.0, 1.5, 2.0, 3.0, 5.0]
    step_to_watermark_list = [50] # [None, 2, 5, 10]  # None=all steps, 2=steps 1-2, 5=steps 1-5, 10=steps 1-10
    model_seed_list = [1] # [1, 2, 3]  # Different random seeds
    
    all_results = []
    
    print("="*80)
    print("LLaDA WATERMARKING METRICS TEST - PROMPT-BASED GENERATION")
    print("="*80)
    print(f"Testing {len(prompts_list)} prompts × {len(gamma_list)} gamma × {len(amp_list)} amplification × {len(step_to_watermark_list)} step patterns × {len(model_seed_list)} seeds")
    print(f"Parameters: gamma={gamma_list}, amplification={amp_list}, step_to_watermark={step_to_watermark_list}, seeds={model_seed_list}")
    print("="*80)
    
    results = []
    
    # Loop through each prompt
    prompts_list = prompts_list[:30] # random.sample(prompts_list, 30)
    for prompt_idx, prompt in enumerate(prompts_list):
        print(f"\n--- Testing Prompt {prompt_idx + 1}/{len(prompts_list)}: {prompt[:50]}... ---")
        
        for model_seed in model_seed_list:
            torch.manual_seed(model_seed)
            
            for gamma in gamma_list:
                for amplification in amp_list:
                    for step_to_watermark in step_to_watermark_list:
                        print(f"Testing prompt {prompt_idx + 1}, seed={model_seed}, gamma={gamma}, amplification={amplification}, steps={step_to_watermark}")
                        
                        # Tokenize the prompt
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                        
                        try:
                            # Generate text with watermarking
                            out = generate(model, input_ids, steps=128, gen_length=128, 
                                         block_length=32, temperature=0., cfg_scale=0., 
                                         remasking='low_confidence', gamma=gamma, 
                                         amplification=amplification, 
                                         watermark_steps=step_to_watermark,
                                         special_token_ids=special_token_ids)
                            
                            # breakpoint()
                            generated_tokens = out[:, input_ids.shape[1]:]
                            generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                            full_output = tokenizer.batch_decode(out[0], skip_special_tokens=True)[0]
                            print(f"  Full output: {full_output}")
                            print(f"  Generated text: {generated_text}")
                            
                            # Calculate watermark detection metrics
                            max_match_percent, actual_length, max_num_matches, best_start, match_arr = calculate_green_matches(
                                generated_tokens, gamma=gamma
                            )
                            
                            # Calculate Z-score
                            true_num_green = gamma * actual_length
                            if math.sqrt(true_num_green * (1-gamma)) == 0:
                                z_score = 0
                            else:
                                z_score = (max_num_matches - true_num_green) / math.sqrt(true_num_green * (1-gamma))
                            
                            # Calculate perplexity
                            perplexity = calculate_perplexity(model, tokenizer, generated_tokens, device)
                            
                            # Store results
                            result = {
                                "prompt_idx": prompt_idx + 1,
                                "prompt": prompt,
                                "model_seed": model_seed,
                                "gamma": gamma,
                                "amplification": amplification,
                                "step_to_watermark": step_to_watermark,
                                "match_percent": max_match_percent,
                                "perplexity": perplexity,
                                "z_score": z_score,
                                "source_file": prompts_file
                            }
                            results.append(result)
                            
                            print(f"  Match %: {max_match_percent:.4f}, Z-score: {z_score:.4f}, Perplexity: {perplexity:.2f}")
                            
                        except Exception as e:
                            print(f"  Error: {e}")
                            # Still record the attempt
                            result = {
                                "prompt_idx": prompt_idx + 1,
                                "prompt": prompt,
                                "model_seed": model_seed,
                                "gamma": gamma,
                                "amplification": amplification,
                                "step_to_watermark": step_to_watermark,
                                "match_percent": 0.0,
                                "perplexity": 0.0,
                                "z_score": 0.0,
                                "source_file": prompts_file
                            }
                            results.append(result)
        
    # Save results to CSV
    filename = 'watermark_results_prompts.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ["prompt_idx", "prompt", "model_seed", "gamma", "amplification", "step_to_watermark", "match_percent", "perplexity", "z_score", "source_file"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {filename}")
    all_results = results
    
    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    # Group by amplification
    for amp in amp_list:
        amp_results = [r for r in all_results if r['amplification'] == amp]
        if amp_results:
            avg_z_score = sum(r['z_score'] for r in amp_results) / len(amp_results)
            avg_match_percent = sum(r['match_percent'] for r in amp_results) / len(amp_results)
            avg_perplexity = sum(r['perplexity'] for r in amp_results) / len(amp_results)
            print(f"Amplification {amp}: Avg Z-score: {avg_z_score:.4f}, Avg Match %: {avg_match_percent:.4f}, Avg Perplexity: {avg_perplexity:.2f}")
    
    # Group by gamma
    print("\nBy Gamma:")
    for gamma in gamma_list:
        gamma_results = [r for r in all_results if r['gamma'] == gamma]
        if gamma_results:
            avg_z_score = sum(r['z_score'] for r in gamma_results) / len(gamma_results)
            avg_match_percent = sum(r['match_percent'] for r in gamma_results) / len(gamma_results)
            print(f"Gamma {gamma}: Avg Z-score: {avg_z_score:.4f}, Avg Match %: {avg_match_percent:.4f}")
    
    return all_results


if __name__ == '__main__':
    test_watermarking_metrics()
