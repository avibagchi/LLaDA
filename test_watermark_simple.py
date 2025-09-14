#!/usr/bin/env python3
"""
Simple test script for LLaDA watermarking implementation.
"""

import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate import generate, calculate_green_matches
from transformers import AutoTokenizer, AutoModel
import torch
import math


def test_basic_watermarking():
    """Test basic watermarking functionality."""
    print("Testing LLaDA watermarking implementation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
        print("Model loaded successfully!")
        
        # Simple test prompt
        prompt = "Hello, how are you?"
        
        # Add special tokens for the Instruct model
        m = [{"role": "user", "content": prompt}, ]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
        print(f"Test prompt: {prompt}")
        print(f"Input IDs shape: {input_ids.shape}")
        
        # Test parameters
        gamma = 0.5
        amplification = 2.0
        gen_length = 32  # Short generation for testing
        steps = 32
        
        print(f"Testing with gamma={gamma}, amplification={amplification}")
        print(f"Generation length: {gen_length}, Steps: {steps}")
        
        # Generate watermarked text
        print("\nGenerating watermarked text...")
        out = generate(model, input_ids, steps=steps, gen_length=gen_length,
                      block_length=16, temperature=0., cfg_scale=0., 
                      remasking='low_confidence', gamma=gamma, amplification=amplification)
        
        generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Generated text: {generated_text}")
        
        # Test watermark detection
        print("\nTesting watermark detection...")
        max_match_percent, actual_length, max_num_matches, best_start, match_arr = calculate_green_matches(
            out[:, input_ids.shape[1]:], gamma=gamma
        )
        
        print(f"Watermark detection results:")
        print(f"  Max match percent: {max_match_percent:.4f}")
        print(f"  Actual length used: {actual_length}")
        print(f"  Max num matches: {max_num_matches}")
        print(f"  Best start: {best_start}")
        
        # Calculate Z-score
        true_num_green = gamma * actual_length
        if math.sqrt(true_num_green * (1-gamma)) == 0:
            z_score = 0
        else:
            z_score = (max_num_matches - true_num_green) / math.sqrt(true_num_green * (1-gamma))
        
        print(f"  Z-score: {z_score:.4f}")
        print(f"  Expected green tokens: {true_num_green:.1f}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_sweep():
    """Test different parameter combinations."""
    print("\n" + "="*60)
    print("PARAMETER SWEEP TEST")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load model and tokenizer
        model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
        
        prompt = "The quick brown fox jumps over the lazy dog."
        
        # Add special tokens for the Instruct model
        m = [{"role": "user", "content": prompt}, ]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
        # Test different parameters
        gamma_values = [0.1, 0.5, 0.9]
        amplification_values = [0.0, 2.0, 5.0]
        
        print(f"Prompt: {prompt}")
        print(f"Testing {len(gamma_values)} gamma values × {len(amplification_values)} amplification values")
        print()
        
        results = []
        
        for gamma in gamma_values:
            for amplification in amplification_values:
                print(f"Testing gamma={gamma}, amplification={amplification}")
                
                # Generate watermarked text
                out = generate(model, input_ids, steps=32, gen_length=32,
                              block_length=16, temperature=0., cfg_scale=0., 
                              remasking='low_confidence', gamma=gamma, amplification=amplification)
                
                # Calculate detection metrics
                max_match_percent, actual_length, max_num_matches, best_start, match_arr = calculate_green_matches(
                    out[:, input_ids.shape[1]:], gamma=gamma
                )
                
                true_num_green = gamma * actual_length
                if math.sqrt(true_num_green * (1-gamma)) == 0:
                    z_score = 0
                else:
                    z_score = (max_num_matches - true_num_green) / math.sqrt(true_num_green * (1-gamma))
                
                results.append({
                    'gamma': gamma,
                    'amplification': amplification,
                    'z_score': z_score,
                    'max_match_percent': max_match_percent,
                    'actual_length': actual_length
                })
                
                print(f"  Z-score: {z_score:.4f}, Match %: {max_match_percent:.4f}")
        
        # Print summary table
        print("\n" + "="*80)
        print("PARAMETER SWEEP RESULTS")
        print("="*80)
        print(f"{'Gamma':<8} {'Amplification':<12} {'Z-score':<10} {'Match %':<10} {'Length':<8}")
        print("-" * 80)
        for result in results:
            print(f"{result['gamma']:<8.2f} {result['amplification']:<12.1f} {result['z_score']:<10.4f} {result['max_match_percent']:<10.4f} {result['actual_length']:<8}")
        
        print("\n✅ Parameter sweep completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Parameter sweep failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("LLaDA Watermarking Test Suite")
    print("="*50)
    
    # Test basic functionality
    success1 = test_basic_watermarking()
    
    # Test parameter sweep
    success2 = test_parameter_sweep()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
