#!/usr/bin/env python3
"""
Quick watermark test with simple questions.
"""

import sys
import os
import torch
import random

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate import generate, calculate_green_matches
from transformers import AutoTokenizer, AutoModel


def test_watermark_impact():
    """Test watermark impact on simple questions."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True
    )
    
    # Simple test questions
    questions = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "What color do you get when you mix red and blue?",
        "What is 10 * 5?",
        "What is the largest planet in our solar system?"
    ]
    
    # Test parameters
    gamma = 0.5
    amplification = 2.0
    watermark_steps = 10
    model_seed = 1
    
    print(f"\n🧪 TESTING WATERMARK IMPACT")
    print(f"Gamma: {gamma}, Amplification: {amplification}, Steps: {watermark_steps}")
    print(f"Questions: {len(questions)}")
    print("="*60)
    
    # Set random seed
    torch.manual_seed(model_seed)
    
    results = []
    
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        
        # Prepare prompt
        prompt = f"Answer this question: {question}\n\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Test without watermarking
        print("  Generating without watermarking...")
        generated_baseline = generate(
            model, input_ids, steps=32, gen_length=64, block_length=32,
            temperature=0., cfg_scale=0., remasking='low_confidence',
            gamma=0.0, amplification=0.0
        )
        baseline_text = tokenizer.batch_decode(
            generated_baseline[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        
        # Test with watermarking
        print("  Generating with watermarking...")
        generated_watermarked = generate(
            model, input_ids, steps=32, gen_length=64, block_length=32,
            temperature=0., cfg_scale=0., remasking='low_confidence',
            gamma=gamma, amplification=amplification, watermark_steps=watermark_steps
        )
        watermarked_text = tokenizer.batch_decode(
            generated_watermarked[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        
        # Calculate watermark detection
        baseline_matches = calculate_green_matches(generated_baseline[:, input_ids.shape[1]:], gamma=gamma)
        watermarked_matches = calculate_green_matches(generated_watermarked[:, input_ids.shape[1]:], gamma=gamma)
        
        print(f"  Baseline: {baseline_text}")
        print(f"  Watermarked: {watermarked_text}")
        print(f"  Baseline match %: {baseline_matches[0]:.3f}")
        print(f"  Watermarked match %: {watermarked_matches[0]:.3f}")
        
        results.append({
            'question': question,
            'baseline_text': baseline_text,
            'watermarked_text': watermarked_text,
            'baseline_match': baseline_matches[0],
            'watermarked_match': watermarked_matches[0],
            'match_improvement': watermarked_matches[0] - baseline_matches[0]
        })
    
    # Print summary
    print(f"\n📊 SUMMARY")
    print("="*60)
    print(f"{'Question':<30} {'Baseline':<8} {'Watermarked':<8} {'Improvement':<10}")
    print("-"*60)
    
    for result in results:
        print(f"{result['question'][:29]:<30} {result['baseline_match']:<8.3f} {result['watermarked_match']:<8.3f} {result['match_improvement']:<+10.3f}")
    
    avg_improvement = sum(r['match_improvement'] for r in results) / len(results)
    print(f"{'Average':<30} {'':8} {'':8} {avg_improvement:<+10.3f}")
    
    print(f"\n✅ Test completed!")
    print(f"Average watermark detection improvement: {avg_improvement:+.3f}")


if __name__ == '__main__':
    test_watermark_impact()

