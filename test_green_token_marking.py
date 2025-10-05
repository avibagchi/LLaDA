#!/usr/bin/env python3
"""
Simple test script to demonstrate green token marking functionality.
"""

import sys
import os
import torch

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_watermark_metrics import identify_green_tokens_with_best_start, format_text_with_bolded_green_tokens
from transformers import AutoTokenizer

def test_green_token_marking():
    """Test the green token identification and marking functions."""
    print("Testing green token marking functionality...")
    
    # Create a simple tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
        print(f"Tokenizer loaded with vocab size: {vocab_size}")
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        return
    
    # Create sample generated tokens
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_tokens = tokenizer(sample_text, return_tensors="pt").input_ids
    print(f"Sample text: {sample_text}")
    print(f"Sample tokens shape: {sample_tokens.shape}")
    
    # Test green token identification with best_start=0
    gamma = 0.5  # 50% green tokens
    best_start = 0  # Use start offset 0
    green_positions = identify_green_tokens_with_best_start(
        sample_tokens, gamma, vocab_size, best_start
    )
    
    print(f"Green positions: {green_positions}")
    print(f"Number of green tokens: {green_positions.sum().item()}")
    
    # Test formatting
    formatted_text = format_text_with_bolded_green_tokens(
        tokenizer, sample_tokens, green_positions
    )
    
    print(f"\nOriginal text: {sample_text}")
    print(f"Text with green tokens marked: {formatted_text}")
    
    # Save to test file
    with open('test_green_token_marking.txt', 'w', encoding='utf-8') as f:
        f.write("Test Green Token Marking\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Original text: {sample_text}\n")
        f.write(f"Text with green tokens marked: {formatted_text}\n")
        f.write(f"Number of green tokens: {green_positions.sum().item()}\n")
        f.write(f"Total tokens: {sample_tokens.shape[1]}\n")
    
    print(f"\nTest file created: test_green_token_marking.txt")

if __name__ == '__main__':
    test_green_token_marking()




