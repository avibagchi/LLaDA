#!/usr/bin/env python3
"""
Simple test script to demonstrate green token formatting functionality.
"""

import sys
import os
import torch

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_watermark_metrics import identify_green_tokens, format_text_with_bolded_green_tokens
from transformers import AutoTokenizer

def test_green_token_formatting():
    """Test the green token identification and formatting functions."""
    print("Testing green token formatting functionality...")
    
    # Create a simple tokenizer (using a smaller model for testing)
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
    
    # Test green token identification
    gamma = 0.5  # 50% green tokens
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    green_positions = identify_green_tokens(sample_tokens, gamma, vocab_size, device)
    
    print(f"Green positions: {green_positions}")
    print(f"Number of green tokens: {green_positions.sum().item()}")
    
    # Test formatting
    formatted_html = format_text_with_bolded_green_tokens(tokenizer, sample_tokens, green_positions, 'html')
    formatted_markdown = format_text_with_bolded_green_tokens(tokenizer, sample_tokens, green_positions, 'markdown')
    formatted_console = format_text_with_bolded_green_tokens(tokenizer, sample_tokens, green_positions, 'console')
    
    print(f"\nHTML format: {formatted_html}")
    print(f"\nMarkdown format: {formatted_markdown}")
    print(f"\nConsole format: {formatted_console}")
    
    # Save to test files
    with open('test_green_tokens.html', 'w', encoding='utf-8') as f:
        f.write(f"<h1>Test Green Token Formatting</h1>\n")
        f.write(f"<p>Original text: {sample_text}</p>\n")
        f.write(f"<p>With green tokens bolded: {formatted_html}</p>\n")
    
    with open('test_green_tokens.md', 'w', encoding='utf-8') as f:
        f.write(f"# Test Green Token Formatting\n\n")
        f.write(f"**Original text:** {sample_text}\n\n")
        f.write(f"**With green tokens bolded:** {formatted_markdown}\n\n")
    
    print(f"\nTest files created:")
    print(f"  - test_green_tokens.html")
    print(f"  - test_green_tokens.md")

if __name__ == '__main__':
    test_green_token_formatting()




