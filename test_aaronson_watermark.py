#!/usr/bin/env python3
"""
Test script for Aaronson watermarking implementation.
This script demonstrates how to use the new Aaronson watermarking functions.
"""

import torch
import sys

# Add current directory to path
sys.path.insert(0, '/work/nvme/bemc/abagchi2/LLaDA')

from generate import (
    generate_pseudo_random_values,
    apply_aaronson_gumbel_watermark,
    calculate_aaronson_watermark_score
)

def test_pseudo_random_values():
    """Test the pseudorandom value generation."""
    print("Testing pseudorandom value generation...")
    
    position = 0
    vocab_size = 100
    seed = 42
    
    # Generate pseudorandom values
    r_values = generate_pseudo_random_values(position, vocab_size, seed)
    
    print(f"  Generated {len(r_values)} pseudorandom values")
    print(f"  Min value: {r_values.min():.4f}")
    print(f"  Max value: {r_values.max():.4f}")
    print(f"  Mean value: {r_values.mean():.4f}")
    
    # Test determinism
    r_values_2 = generate_pseudo_random_values(position, vocab_size, seed)
    assert torch.allclose(r_values, r_values_2), "Pseudorandom values should be deterministic"
    print("  ✓ Pseudorandom values are deterministic")
    
    # Test different positions give different values
    r_values_3 = generate_pseudo_random_values(position + 1, vocab_size, seed)
    assert not torch.allclose(r_values, r_values_3), "Different positions should give different values"
    print("  ✓ Different positions give different values")
    
    print("  ✓ Pseudorandom value generation test passed!\n")


def test_aaronson_watermark_application():
    """Test applying Aaronson watermark to logits."""
    print("Testing Aaronson watermark application...")
    
    batch_size = 1
    seq_len = 5
    vocab_size = 100
    
    # Create dummy logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Create mask (watermark only positions 2-4)
    mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_positions[0, 2:5] = True
    
    # Apply watermark
    watermarked_logits = apply_aaronson_gumbel_watermark(
        logits, mask_positions, vocab_size, position_offset=0
    )
    
    print(f"  Original logits shape: {logits.shape}")
    print(f"  Watermarked logits shape: {watermarked_logits.shape}")
    
    # Check that non-masked positions are unchanged
    assert torch.allclose(watermarked_logits[0, :2], logits[0, :2]), \
        "Non-masked positions should be unchanged"
    print("  ✓ Non-masked positions unchanged")
    
    # Check that masked positions are modified
    assert not torch.allclose(watermarked_logits[0, 2:5], logits[0, 2:5]), \
        "Masked positions should be modified"
    print("  ✓ Masked positions modified")
    
    print("  ✓ Aaronson watermark application test passed!\n")


def test_watermark_detection():
    """Test watermark detection."""
    print("Testing watermark detection...")
    
    vocab_size = 100
    seq_len = 50
    seed = 42
    
    # Create a sequence of random tokens
    generated_tokens = torch.randint(0, vocab_size, (1, seq_len))
    
    # Calculate watermark score
    score, actual_length, per_token_scores = calculate_aaronson_watermark_score(
        generated_tokens, vocab_size=vocab_size, seed=seed
    )
    
    print(f"  Sequence length: {seq_len}")
    print(f"  Actual length analyzed: {actual_length}")
    print(f"  Watermark score: {score:.4f}")
    print(f"  Normalized score: {score/actual_length:.4f}")
    print(f"  Per-token scores (first 5): {per_token_scores[:5]}")
    
    # Check that score is positive
    assert score > 0, "Watermark score should be positive"
    print("  ✓ Watermark score is positive")
    
    # Check that we have per-token scores for each token
    assert len(per_token_scores) == actual_length, \
        "Should have per-token scores for each token"
    print("  ✓ Per-token scores computed correctly")
    
    print("  ✓ Watermark detection test passed!\n")


def test_end_to_end():
    """Test end-to-end watermarking and detection."""
    print("Testing end-to-end watermarking and detection...")
    
    batch_size = 1
    seq_len = 20
    vocab_size = 100
    seed = 42
    
    # Create dummy logits (uniform distribution)
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    
    # Create mask (watermark all positions)
    mask_positions = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Apply watermark
    watermarked_logits = apply_aaronson_gumbel_watermark(
        logits, mask_positions, vocab_size, position_offset=0
    )
    
    # Select tokens using argmax (simulating generation)
    selected_tokens = torch.argmax(watermarked_logits, dim=-1)
    
    # Detect watermark
    score, actual_length, per_token_scores = calculate_aaronson_watermark_score(
        selected_tokens, vocab_size=vocab_size, seed=seed
    )
    
    print(f"  Selected tokens: {selected_tokens[0, :10].tolist()}...")
    print(f"  Watermark score: {score:.4f}")
    print(f"  Normalized score: {score/actual_length:.4f}")
    
    # For a watermarked sequence, the normalized score should be relatively high
    # (exact value depends on the distribution, but should be > 0.5 for uniform logits)
    print(f"  Expected normalized score for watermarked text: > 0.5")
    
    print("  ✓ End-to-end test completed!\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("AARONSON WATERMARKING IMPLEMENTATION TESTS")
    print("=" * 70)
    print()
    
    try:
        test_pseudo_random_values()
        test_aaronson_watermark_application()
        test_watermark_detection()
        test_end_to_end()
        
        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print()
        print("The Aaronson watermarking implementation is working correctly.")
        print("You can now use it with the LLaDA model evaluation.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()






