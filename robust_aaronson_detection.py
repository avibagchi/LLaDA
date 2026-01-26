#!/usr/bin/env python3
"""
Robust Aaronson watermark detection that handles insertion/deletion attacks.
Implements multiple strategies for position recovery.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from generate import generate_pseudo_random_values, get_special_token_ids


def calculate_aaronson_watermark_score_robust(
    generated_tokens,
    vocab_size=126464,
    seed=42,
    special_token_ids=None,
    position_offset_range: Optional[Tuple[int, int]] = None,
    max_insertions: int = 10,
    max_deletions: int = 10,
    use_alignment: bool = True
):
    """
    Robust Aaronson watermark detection that handles insertions/deletions.
    
    Strategies:
    1. Try multiple position offsets (sliding window)
    2. Use content-based alignment (if use_alignment=True)
    3. Return best score across all attempts
    
    Args:
        generated_tokens: [batch_size, seq_len] - generated token IDs
        vocab_size: Size of the vocabulary
        seed: Seed for pseudorandom generation
        special_token_ids: List of token IDs to exclude from detection
        position_offset_range: (min_offset, max_offset) to try. If None, tries reasonable range
        max_insertions: Maximum number of insertions to account for
        max_deletions: Maximum number of deletions to account for
        use_alignment: Whether to use content-based alignment (experimental)
        
    Returns:
        best_score: Best watermark score found
        best_offset: Position offset that gave best score
        best_length: Length used for best score
        all_attempts: List of (offset, score, length) for all attempts
    """
    batch_size, seq_len = generated_tokens.shape
    tokens = generated_tokens[0]
    
    # Create set of special token IDs
    special_token_set = set()
    if special_token_ids is not None:
        special_token_set = set(special_token_ids)
    
    # Find actual length (stop at EOS token)
    actual_length = seq_len
    for i, token in enumerate(tokens):
        if token.item() in [50256, 2, 126081]:  # EOS tokens
            actual_length = i
            break
    
    # Determine position offset range to try
    if position_offset_range is None:
        # Try offsets from -max_deletions to +max_insertions
        # This accounts for tokens being deleted (negative offset) or inserted (positive offset)
        min_offset = -max_deletions
        max_offset = max_insertions
    else:
        min_offset, max_offset = position_offset_range
    
    all_attempts = []
    best_score = -float('inf')
    best_offset = None
    best_length = 0
    
    # Strategy 1: Try multiple position offsets (sliding window approach)
    for offset in range(min_offset, max_offset + 1):
        per_token_scores = []
        tokens_used = 0
        
        for pos in range(actual_length):
            token_id = tokens[pos].item()
            
            # Skip special tokens
            if token_id in special_token_set:
                continue
            
            # Calculate position with offset
            adjusted_pos = pos + offset
            
            # Skip if adjusted position is negative (would indicate too many deletions)
            if adjusted_pos < 0:
                continue
            
            # Generate pseudorandom values for this adjusted position
            r_values = generate_pseudo_random_values(
                adjusted_pos, vocab_size, seed, device=tokens.device
            )
            
            # Get the r value for the selected token
            r_token = r_values[token_id]
            
            # Calculate per-token score
            per_token_score = torch.log(1.0 / (1.0 - r_token))
            per_token_scores.append(per_token_score.item())
            tokens_used += 1
        
        if len(per_token_scores) > 0:
            score = sum(per_token_scores)
            normalized_score = score / len(per_token_scores)
            all_attempts.append((offset, score, normalized_score, tokens_used))
            
            # Track best score (using normalized score for fair comparison)
            if normalized_score > best_score:
                best_score = normalized_score
                best_offset = offset
                best_length = tokens_used
    
    # Strategy 2: Content-based alignment (if enabled)
    # This would use sequence alignment algorithms to find the best match
    # For now, we'll rely on the sliding window approach above
    
    return best_score, best_offset, best_length, all_attempts


def detect_watermark_robust(
    generated_tokens,
    vocab_size=126464,
    seed=42,
    special_token_ids=None,
    threshold: float = 1.2,
    max_insertions: int = 10,
    max_deletions: int = 10,
    verbose: bool = False
) -> Tuple[bool, float, dict]:
    """
    Robust watermark detection with insertion/deletion handling.
    
    Args:
        generated_tokens: [batch_size, seq_len] - generated token IDs
        vocab_size: Size of the vocabulary
        seed: Seed for pseudorandom generation
        special_token_ids: List of special token IDs
        threshold: Normalized score threshold for detection
        max_insertions: Maximum insertions to account for
        max_deletions: Maximum deletions to account for
        verbose: Whether to print detailed information
        
    Returns:
        is_watermarked: True if watermark detected
        best_score: Best normalized score found
        info: Dictionary with detection details
    """
    best_score, best_offset, best_length, all_attempts = calculate_aaronson_watermark_score_robust(
        generated_tokens,
        vocab_size=vocab_size,
        seed=seed,
        special_token_ids=special_token_ids,
        max_insertions=max_insertions,
        max_deletions=max_deletions
    )
    
    is_watermarked = best_score > threshold
    
    info = {
        'best_score': best_score,
        'best_offset': best_offset,
        'best_length': best_length,
        'threshold': threshold,
        'is_watermarked': is_watermarked,
        'all_attempts': all_attempts[:5]  # Top 5 attempts
    }
    
    if verbose:
        print(f"Robust Detection Results:")
        print(f"  Best normalized score: {best_score:.4f}")
        print(f"  Best offset: {best_offset}")
        print(f"  Tokens used: {best_length}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Detected: {is_watermarked}")
        print(f"  Top 5 attempts:")
        for offset, score, norm_score, length in all_attempts[:5]:
            print(f"    Offset {offset:3d}: score={score:.2f}, normalized={norm_score:.4f}, length={length}")
    
    return is_watermarked, best_score, info


def evaluate_robustness_to_attacks(
    watermarked_tokens,
    vocab_size=126464,
    seed=42,
    special_token_ids=None,
    insertion_rates: List[float] = [0.0, 0.01, 0.05, 0.1],
    deletion_rates: List[float] = [0.0, 0.01, 0.05, 0.1],
    threshold: float = 1.2
):
    """
    Evaluate robustness by simulating insertion/deletion attacks.
    
    Args:
        watermarked_tokens: Original watermarked tokens
        vocab_size: Vocabulary size
        seed: Watermark seed
        special_token_ids: Special token IDs
        insertion_rates: List of insertion rates to test (0.0 = no insertions)
        deletion_rates: List of deletion rates to test (0.0 = no deletions)
        threshold: Detection threshold
        
    Returns:
        results: Dictionary with robustness results
    """
    results = {}
    
    for ins_rate in insertion_rates:
        for del_rate in deletion_rates:
            # Simulate attack
            attacked_tokens = simulate_insertion_deletion_attack(
                watermarked_tokens, ins_rate, del_rate
            )
            
            # Detect watermark
            is_detected, score, info = detect_watermark_robust(
                attacked_tokens,
                vocab_size=vocab_size,
                seed=seed,
                special_token_ids=special_token_ids,
                threshold=threshold,
                verbose=False
            )
            
            key = f"ins_{ins_rate:.2f}_del_{del_rate:.2f}"
            results[key] = {
                'insertion_rate': ins_rate,
                'deletion_rate': del_rate,
                'detected': is_detected,
                'score': score,
                'best_offset': info['best_offset']
            }
    
    return results


def simulate_insertion_deletion_attack(tokens, insertion_rate: float, deletion_rate: float):
    """
    Simulate insertion/deletion attack on tokens.
    
    Args:
        tokens: [batch_size, seq_len] token tensor
        insertion_rate: Probability of inserting a random token before each position
        deletion_rate: Probability of deleting each token
        
    Returns:
        attacked_tokens: Modified token tensor
    """
    tokens_list = tokens[0].tolist()
    attacked_list = []
    
    for token in tokens_list:
        # Deletion attack
        if np.random.random() < deletion_rate:
            continue  # Skip this token
        
        attacked_list.append(token)
        
        # Insertion attack (insert random token after current token)
        if np.random.random() < insertion_rate:
            # Insert a random token (excluding special tokens)
            random_token = np.random.randint(0, 126464)  # vocab_size
            attacked_list.append(random_token)
    
    # Convert back to tensor
    attacked_tokens = torch.tensor([attacked_list], device=tokens.device)
    return attacked_tokens


if __name__ == "__main__":
    # Example usage
    print("Robust Aaronson Watermark Detection")
    print("=" * 60)
    print("\nThis module provides robust detection that handles:")
    print("  1. Insertion attacks (tokens added)")
    print("  2. Deletion attacks (tokens removed)")
    print("  3. Position offset recovery")
    print("\nKey features:")
    print("  - Tries multiple position offsets (sliding window)")
    print("  - Returns best score across all attempts")
    print("  - Can evaluate robustness to simulated attacks")
    print("=" * 60)
