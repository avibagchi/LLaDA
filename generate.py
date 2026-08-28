import torch
import numpy as np
import torch.nn.functional as F
import math

from transformers import AutoTokenizer, AutoModel


def get_special_token_ids(tokenizer):
    """
    Get a list of special token IDs from the tokenizer to exclude from watermarking amplification.
    
    Args:
        tokenizer: The tokenizer object
        
    Returns:
        list: List of special token IDs
    """
    special_tokens = []
    
    # Add common special token IDs
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_tokens.append(tokenizer.bos_token_id)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_tokens.append(tokenizer.eos_token_id)
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        special_tokens.append(tokenizer.pad_token_id)
    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
        special_tokens.append(tokenizer.unk_token_id)
    if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
        special_tokens.append(tokenizer.sep_token_id)
    if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
        special_tokens.append(tokenizer.cls_token_id)
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        special_tokens.append(tokenizer.mask_token_id)
    
    # Add any additional special tokens from special_tokens_map
    if hasattr(tokenizer, 'special_tokens_map'):
        for token_name, token_value in tokenizer.special_tokens_map.items():
            if isinstance(token_value, str):
                token_id = tokenizer.convert_tokens_to_ids(token_value)
                if token_id != tokenizer.unk_token_id:  # Avoid duplicates
                    special_tokens.append(token_id)
    
    return list(set(special_tokens))  # Remove duplicates


def add_gumbel_noise(logits, temperature, chunk_size=50000):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Memory-optimized version: processes vocabulary in chunks to reduce peak memory usage.
    '''
    if temperature == 0:
        return logits
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # Convert to float32 to save memory
    logits_f32 = logits.to(torch.float32)
    
    # Process vocabulary in chunks to avoid OOM
    result_chunks = []
    for vocab_start in range(0, vocab_size, chunk_size):
        vocab_end = min(vocab_start + chunk_size, vocab_size)
        
        # Extract chunk
        logits_chunk = logits_f32[:, :, vocab_start:vocab_end]
        
        # Generate noise for this chunk only
        noise_chunk = torch.rand_like(logits_chunk, dtype=torch.float32)
        gumbel_noise_chunk = (- torch.log(noise_chunk + 1e-8)) ** temperature
        
        # Compute result for this chunk
        result_chunk = logits_chunk.exp() / gumbel_noise_chunk
        result_chunks.append(result_chunk)
        
        # Clear intermediate tensors to free memory
        del noise_chunk, gumbel_noise_chunk, logits_chunk
        torch.cuda.empty_cache()
    
    # Concatenate chunks
    result = torch.cat(result_chunks, dim=-1)
    
    # Convert back to original dtype
    return result.to(logits.dtype)


def generate_green_mask(sequence_length, vocab_size, gamma, device, n=5):
    """
    Generate green masks for watermarking.
    Creates a random partition of the vocabulary for each position, seeded by position index.
    
    Args:
        sequence_length: Length of the sequence
        vocab_size: Size of the vocabulary
        gamma: Fraction of tokens that are "green" (watermarked)
        device: Device to place tensors on
        n: Modulo parameter for seeding (default 5)
    
    Returns:
        green_mask: [sequence_length, vocab_size] binary mask
    """
    green_masks = []
    for pos in range(sequence_length):
        torch.manual_seed((pos) % n)  # Seed based on position
        # Create exactly gamma*|V| green tokens and (1-gamma)*|V| red tokens
        num_green = int(gamma * vocab_size)
        pos_green_mask = torch.zeros(vocab_size, device=device)
        pos_green_mask[:num_green] = 1
        pos_green_mask = pos_green_mask[torch.randperm(vocab_size, device=device)]
        green_masks.append(pos_green_mask)
    
    green_mask = torch.stack(green_masks, dim=0)  # [seq_len, vocab_size]
    return green_mask


def apply_watermark_to_logits(logits, green_mask, amplification, mask_positions, special_token_ids=None):
    """
    Apply watermark by biasing logits for green tokens.
    
    Args:
        logits: [batch_size, seq_len, vocab_size] - model logits
        green_mask: [seq_len, vocab_size] - binary mask for green tokens
        amplification: float - amplification factor for green tokens
        mask_positions: [batch_size, seq_len] - boolean mask for positions to watermark
        special_token_ids: list of token IDs to exclude from amplification (optional)
    """
    if amplification <= 0:
        return logits
    
    # Expand green_mask to match logits shape
    green_mask_expanded = green_mask.unsqueeze(0).expand_as(logits)  # [batch_size, seq_len, vocab_size]
    
    # Only apply watermark to masked positions
    mask_positions_expanded = mask_positions.unsqueeze(-1).expand_as(logits)  # [batch_size, seq_len, vocab_size]
    
    # Create special token mask to exclude special tokens from amplification
    special_token_mask = torch.ones_like(green_mask_expanded, dtype=torch.bool)
    if special_token_ids is not None:
        for token_id in special_token_ids:
            if token_id < logits.shape[-1]:  # Make sure token_id is within vocab size
                special_token_mask[:, :, token_id] = False
    
    # Apply amplification: add amplification value to green token logits
    # Only for masked positions and exclude special tokens
    amplification_addition = green_mask_expanded * amplification
    amplification_addition = amplification_addition * special_token_mask  # Zero out special tokens
    watermarked_logits = torch.where(
        mask_positions_expanded,
        logits + amplification_addition,
        logits
    )
    
    return watermarked_logits


def calculate_green_matches(generated_tokens, gamma=0.5, vocab_size=126464, n=5):
    """
    Calculate watermark detection metrics for generated text.
    
    Args:
        generated_tokens: [batch_size, seq_len] - generated token IDs
        gamma: Fraction of tokens that should be green
        vocab_size: Size of the vocabulary
        n: Modulo parameter for seeding (should match generation)
    
    Returns:
        max_match_percent: Maximum percentage of tokens in green list
        actual_length_used: Actual sequence length used (excluding EOS)
        max_num_matches: Maximum number of green token matches
        best_start: Best starting offset for detection
        match_arr: Array of match percentages for different offsets
    """
    sequence_length = generated_tokens.shape[1]
    max_match_percent = 0
    best_start = 0
    actual_length_used = 0
    max_num_matches = 0
    
    match_arr = []
    for start in range(0, n): 
        matches = 0
        actual_length = 0
        
        for pos in range(sequence_length):
            # Stop when we reach the EOS token (if any)
            # Check for common EOS tokens: 50256 (GPT-2), 2 (LLaMA), 126081 (LLaDA)
            if generated_tokens[0, pos] in [50256, 2, 126081]:  # EOS tokens
                break
                
            torch.manual_seed((pos + start) % n) 
            # Create exactly gamma*|V| green tokens and (1-gamma)*|V| red tokens
            num_green = int(gamma * vocab_size)
            pos_green_mask = torch.zeros(vocab_size, device=generated_tokens.device)
            pos_green_mask[:num_green] = 1
            pos_green_mask = pos_green_mask[torch.randperm(vocab_size, device=generated_tokens.device)]
            
            token = generated_tokens[0, pos]  
            
            if pos_green_mask[token] == 1:
                matches += 1
            
            actual_length += 1
        
        # Use actual_length instead of sequence_length for percentage calculation
        if actual_length > 0:
            percent_match = matches / actual_length
        else:
            percent_match = 0
            
        match_arr.append([start, percent_match])
        if percent_match > max_match_percent:
            max_match_percent = percent_match
            actual_length_used = actual_length
            max_num_matches = matches
            best_start = start
    
    return max_match_percent, actual_length_used, max_num_matches, best_start, match_arr


def generate_pseudo_random_values(position, vocab_size, seed=42, device="cpu"):
    """
    Generate pseudorandom values r_{t,i} for each token i at position t.
    Uses position-based seeding for deterministic generation with local Generator.
    
    Args:
        position: Position index t
        vocab_size: Size of the vocabulary
        seed: Base seed for pseudorandom generation
        device: Device to generate values on
        
    Returns:
        r_values: [vocab_size] tensor with values in (0, 1)
    """
    # Use a local Generator to avoid mutating global RNG state (Fix E)
    g = torch.Generator(device=device)
    g.manual_seed(seed + position)
    
    # Generate pseudorandom values in (0, 1)
    r_values = torch.rand(vocab_size, generator=g, device=device)
    
    # Ensure values are strictly in (0, 1) to avoid numerical issues
    r_values = torch.clamp(r_values, min=1e-8, max=1-1e-8)
    
    return r_values


def apply_aaronson_gumbel_watermark(logits, mask_positions, vocab_size, position_offset=0, seed=42, special_token_ids=None):
    """
    Apply Aaronson's watermarking scheme during generation.
    At each position t, selects the token that maximizes r_{t,i}^{1/p_{t,i}}.
    
    Args:
        logits: [batch_size, seq_len, vocab_size] - model logits
        mask_positions: [batch_size, seq_len] - boolean mask for positions to watermark
        vocab_size: Size of the vocabulary
        position_offset: Offset to add to position indices (for prompt length)
        seed: Seed for pseudorandom generation
        special_token_ids: List of token IDs to exclude from watermarking (optional)
        
    Returns:
        watermarked_choices: [batch_size, seq_len] - token choices with watermarking applied
                            (-1 for positions where watermarking is not applied)
        watermark_confidences: [batch_size, seq_len] - model probability of watermarked token
                               (0.0 for positions where watermarking is not applied)
    """
    batch_size, seq_len, _ = logits.shape
    
    # Initialize with -1 (indicating no watermark applied at this position)
    watermarked_choices = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=logits.device)
    watermark_confidences = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=logits.device)
    
    # Apply watermarking to each masked position
    for batch_idx in range(batch_size):
        for pos in range(seq_len):
            if mask_positions[batch_idx, pos]:
                # Compute model probabilities for this position (Fix A)
                model_probs = F.softmax(logits[batch_idx, pos], dim=-1)  # shape [V]
                
                # Generate pseudorandom values for this position (Fix E - using local generator)
                r_values = generate_pseudo_random_values(
                    position_offset + pos, vocab_size, seed, device=logits.device
                )
                
                # Compute Aaronson scores: log(r_{t,i}) / p_{t,i} (Fix A)
                # This is equivalent to maximizing r_{t,i}^{1/p_{t,i}}
                log_r = torch.log(r_values)
                watermark_scores = log_r / (model_probs + 1e-8)  # (1/p_i) * log r_i
                
                # Exclude special tokens by making them impossible to win (Fix C)
                if special_token_ids is not None:
                    for token_id in special_token_ids:
                        if token_id < vocab_size:
                            watermark_scores[token_id] = -float("inf")
                
                # Select the token with highest Aaronson score (Fix B)
                i_star = torch.argmax(watermark_scores).item()
                watermarked_choices[batch_idx, pos] = i_star
                
                # Store the model's confidence in this watermarked token
                watermark_confidences[batch_idx, pos] = model_probs[i_star].item()
    
    return watermarked_choices, watermark_confidences


def calculate_aaronson_watermark_score(generated_tokens, vocab_size=126464, seed=42, special_token_ids=None, position_offset=0):
    """
    Calculate the watermark detection score for generated text.
    Uses the formula: sum_{t=1}^{n} ln(1 / (1 - r_{t,i(t)}))
    
    Args:
        generated_tokens: [batch_size, seq_len] - generated token IDs
        vocab_size: Size of the vocabulary
        seed: Seed for pseudorandom generation (must match generation)
        special_token_ids: List of token IDs to exclude from detection (optional)
        position_offset: Position offset to match generation (e.g., prompt length). CRITICAL for correct detection!
        
    Returns:
        watermark_score: Total watermark score
        actual_length: Number of tokens analyzed (excluding special tokens)
        per_token_scores: [actual_length] array of per-token scores
    """
    batch_size, seq_len = generated_tokens.shape
    per_token_scores = []
    
    # Process only the first batch (assuming batch_size=1 for generation)
    tokens = generated_tokens[0]
    
    # Create set of special token IDs for efficient lookup
    special_token_set = set()
    if special_token_ids is not None:
        special_token_set = set(special_token_ids)
    
    # Find actual length (stop at EOS token if present)
    actual_length = seq_len
    for i, token in enumerate(tokens):
        # Check for common EOS tokens: 50256 (GPT-2), 2 (LLaMA), 126081 (LLaDA)
        if token.item() in [50256, 2, 126081]:
            actual_length = i
            break
    
    # Calculate per-token scores, skipping special tokens
    for pos in range(actual_length):
        token_id = tokens[pos].item()
        
        # Skip special tokens
        if token_id in special_token_set:
            continue
        
        # Generate pseudorandom values for this position (MUST match generation offset!)
        r_values = generate_pseudo_random_values(position_offset + pos, vocab_size, seed, device=tokens.device)
        
        # Get the r value for the selected token
        r_token = r_values[token_id]
        
        # Calculate per-token score: ln(1 / (1 - r_{t,i(t)}))
        # This grows as r approaches 1, so watermarked text has higher scores
        per_token_score = torch.log(1.0 / (1.0 - r_token))
        per_token_scores.append(per_token_score.item())
    
    # Convert to tensor for easier handling
    per_token_scores = torch.tensor(per_token_scores)
    
    # Calculate total watermark score
    watermark_score = per_token_scores.sum().item()
    
    # Update actual_length to reflect only non-special tokens
    actual_length = len(per_token_scores)
    
    return watermark_score, actual_length, per_token_scores


# ---------------------------------------------------------------------------
# DMark: Order-Agnostic Watermarking (Wu et al., 2025)
# ---------------------------------------------------------------------------
_DMARK_C1 = 2654435761   # Knuth multiplicative hash constant
_DMARK_C2 = 1013904223   # LCG additive constant
_DMARK_C3 = 1664525      # LCG multiplicative constant
_DMARK_MIX = 0x45d9f3b   # Wang hash multiplier


def _dmark_hash_score_scalar(secret_key, context_token, target_token):
    """
    Scalar pseudo-random score in [0,1) for a (context, target) pair.
    target is 'forward-green' iff score < gamma.
    Must match the vectorized version exactly.
    """
    h = (int(secret_key) * _DMARK_C1) ^ (int(context_token) * _DMARK_C2) ^ (int(target_token) * _DMARK_C3)
    h = h & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
    h = (h >> 16) ^ h
    return (h & 0x7FFFFFFF) / 2147483647.0


def _dmark_forward_green_mask(secret_key, context_token, vocab_size, gamma, device):
    """
    Forward green mask G_i(context): token v is green iff score(key, context, v) < gamma.
    Returns [vocab_size] bool tensor.
    """
    key = int(secret_key) & 0xFFFFFFFF
    ctx = int(context_token) & 0xFFFFFFFF
    base = (key * _DMARK_C1 ^ ctx * _DMARK_C2) & 0xFFFFFFFF
    all_v = torch.arange(vocab_size, dtype=torch.int64, device=device)
    h = (base ^ (all_v * _DMARK_C3)) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
    h = (h >> 16) ^ h
    scores = (h & 0x7FFFFFFF).float() / 2147483647.0
    return scores < gamma


def _dmark_backward_green_mask(secret_key, target_token, vocab_size, gamma, device):
    """
    Backward green mask G'_i(target): candidate v is backward-green iff
    target ∈ forward_green(key, v), i.e. score(key, v, target) < gamma.
    Returns [vocab_size] bool tensor.
    """
    key = int(secret_key) & 0xFFFFFFFF
    tgt = int(target_token) & 0xFFFFFFFF
    all_v = torch.arange(vocab_size, dtype=torch.int64, device=device)
    base_v = (key * _DMARK_C1 ^ all_v * _DMARK_C2) & 0xFFFFFFFF
    h = (base_v ^ (tgt * _DMARK_C3)) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
    h = (h >> 16) ^ h
    scores = (h & 0x7FFFFFFF).float() / 2147483647.0
    return scores < gamma


def apply_dmark_watermark(logits, x, block_mask, secret_key, gamma, delta,
                          vocab_size, variant, mask_id):
    """
    Apply DMark logit bias for all positions flagged in block_mask.

    variant: 'predictive' | 'bidirectional' | 'predictive_bidirectional'
    block_mask: [1, seq_len] bool — True at positions to watermark (masked & in current block)
    """
    if delta <= 0:
        return logits

    modified = logits.clone()
    seq_len = logits.shape[1]
    device = logits.device

    fwd_cache = {}   # context_token  -> [vocab_size] bool
    bwd_cache = {}   # target_token   -> [vocab_size] bool

    use_fwd = variant in ('predictive', 'bidirectional', 'predictive_bidirectional')
    use_bwd = variant in ('bidirectional', 'predictive_bidirectional')
    predict_missing = variant in ('predictive', 'predictive_bidirectional')

    for pos in range(seq_len):
        if not block_mask[0, pos]:
            continue

        fwd_ctx = None
        bwd_ctx = None

        if use_fwd and pos > 0:
            prev = x[0, pos - 1].item()
            if prev != mask_id:
                fwd_ctx = prev
            elif predict_missing:
                fwd_ctx = torch.argmax(logits[0, pos - 1]).item()

        if use_bwd and pos < seq_len - 1:
            nxt = x[0, pos + 1].item()
            if nxt != mask_id:
                bwd_ctx = nxt
            elif variant == 'predictive_bidirectional':
                bwd_ctx = torch.argmax(logits[0, pos + 1]).item()

        if fwd_ctx is None and bwd_ctx is None:
            continue

        bias = torch.zeros(vocab_size, device=device)

        if fwd_ctx is not None:
            if fwd_ctx not in fwd_cache:
                fwd_cache[fwd_ctx] = _dmark_forward_green_mask(
                    secret_key, fwd_ctx, vocab_size, gamma, device)
            bias = bias + delta * fwd_cache[fwd_ctx].float()

        if bwd_ctx is not None:
            if bwd_ctx not in bwd_cache:
                bwd_cache[bwd_ctx] = _dmark_backward_green_mask(
                    secret_key, bwd_ctx, vocab_size, gamma, device)
            bias = bias + delta * bwd_cache[bwd_ctx].float()

        modified[0, pos] = modified[0, pos] + bias

    return modified


def calculate_dmark_score(generated_tokens, secret_key=42, gamma=0.5, vocab_size=126464,
                          variant='predictive_bidirectional', mask_id=126336):
    """
    Calculate DMark watermark z-score for generated text.

    Returns: (z_score, valid_positions)
    Under the null (no watermark), z ~ N(0,1) and threshold z >= 4 gives ~0.003% FPR.
    """
    tokens = generated_tokens[0]
    actual_end = len(tokens)
    for i, t in enumerate(tokens):
        if t.item() in [50256, 2, 126081]:
            actual_end = i
            break

    SPECIAL = {50256, 2, 126081, mask_id}

    use_fwd = variant in ('predictive', 'bidirectional', 'predictive_bidirectional')
    use_bwd = variant in ('bidirectional', 'predictive_bidirectional')

    count_green = 0
    valid_positions = 0

    for i in range(actual_end):
        tok = tokens[i].item()
        if tok in SPECIAL:
            continue

        fwd_ctx = None
        bwd_ctx = None

        if use_fwd and i > 0 and tokens[i - 1].item() not in SPECIAL:
            fwd_ctx = tokens[i - 1].item()

        if use_bwd and i < actual_end - 1 and tokens[i + 1].item() not in SPECIAL:
            bwd_ctx = tokens[i + 1].item()

        if fwd_ctx is None and bwd_ctx is None:
            continue

        valid_positions += 1
        is_green = False

        if fwd_ctx is not None:
            if _dmark_hash_score_scalar(secret_key, fwd_ctx, tok) < gamma:
                is_green = True

        if bwd_ctx is not None:
            # tok is backward-green iff score(key, tok, bwd_ctx) < gamma
            if _dmark_hash_score_scalar(secret_key, tok, bwd_ctx) < gamma:
                is_green = True

        if is_green:
            count_green += 1

    if valid_positions == 0:
        return 0.0, 0

    # Expected green fraction under null:
    # forward-only: gamma; bidirectional (OR): 2*gamma - gamma^2
    if use_bwd:
        gamma_eff = 2.0 * gamma - gamma * gamma
    else:
        gamma_eff = gamma

    expected = gamma_eff * valid_positions
    std = math.sqrt(gamma_eff * (1.0 - gamma_eff) * valid_positions + 1e-10)
    z_score = (count_green - expected) / std
    return z_score, valid_positions


# ---------------------------------------------------------------------------
# CDMArk: CDMA-style holographic encoding (adapted to zero-bit, arXiv:2412.02217)
# ---------------------------------------------------------------------------
_CDMARK_SIGNAL_CACHE: dict = {}


def get_cdmark_signal_vectors(vocab_size: int, m: int, seed: int, device):
    """Build CDMArk signal matrix V ∈ R^{vocab_size × m} via QR orthogonalization."""
    cache_key = (vocab_size, m, seed, str(device))
    if cache_key in _CDMARK_SIGNAL_CACHE:
        return _CDMARK_SIGNAL_CACHE[cache_key]
    rng = torch.Generator()
    rng.manual_seed(seed)
    G = torch.randn(vocab_size, m, generator=rng, dtype=torch.float32)
    G_prime = G - G.mean(dim=0, keepdim=True)
    if m == 1:
        norm = G_prime[:, 0].norm()
        Q = G_prime / (norm + 1e-10)
    else:
        Q, _ = torch.linalg.qr(G_prime)
    V = Q * (vocab_size ** 0.5)
    V = V.to(device)
    _CDMARK_SIGNAL_CACHE[cache_key] = V
    return V


def apply_cdmark_watermark(logits, block_mask, secret_key, delta, vocab_size, m=1):
    """
    CDMArk generation: logit bias = delta * (V[w] @ s) for each token w.
    Zero-bit (m=1): s = [1], so bias = delta * V[:,0].
    """
    if delta <= 0:
        return logits
    device = logits.device
    V = get_cdmark_signal_vectors(vocab_size, m, int(secret_key), device)
    s = V.new_zeros(m)
    s[0] = 1.0
    bias = ((V @ s) * delta).to(logits.dtype)   # [vocab_size], match bfloat16
    modified = logits.clone()
    pos_mask = block_mask[0]         # [seq_len]
    modified[0, pos_mask] = modified[0, pos_mask] + bias
    return modified


def calculate_cdmark_score(generated_tokens, secret_key=42, vocab_size=126464, m=1, mask_id=126336):
    """
    CDMArk detection (zero-bit).  z = sum(V[x_i,0]) / sqrt(N) ~ N(0,1) under null.
    Returns (z_score, valid_positions).
    """
    tokens = generated_tokens[0]
    SPECIAL = {50256, 2, 126081, mask_id}
    device = tokens.device
    V = get_cdmark_signal_vectors(vocab_size, m, int(secret_key), device)
    signal = 0.0
    count = 0
    for tok in tokens:
        t = tok.item()
        if t in SPECIAL:
            break
        if t < vocab_size:
            signal += V[t, 0].item()
            count += 1
    if count == 0:
        return 0.0, 0
    return float(signal / (count ** 0.5 + 1e-10)), count


# ---------------------------------------------------------------------------
# dgMARK: Decoding-Guided Watermarking (Yoo et al., 2024, arXiv:2411.xxxxx)
# ---------------------------------------------------------------------------

def _dgmark_parity_match(x0, secret_key, device, prompt_offset=0):
    """
    Returns [batch, seq_len] bool: True where predicted token matches position parity.
    G_i = {v : hash(secret_key, v) % 2 == i % 2}
    prompt_offset: subtract so parity is relative to generated sequence start,
    matching detection which uses positions 0..n-1 within the generated tokens.
    """
    v = x0.long()
    _, seq_len = v.shape
    key_c1 = int(int(secret_key) * _DMARK_C1) & 0xFFFFFFFF
    h = (key_c1 ^ (v * _DMARK_C3)) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
    h = (h >> 16) ^ h
    token_parity = h % 2
    pos = torch.arange(seq_len, device=device).unsqueeze(0) - prompt_offset
    return token_parity == (pos % 2)


def calculate_dgmark_score(generated_tokens, secret_key=42, vocab_size=126464, mask_id=126336):
    """
    dgMARK detection: z = (G - n/2) / sqrt(n/4).
    G = count of positions where y_i ∈ G_i (parity match).
    Returns (z_score, valid_positions).
    """
    tokens = generated_tokens[0]
    SPECIAL = {50256, 2, 126081, mask_id}
    key_c1 = int(int(secret_key) * _DMARK_C1) & 0xFFFFFFFF
    count = 0
    matches = 0
    for i, tok in enumerate(tokens):
        t = tok.item()
        if t in SPECIAL:
            break
        count += 1
        h = (key_c1 ^ (int(t) * _DMARK_C3)) & 0xFFFFFFFF
        h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
        h = (((h >> 16) ^ h) * _DMARK_MIX) & 0xFFFFFFFF
        h = (h >> 16) ^ h
        if (h % 2) == (i % 2):
            matches += 1
    if count == 0:
        return 0.0, 0
    expected = count / 2.0
    std = math.sqrt(count / 4.0 + 1e-10)
    return float((matches - expected) / std), count


# ---------------------------------------------------------------------------
# LR-DWM: Left-Right Diffusion Watermarking (Hou et al., 2025)
# ---------------------------------------------------------------------------
_LRDWM_KEY_MIX = 0x4B5D3A2E   # separates k_L and k_R derived from the same seed


def apply_lrdwm_watermark(logits, x, block_mask, secret_key, gamma, delta, vocab_size, mask_id):
    """
    LR-DWM generation: bias logits using left green list (k_L) and right green list (k_R).
    Uses only ACTUALLY revealed neighbors — no predictions, unlike DMark.
    k_L = secret_key,  k_R = secret_key ^ _LRDWM_KEY_MIX.
    """
    if delta <= 0:
        return logits
    k_L = int(secret_key)
    k_R = int(secret_key) ^ _LRDWM_KEY_MIX
    modified = logits.clone()
    seq_len = logits.shape[1]
    device = logits.device
    fwd_cache: dict = {}
    bwd_cache: dict = {}
    for pos in range(seq_len):
        if not block_mask[0, pos]:
            continue
        fwd_ctx = None
        bwd_ctx = None
        if pos > 0:
            prev = x[0, pos - 1].item()
            if prev != mask_id:
                fwd_ctx = prev
        if pos < seq_len - 1:
            nxt = x[0, pos + 1].item()
            if nxt != mask_id:
                bwd_ctx = nxt
        if fwd_ctx is None and bwd_ctx is None:
            continue
        bias = torch.zeros(vocab_size, device=device)
        if fwd_ctx is not None:
            ck = (k_L, fwd_ctx)
            if ck not in fwd_cache:
                fwd_cache[ck] = _dmark_forward_green_mask(k_L, fwd_ctx, vocab_size, gamma, device)
            bias = bias + delta * fwd_cache[ck].float()
        if bwd_ctx is not None:
            ck = (k_R, bwd_ctx)
            if ck not in bwd_cache:
                bwd_cache[ck] = _dmark_backward_green_mask(k_R, bwd_ctx, vocab_size, gamma, device)
            bias = bias + delta * bwd_cache[ck].float()
        modified[0, pos] = modified[0, pos] + bias
    return modified


def calculate_lrdwm_score(generated_tokens, secret_key=42, gamma=0.5, vocab_size=126464, mask_id=126336):
    """
    LR-DWM detection: s_i = m_L + m_R - 1 for interior positions with both neighbors.
    Under the null, E[s_i] = 2*gamma - 1 and Var[s_i] = 2*gamma*(1-gamma);
    Z = (sum(s_i) - T*E[s_i]) / sqrt(T*Var[s_i]).  Returns (z_score, valid_positions).
    """
    tokens = generated_tokens[0]
    k_L = int(secret_key)
    k_R = int(secret_key) ^ _LRDWM_KEY_MIX
    SPECIAL = {50256, 2, 126081, mask_id}
    actual_end = len(tokens)
    for i, t in enumerate(tokens):
        if t.item() in SPECIAL:
            actual_end = i
            break
    total_score = 0.0
    count = 0
    for i in range(1, actual_end - 1):
        tok = tokens[i].item()
        if tok in SPECIAL:
            continue
        left = tokens[i - 1].item()
        right = tokens[i + 1].item()
        if left in SPECIAL or right in SPECIAL:
            continue
        m_L = 1 if _dmark_hash_score_scalar(k_L, left, tok) < gamma else 0
        m_R = 1 if _dmark_hash_score_scalar(k_R, tok, right) < gamma else 0
        total_score += m_L + m_R - 1
        count += 1
    if count == 0:
        return 0.0, 0
    mean = (2.0 * gamma - 1.0) * count
    var = 2.0 * gamma * (1.0 - gamma)
    z_score = (total_score - mean) / (math.sqrt(var * count) + 1e-10)
    return float(z_score), count


# ---------------------------------------------------------------------------
# UMR: Unbiased Multi-bit Watermarking (Zhang et al., 2025, zero-bit adapted)
# ---------------------------------------------------------------------------

def apply_umr_watermark(logits, x, block_mask, secret_key, gamma, delta, vocab_size, mask_id):
    """
    UMR generation (zero-bit adaptation).
    Unbiased multiplicative modulation: E[P_w] = P (no quality loss in expectation).
    Stability constraint: only watermark if left neighbor is revealed.
    P_w(v) = P(v)*(1+delta) if v ∈ green_list, else P(v)*(1 - delta*tau/(1-tau)).

    Returns (modified_logits, not_watermarked) where not_watermarked is a bool mask
    of block positions that were skipped due to unstable context (for R-remasking).
    """
    seq_len = logits.shape[1]
    device = logits.device
    not_watermarked = torch.zeros(logits.shape[0], seq_len, dtype=torch.bool, device=device)
    if delta <= 0:
        return logits, not_watermarked
    modified = logits.clone()
    fwd_cache: dict = {}
    for pos in range(seq_len):
        if not block_mask[0, pos]:
            continue
        if pos == 0:
            continue
        prev_actual = x[0, pos - 1].item()
        if prev_actual == mask_id:
            # Reference UMR (Yang et al. 2026): predict left neighbor from the
            # watermark-modified logits (= same logits x0 uses), not the originals.
            # Using original logits mispredicts 90% of the time when gamma=0.1
            # because watermarking pushes argmax from red→green; using modified
            # logits gives exactly x0[pos-1], matching what detection will see.
            prev = int(torch.argmax(modified[0, pos - 1]).item())
            if prev == mask_id:
                not_watermarked[0, pos] = True
                continue
        else:
            prev = prev_actual
        if prev not in fwd_cache:
            fwd_cache[prev] = _dmark_forward_green_mask(secret_key, prev, vocab_size, gamma, device)
        green_mask = fwd_cache[prev]
        probs = F.softmax(modified[0, pos].float(), dim=-1)
        tau = probs[green_mask].sum().item()
        if tau <= 0.0 or tau >= 1.0:
            continue
        delta_prime = float(delta) * tau / (1.0 - tau)
        if delta_prime >= 1.0:
            delta_prime = 1.0 - 1e-6
        boost = torch.ones(vocab_size, dtype=torch.float32, device=device)
        boost[green_mask] = 1.0 + float(delta)
        boost[~green_mask] = 1.0 - delta_prime
        new_probs = probs * boost
        new_probs = new_probs / (new_probs.sum() + 1e-10)
        modified[0, pos] = torch.log(new_probs + 1e-10).to(logits.dtype)
    return modified, not_watermarked


def calculate_umr_score(generated_tokens, secret_key=42, gamma=0.5, vocab_size=126464, mask_id=126336):
    """
    UMR detection (zero-bit): forward KGW z-score over stable (left-neighbor-revealed) positions.
    Returns (z_score, valid_positions).
    """
    tokens = generated_tokens[0]
    SPECIAL = {50256, 2, 126081, mask_id}
    actual_end = len(tokens)
    for i, t in enumerate(tokens):
        if t.item() in SPECIAL:
            actual_end = i
            break
    count_green = 0
    valid_positions = 0
    for i in range(1, actual_end):
        tok = tokens[i].item()
        if tok in SPECIAL:
            continue
        prev = tokens[i - 1].item()
        if prev in SPECIAL:
            continue
        valid_positions += 1
        if _dmark_hash_score_scalar(int(secret_key), prev, tok) < gamma:
            count_green += 1
    if valid_positions == 0:
        return 0.0, 0
    expected = gamma * valid_positions
    std = math.sqrt(gamma * (1.0 - gamma) * valid_positions + 1e-10)
    return float((count_green - expected) / std), valid_positions


# ---------------------------------------------------------------------------


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def _should_watermark(i, watermark_steps):
    """
    Helper function to determine if watermarking should be applied at step i.
    Uses consistent 1-indexed step numbers for user-friendliness.
    
    Args:
        i: 0-indexed step counter (internal loop variable)
        watermark_steps: None (watermark all steps), int (watermark steps 1 to N), 
                        or list of step numbers (1-indexed, e.g., [1, 2, 5, 10])
    
    Returns:
        bool: True if watermarking should be applied at this step
    """
    if watermark_steps is None:
        return True
    if isinstance(watermark_steps, int):
        return (i + 1) <= watermark_steps  # 1-indexed: step 1 is i=0
    # For lists, convert 1-indexed user steps to 0-indexed internal steps
    wanted = {s - 1 for s in watermark_steps}  # e.g., [1, 2, 5] -> {0, 1, 4}
    return i in wanted


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,
             gamma=0.5, amplification=0.0, vocab_size=126464, watermark_steps=None,
             special_token_ids=None, watermark_type='green_list', aaronson_seed=42,
             aaronson_remasking_strategy='original', aaronson_tau_wm=0.2, aaronson_tau_orig=0.01, aaronson_lambda=0.7,
             gloaguen_watermark=None,
             dmark_variant='predictive_bidirectional', dmark_seed=42,
             cdmark_seed=42, cdmark_m=1,
             dgmark_seed=42,
             lrdwm_seed=42,
             umr_seed=42):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        gamma: Fraction of tokens that are "green" (watermarked) - only for green_list watermarking.
        amplification: Amplification factor for green tokens (0 = no watermarking) - only for green_list watermarking.
        vocab_size: Size of the vocabulary.
        watermark_steps: Maximum step to watermark at (int), list of specific steps (1-indexed), or None for all steps.
                        If int, watermarks at steps 1 to watermark_steps (e.g., 100 means steps 1-100).
                        If list, watermarks at the specified 1-indexed steps (e.g., [1, 2, 5, 10]).
                        If None, watermarks at all steps.
        special_token_ids: List of token IDs to exclude from amplification (optional) - only for green_list watermarking.
        watermark_type: Type of watermarking to use ('green_list', 'aaronson', or 'gloaguen').
        gloaguen_watermark: Optional OurWatermark instance (Gloaguen et al. / diffusion-lm-watermark) when watermark_type is 'gloaguen'.
        aaronson_seed: Seed for pseudorandom generation in Aaronson watermarking.
        aaronson_remasking_strategy: Remasking strategy for Aaronson watermarking.
                                    'original': Use only original model confidence (default, best quality)
                                    'dual_gate': Require both wm_conf >= tau_wm AND orig_conf >= tau_orig
                                    'blend': Combine confidences as lambda*wm_conf + (1-lambda)*orig_conf
                                    'hard_favor': Give watermarked tokens high sentinel confidence
        aaronson_tau_wm: Watermark confidence threshold for dual_gate strategy (default: 0.2)
        aaronson_tau_orig: Original confidence threshold for dual_gate strategy (default: 0.01)
        aaronson_lambda: Blending weight for blend strategy (default: 0.7, range: 0-1)
                        Higher = stronger detectability, lower = better quality
    '''
    # breakpoint()
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # Initialize watermarking based on type
    green_mask = None
    if watermark_type == 'green_list' and amplification > 0:
        # Create green mask for the full sequence (prompt + generated)
        full_seq_length = prompt.shape[1] + gen_length
        green_mask = generate_green_mask(full_seq_length, vocab_size, gamma, model.device)

    # UMR Regret-based Remasking (R-remasking): tracks committed positions that
    # were generated without watermark (unstable context) and re-masks them once
    # their context stabilizes (Yang et al., ACL 2026, Algorithm 4-5).
    umr_candidates = None  # R-remasking disabled: cascading remasks confuse the model and hurt z-scores

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            # UMR R-remasking: before this step, force-remask candidates whose left
            # neighbor is now committed (stable context), giving them a second
            # watermarking chance this step. Only fires during watermark steps — R-remasking
            # during non-watermark steps would re-commit without watermark, destroying the
            # signal placed in earlier watermark steps.
            if (umr_candidates is not None and umr_candidates.any()
                    and _should_watermark(i, watermark_steps)):
                left_ok = torch.zeros_like(x, dtype=torch.bool)
                right_ok = torch.zeros_like(x, dtype=torch.bool)
                left_ok[:, 1:] = (x[:, :-1] != mask_id)
                right_ok[:, :-1] = (x[:, 1:] != mask_id)
                to_remask = umr_candidates & left_ok & (x != mask_id)
                if to_remask.any():
                    # Cascade: right-neighbor of a remasked position used that
                    # position's old token as its green-list key during generation.
                    # That key is about to change, so the right-neighbor's watermark
                    # is invalid — remask it so it can be re-watermarked with the
                    # new stable left context.
                    cascade = torch.zeros_like(x, dtype=torch.bool)
                    cascade[:, 1:] = to_remask[:, :-1]
                    cascade = cascade & (x != mask_id) & ~to_remask
                    x[to_remask] = mask_id
                    umr_candidates[to_remask] = False
                    if cascade.any():
                        x[cascade] = mask_id
                        # cascade positions: left context (the remasked position) is now
                        # masked, so apply_umr_watermark will mark them not_watermarked
                        # this step → they join umr_candidates via the normal path,
                        # then get re-watermarked once their left context stabilizes.

            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Separate sampling and remasking logits (like diffusion-lm-watermark)
            # Keep original logits for remasking confidence calculation
            remasking_logits = logits.clone()

            # Apply watermark to create sampling logits
            aaronson_choices = None  # Will hold watermarked token choices for Aaronson
            aaronson_wm_confidences = None  # Will hold watermark confidences for Aaronson
            umr_not_watermarked = None  # Positions skipped by UMR due to unstable context
            sampling_logits = logits.clone()  # Start with original logits
            
            if watermark_type == 'green_list' and amplification > 0 and green_mask is not None:
                # Check if we should apply watermarking at this step (using consistent 1-indexed logic)
                if _should_watermark(i, watermark_steps):
                    # Only apply to the current block being generated
                    current_block_start = prompt.shape[1] + num_block * block_length
                    current_block_end = prompt.shape[1] + (num_block + 1) * block_length
                    
                    # Create a full-size green mask for the entire sequence
                    full_green_mask = torch.zeros(logits.shape[1], logits.shape[2], device=logits.device)
                    # Only apply green mask to the generated portion (not the prompt)
                    full_green_mask[prompt.shape[1]:] = green_mask[prompt.shape[1]:]
                    
                    # Create mask for current block positions
                    current_block_mask = torch.zeros_like(x, dtype=torch.bool)
                    current_block_mask[:, current_block_start:current_block_end] = True
                    current_block_mask = current_block_mask & mask_index
                    
                    # Apply watermark to sampling logits only
                    sampling_logits = apply_watermark_to_logits(sampling_logits, full_green_mask, amplification, current_block_mask, special_token_ids)

            elif watermark_type == 'gloaguen' and gloaguen_watermark is not None:
                # Gloaguen et al. optimal Diffusion-KGW (OurWatermark in diffusion-lm-watermark)
                if _should_watermark(i, watermark_steps):
                    gloaguen_watermark.set_temperature(temperature)
                    gloaguen_watermark.set_mask_token(mask_id)
                    sampling_logits_wm, _ = gloaguen_watermark.watermark_logits(x, sampling_logits)
                    sampling_logits = sampling_logits_wm
            
            elif watermark_type == 'aaronson':
                # Aaronson watermarking should ONLY be applied at the final step(s) when distribution is close to final.
                # The theoretical guarantee of distortion-freeness only holds when sampling from the final distribution.
                # 
                # Logic: Only watermark steps >= watermark_steps (i.e., the last N steps)
                # - If watermark_steps is None: default to only the last step (steps)
                # - If watermark_steps is int: watermark steps >= watermark_steps (e.g., 295 means steps 295-300)
                # - If watermark_steps is list: watermark only those specific steps
                should_apply = False
                if watermark_steps is None:
                    # Default: only apply at the very last step to preserve quality
                    should_apply = (i == steps - 1)
                elif isinstance(watermark_steps, int):
                    # Apply at steps >= watermark_steps (only the last N steps)
                    # Example: if watermark_steps=295 and steps=300, apply at steps 295, 296, 297, 298, 299, 300
                    # WARNING: Using low watermark_steps (e.g., 200) will degrade quality because the distribution
                    # at intermediate steps is still noisy and far from the final distribution.
                    should_apply = (i + 1) <= watermark_steps
                else:
                    # For lists, check if current step is in the list
                    wanted = {s - 1 for s in watermark_steps}
                    should_apply = i in wanted
                
                if should_apply:
                    # Only apply to the current block being generated
                    current_block_start = prompt.shape[1] + num_block * block_length
                    current_block_end = prompt.shape[1] + (num_block + 1) * block_length
                    
                    # Create mask for current block positions
                    current_block_mask = torch.zeros_like(x, dtype=torch.bool)
                    current_block_mask[:, current_block_start:current_block_end] = True
                    current_block_mask = current_block_mask & mask_index
                    
                    # Apply Aaronson watermarking - returns token choices and confidences
                    # NOTE: position_offset=0 because pos in apply_aaronson_gumbel_watermark is already absolute
                    aaronson_choices, aaronson_wm_confidences = apply_aaronson_gumbel_watermark(
                        sampling_logits, current_block_mask, vocab_size, 
                        position_offset=0, seed=aaronson_seed,
                        special_token_ids=special_token_ids
                    )

            elif watermark_type == 'dmark':
                if _should_watermark(i, watermark_steps):
                    current_block_start = prompt.shape[1] + num_block * block_length
                    current_block_end = prompt.shape[1] + (num_block + 1) * block_length
                    current_block_mask = torch.zeros_like(x, dtype=torch.bool)
                    current_block_mask[:, current_block_start:current_block_end] = True
                    current_block_mask = current_block_mask & mask_index
                    sampling_logits = apply_dmark_watermark(
                        sampling_logits, x, current_block_mask,
                        secret_key=dmark_seed, gamma=gamma, delta=amplification,
                        vocab_size=vocab_size, variant=dmark_variant, mask_id=mask_id
                    )

            elif watermark_type == 'cdmark':
                if _should_watermark(i, watermark_steps):
                    current_block_start = prompt.shape[1] + num_block * block_length
                    current_block_end = prompt.shape[1] + (num_block + 1) * block_length
                    current_block_mask = torch.zeros_like(x, dtype=torch.bool)
                    current_block_mask[:, current_block_start:current_block_end] = True
                    current_block_mask = current_block_mask & mask_index
                    sampling_logits = apply_cdmark_watermark(
                        sampling_logits, current_block_mask,
                        secret_key=cdmark_seed, delta=amplification,
                        vocab_size=vocab_size, m=cdmark_m
                    )

            elif watermark_type == 'lrdwm':
                if _should_watermark(i, watermark_steps):
                    current_block_start = prompt.shape[1] + num_block * block_length
                    current_block_end = prompt.shape[1] + (num_block + 1) * block_length
                    current_block_mask = torch.zeros_like(x, dtype=torch.bool)
                    current_block_mask[:, current_block_start:current_block_end] = True
                    current_block_mask = current_block_mask & mask_index
                    sampling_logits = apply_lrdwm_watermark(
                        sampling_logits, x, current_block_mask,
                        secret_key=lrdwm_seed, gamma=gamma, delta=amplification,
                        vocab_size=vocab_size, mask_id=mask_id
                    )

            elif watermark_type == 'umr':
                pass  # watermark applied per-position after top-k selection, see below

            # Use sampling_logits for Gumbel noise and argmax (for sampling)
            logits_with_noise = add_gumbel_noise(sampling_logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            # Override with Aaronson watermarked choices where applicable
            if aaronson_choices is not None:
                watermark_mask = (aaronson_choices != -1)  # Positions where watermarking was applied
                x0 = torch.where(watermark_mask, aaronson_choices, x0)

            # Use remasking_logits (original, unwatermarked) for confidence calculation
            if remasking == 'low_confidence':
                p = F.softmax(remasking_logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Apply Aaronson remasking strategy if applicable
            if aaronson_choices is not None and aaronson_wm_confidences is not None:
                watermark_mask = (aaronson_choices != -1)
                
                if aaronson_remasking_strategy == 'dual_gate':
                    # Dual-gate: Require both wm_conf >= tau_wm AND orig_conf >= tau_orig
                    # Set confidence to -inf if either threshold is not met
                    wm_conf_ok = aaronson_wm_confidences >= aaronson_tau_wm
                    orig_conf_ok = x0_p >= aaronson_tau_orig
                    dual_gate_pass = wm_conf_ok & orig_conf_ok
                    # For watermarked positions, use original confidence if dual-gate passes, else -inf
                    x0_p = torch.where(watermark_mask & ~dual_gate_pass, 
                                      torch.full_like(x0_p, -np.inf), x0_p)
                
                elif aaronson_remasking_strategy == 'blend':
                    # Blend: conf = lambda * wm_conf + (1-lambda) * orig_conf
                    blended_conf = aaronson_lambda * aaronson_wm_confidences + (1 - aaronson_lambda) * x0_p
                    x0_p = torch.where(watermark_mask, blended_conf, x0_p)
                
                elif aaronson_remasking_strategy == 'hard_favor':
                    # Hard favor: Give watermarked tokens high sentinel confidence (0.99)
                    # breakpoint()
                    
                    x0_p = torch.where(watermark_mask, torch.full_like(x0_p, 0.99), x0_p)
                    
                
                # else: 'original' strategy - keep x0_p as is (default behavior)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # dgMARK: steer unmasking order by boosting parity-matching positions
            if watermark_type == 'dgmark' and _should_watermark(i, watermark_steps):
                dgmark_block_start = prompt.shape[1] + num_block * block_length
                dgmark_block_end = prompt.shape[1] + (num_block + 1) * block_length
                dgmark_block_mask = torch.zeros_like(x, dtype=torch.bool)
                dgmark_block_mask[:, dgmark_block_start:dgmark_block_end] = True
                dgmark_block_mask = dgmark_block_mask & mask_index
                parity_match = _dgmark_parity_match(x0, dgmark_seed, x0.device, prompt_offset=prompt.shape[1])
                boost_mask = parity_match & dgmark_block_mask
                confidence = torch.where(boost_mask, confidence + 1e4, confidence)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            # UMR: apply watermark per-position AFTER top-k selection (reference approach).
            # Confidence uses original logits, so high-probability tokens are committed;
            # watermark overwrites x0 only for committed positions.
            if watermark_type == 'umr' and _should_watermark(i, watermark_steps):
                umr_not_watermarked = torch.zeros_like(x, dtype=torch.bool)
                fwd_umr_cache: dict = {}
                for pos in range(x.shape[1]):
                    if not transfer_index[0, pos]:
                        continue
                    if pos == 0:
                        continue
                    prev_actual = x[0, pos - 1].item()
                    if prev_actual == mask_id:
                        # Left neighbor not yet committed — skip watermark and add to
                        # R-remasking candidates. Once the left token is committed this
                        # position will be remasked and re-watermarked with the correct key.
                        umr_not_watermarked[0, pos] = True
                        continue
                    else:
                        prev = prev_actual
                    if prev not in fwd_umr_cache:
                        fwd_umr_cache[prev] = _dmark_forward_green_mask(
                            umr_seed, prev, vocab_size, gamma, logits.device
                        )
                    green_mask_umr = fwd_umr_cache[prev]
                    probs_umr = F.softmax(logits[0, pos].float(), dim=-1)
                    tau_umr = probs_umr[green_mask_umr].sum().item()
                    if tau_umr <= 0.0 or tau_umr >= 1.0:
                        continue
                    dp_umr = float(amplification) * tau_umr / (1.0 - tau_umr)
                    if dp_umr >= 1.0:
                        dp_umr = 1.0 - 1e-6
                    boost_umr = torch.ones(vocab_size, dtype=torch.float32, device=logits.device)
                    boost_umr[green_mask_umr] = 1.0 + float(amplification)
                    boost_umr[~green_mask_umr] = 1.0 - dp_umr
                    new_probs_umr = probs_umr * boost_umr
                    new_probs_umr = new_probs_umr / (new_probs_umr.sum() + 1e-10)
                    x0[0, pos] = torch.argmax(new_probs_umr)

            x[transfer_index] = x0[transfer_index]

            # UMR R-remasking: positions committed this step without watermark
            # (unstable context at the time of generation) become regret candidates.
            if umr_candidates is not None and umr_not_watermarked is not None:
                umr_candidates |= transfer_index & umr_not_watermarked

    return x
