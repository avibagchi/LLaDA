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
             aaronson_remasking_strategy='original', aaronson_tau_wm=0.2, aaronson_tau_orig=0.01, aaronson_lambda=0.7):
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
        watermark_type: Type of watermarking to use ('green_list' or 'aaronson').
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

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
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

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x
