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


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


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


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, 
             gamma=0.5, amplification=0.0, vocab_size=126464, watermark_steps=None, 
             special_token_ids=None):
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
        gamma: Fraction of tokens that are "green" (watermarked).
        amplification: Amplification factor for green tokens (0 = no watermarking).
        vocab_size: Size of the vocabulary.
        watermark_steps: Maximum step to watermark at (int), list of specific steps, or None for all steps.
                        If int, watermarks at steps 1 to watermark_steps.
                        If list, watermarks at the specified step indices.
                        If None, watermarks at all steps.
        special_token_ids: List of token IDs to exclude from amplification (optional).
    '''
    # breakpoint()
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # Generate green mask for watermarking if amplification > 0
    green_mask = None
    if amplification > 0:
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

            # Apply watermark to logits before adding Gumbel noise
            if amplification > 0 and green_mask is not None:
                # Check if we should apply watermarking at this step
                # If watermark_steps is None, apply to all steps
                # If watermark_steps is an integer, apply to steps 1 to watermark_steps
                # If watermark_steps is a list, apply to those specific steps
                if watermark_steps is None:
                    should_watermark = True
                elif isinstance(watermark_steps, int):
                    should_watermark = (i + 1) <= watermark_steps  # i is 0-indexed, so i+1 is the step number
                else:
                    should_watermark = i in watermark_steps
                
                if should_watermark:
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
                    # breakpoint()
                    logits = apply_watermark_to_logits(logits, full_green_mask, amplification, current_block_mask, special_token_ids)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x
