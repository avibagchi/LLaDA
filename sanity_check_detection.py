#!/usr/bin/env python3
"""
Sanity check script to verify Aaronson watermark detection indices align correctly.

This script:
1. Generates text with watermarking at known positions
2. Detects watermark with CORRECT position_offset
3. Detects watermark with WRONG position_offset (should fail)
4. Verifies detection scores are as expected
"""
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from generate import generate, calculate_aaronson_watermark_score, get_special_token_ids


def test_detection_alignment(model, tokenizer, prompt_text, gen_length, steps, device, aaronson_seed=42):
    """
    Test that detection indices align with generation indices.
    """
    print(f"\n{'='*80}")
    print("SANITY CHECK: Position Index Alignment")
    print(f"{'='*80}")
    
    # Get special token IDs
    special_token_ids = get_special_token_ids(tokenizer)
    
    # Tokenize prompt
    prompt_tokens = tokenizer(prompt_text)["input_ids"]
    prompt_tensor = torch.tensor([prompt_tokens]).to(device)
    prompt_len = len(prompt_tokens)
    
    print(f"\nPrompt: {prompt_text}")
    print(f"Prompt length: {prompt_len} tokens")
    print(f"Generation length: {gen_length} tokens")
    print(f"Aaronson seed: {aaronson_seed}")
    
    # ========================================================================
    # TEST 1: Generate with watermarking
    # ========================================================================
    print(f"\n{'='*80}")
    print("TEST 1: Generate with Aaronson watermarking")
    print(f"{'='*80}")
    
    with torch.no_grad():
        generated = generate(
            model=model,
            prompt=prompt_tensor,
            steps=steps,
            gen_length=gen_length,
            block_length=gen_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            mask_id=126336,
            watermark_type='aaronson',
            aaronson_seed=aaronson_seed,
            watermark_steps=None,  # Watermark all steps
            vocab_size=126464,
            special_token_ids=special_token_ids
        )
    
    generated_tokens = generated[0, prompt_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"✓ Generated {len(generated_tokens)} tokens")
    print(f"Generated text: {generated_text[:100]}...")
    
    # ========================================================================
    # TEST 2: Detect with CORRECT position_offset
    # ========================================================================
    print(f"\n{'='*80}")
    print("TEST 2: Detect with CORRECT position_offset")
    print(f"{'='*80}")
    
    score_correct, length_correct, _ = calculate_aaronson_watermark_score(
        generated_tokens.unsqueeze(0),
        vocab_size=126464,
        seed=aaronson_seed,
        special_token_ids=special_token_ids,
        position_offset=prompt_len  # CORRECT: matches generation
    )
    
    normalized_score_correct = score_correct / length_correct if length_correct > 0 else 0
    
    print(f"Position offset used: {prompt_len} (CORRECT)")
    print(f"Raw score: {score_correct:.4f}")
    print(f"Normalized score: {normalized_score_correct:.4f}")
    print(f"Length analyzed: {length_correct} tokens")
    
    # ========================================================================
    # TEST 3: Detect with WRONG position_offset (should give lower score)
    # ========================================================================
    print(f"\n{'='*80}")
    print("TEST 3: Detect with WRONG position_offset (offset=0)")
    print(f"{'='*80}")
    
    score_wrong, length_wrong, _ = calculate_aaronson_watermark_score(
        generated_tokens.unsqueeze(0),
        vocab_size=126464,
        seed=aaronson_seed,
        special_token_ids=special_token_ids,
        position_offset=0  # WRONG: doesn't match generation
    )
    
    normalized_score_wrong = score_wrong / length_wrong if length_wrong > 0 else 0
    
    print(f"Position offset used: 0 (WRONG - should be {prompt_len})")
    print(f"Raw score: {score_wrong:.4f}")
    print(f"Normalized score: {normalized_score_wrong:.4f}")
    print(f"Length analyzed: {length_wrong} tokens")
    
    # ========================================================================
    # TEST 4: Detect with ANOTHER WRONG offset
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"TEST 4: Detect with WRONG position_offset (offset={prompt_len + 50})")
    print(f"{'='*80}")
    
    score_wrong2, length_wrong2, _ = calculate_aaronson_watermark_score(
        generated_tokens.unsqueeze(0),
        vocab_size=126464,
        seed=aaronson_seed,
        special_token_ids=special_token_ids,
        position_offset=prompt_len + 50  # WRONG: off by 50
    )
    
    normalized_score_wrong2 = score_wrong2 / length_wrong2 if length_wrong2 > 0 else 0
    
    print(f"Position offset used: {prompt_len + 50} (WRONG - should be {prompt_len})")
    print(f"Raw score: {score_wrong2:.4f}")
    print(f"Normalized score: {normalized_score_wrong2:.4f}")
    print(f"Length analyzed: {length_wrong2} tokens")
    
    # ========================================================================
    # VERIFICATION
    # ========================================================================
    print(f"\n{'='*80}")
    print("VERIFICATION RESULTS")
    print(f"{'='*80}")
    
    # For watermarked text, the correct offset should give a significantly higher score
    # The expected normalized score for random text is around 0.69 (ln 2)
    # Watermarked text should be noticeably higher
    
    baseline_expected = 0.69  # Expected score for random text
    
    print(f"\nExpected baseline (random text): ~{baseline_expected:.2f}")
    print(f"\nCorrect offset score:  {normalized_score_correct:.4f}")
    print(f"Wrong offset (0):      {normalized_score_wrong:.4f}")
    print(f"Wrong offset (+50):    {normalized_score_wrong2:.4f}")
    
    # Check if correct offset gives higher score
    correct_is_best = (normalized_score_correct > normalized_score_wrong and 
                       normalized_score_correct > normalized_score_wrong2)
    
    if correct_is_best:
        print(f"\n✓ PASS: Correct offset gives highest score!")
        print(f"  Correct offset is {normalized_score_correct / normalized_score_wrong:.2f}x higher than wrong offset (0)")
        print(f"  Correct offset is {normalized_score_correct / normalized_score_wrong2:.2f}x higher than wrong offset (+50)")
    else:
        print(f"\n✗ FAIL: Correct offset does NOT give highest score!")
        print(f"  This indicates a bug in position alignment!")
    
    # Check if correct offset is significantly above baseline
    if normalized_score_correct > baseline_expected * 1.5:
        print(f"\n✓ PASS: Watermark is detectable (score is {normalized_score_correct / baseline_expected:.2f}x baseline)")
    else:
        print(f"\n⚠ WARNING: Watermark signal may be weak (score is only {normalized_score_correct / baseline_expected:.2f}x baseline)")
    
    # Summary
    print(f"\n{'='*80}")
    if correct_is_best and normalized_score_correct > baseline_expected * 1.5:
        print("OVERALL: ✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("Position indices are correctly aligned between generation and detection!")
    else:
        print("OVERALL: ✗✗✗ CHECKS FAILED ✗✗✗")
        print("There may be an issue with position index alignment!")
    print(f"{'='*80}\n")
    
    return {
        'correct_score': normalized_score_correct,
        'wrong_score_0': normalized_score_wrong,
        'wrong_score_offset': normalized_score_wrong2,
        'passed': correct_is_best
    }


def main():
    parser = argparse.ArgumentParser(description='Sanity check for Aaronson watermark detection')
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Base', help='Model path')
    parser.add_argument('--prompt', type=str, default='What is the capital of France?', help='Test prompt')
    parser.add_argument('--gen_length', type=int, default=128, help='Number of tokens to generate')
    parser.add_argument('--steps', type=int, default=128, help='Number of sampling steps')
    parser.add_argument('--aaronson_seed', type=int, default=42, help='Aaronson seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_tests', type=int, default=3, help='Number of test prompts to run')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model = AutoModel.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    print("✓ Model loaded successfully")
    
    # Test prompts of different lengths
    test_prompts = [
        "What is the capital of France?",
        "Explain machine learning.",
        "Once upon a time, in a land far away, there lived a wise old wizard who knew the secrets of the universe.",
    ][:args.num_tests]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n\n{'#'*80}")
        print(f"# TEST PROMPT {i+1}/{len(test_prompts)}")
        print(f"{'#'*80}")
        
        result = test_detection_alignment(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            gen_length=args.gen_length,
            steps=args.steps,
            device=args.device,
            aaronson_seed=args.aaronson_seed
        )
        
        results.append(result)
    
    # Final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    print(f"\nTests passed: {passed_count}/{total_count}")
    
    for i, result in enumerate(results):
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"\nTest {i+1}: {status}")
        print(f"  Correct offset score: {result['correct_score']:.4f}")
        print(f"  Wrong offset (0):     {result['wrong_score_0']:.4f}")
        print(f"  Wrong offset (+50):   {result['wrong_score_offset']:.4f}")
    
    if passed_count == total_count:
        print(f"\n{'='*80}")
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Position indices are correctly aligned!")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("⚠ SOME TESTS FAILED ⚠")
        print("Please check the position index alignment!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

