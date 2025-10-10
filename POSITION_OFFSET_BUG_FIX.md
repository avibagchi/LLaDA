# Position Offset Bug Fix

## The Bug

The Aaronson watermark was using **incorrect position indices** causing a mismatch between generation and detection r-values.

### During Generation:
```python
# In apply_aaronson_gumbel_watermark()
for pos in range(seq_len):  # pos iterates over FULL sequence (prompt + generation)
    r_values = generate_pseudo_random_values(position_offset + pos, ...)
```

Called with: `position_offset=prompt_len`

**For first generated token** (at absolute position 7):
- `pos = 7` (already absolute position in full sequence)
- r-seed = `prompt_len + pos` = `7 + 7` = **14** ❌

### During Detection:
```python
for pos in range(actual_length):  # pos iterates over GENERATED tokens only
    r_values = generate_pseudo_random_values(position_offset + pos, ...)
```

Called with: `position_offset=prompt_len`

**For first generated token**:
- `pos = 0` (relative to start of generated tokens)
- r-seed = `prompt_len + pos` = `7 + 0` = **7** ❌

**Result:** Seeds don't match (14 vs 7), watermark detection fails!

## The Fix

Changed generation to use `position_offset=0`:

```python
# In generate() function, line 498-502
aaronson_choices, aaronson_wm_confidences = apply_aaronson_gumbel_watermark(
    logits, current_block_mask, vocab_size, 
    position_offset=0,  # ← Changed from prompt.shape[1] to 0
    seed=aaronson_seed,
    special_token_ids=special_token_ids
)
```

### Why This Works:

In `apply_aaronson_gumbel_watermark()`, the `pos` variable iterates over the **full sequence** (including prompt). So `pos` is already the **absolute position**. We don't need to add an offset.

**After fix, for first generated token** (at absolute position 7):
- Generation: `pos=7`, r-seed = `0 + 7` = **7** ✓
- Detection: `pos=0` (in slice), r-seed = `7 + 0` = **7** ✓
- **Seeds match!** ✓✓✓

### Detection Keeps position_offset:

Detection is correct as-is because it processes **only the generated tokens** (a slice), so it needs `position_offset` to convert relative positions to absolute positions.

## Verification

Run the sanity check script to verify the fix:

```bash
bash run_sanity_check.sh
```

Expected output:
```
✓ PASS: Correct offset gives highest score!
✓ PASS: Watermark is detectable
OVERALL: ✓✓✓ ALL CHECKS PASSED ✓✓✓
```

## Files Modified

- `generate.py`: Line 500, changed `position_offset=prompt.shape[1]` to `position_offset=0`

## Impact

This was a **critical bug** that completely broke Aaronson watermark detection. After this fix:
- ✓ Generation and detection r-values now align correctly
- ✓ Watermark is properly detectable
- ✓ Detection scores are significantly higher with correct offset vs wrong offset

