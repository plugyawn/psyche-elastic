# Muon + DisTrO Integration Experiments

**Status**: FAILED - Integration does not provide benefits
**Date**: 2024-12 (Phase 6-7 of modded-nanogpt port)

## Background

Muon is an optimizer that orthogonalizes momentum updates using Newton-Schulz/Polar Express iteration. The hypothesis was that combining Muon's orthogonalization with DisTrO's DCT compression might be complementary.

## Implementation

Added to `shared/modeling/src/distro.rs`:
- `use_muon: bool` flag in `OptimizerDefinition::Distro`
- `muon_momentum: Option<f32>` (intended for separate momentum, never used)
- Orthogonalization applied in `generate()` before DCT encoding

Orthogonalization implementation in `shared/modeling/src/kernels/orthogonalize.rs`:
- Polar Express algorithm with pre-computed optimal coefficients
- Adaptive precision: FP32 on CPU, FP16 on V100, BF16 on A100+

## Hypothesis

> "Muon's orthogonalization makes singular values uniform, which might make DCT coefficients more uniformly distributed, allowing top-k to capture more representative information."

## Experiments

### Experiment 1: Pre-compression orthogonalization quality

**Test**: Compare reconstruction error, sign agreement, and cosine similarity with/without orthogonalization before DCT compression.

**Results** (from `test_orthogonalization_compression_quality`, `test_sign_agreement_after_compression`, `test_orthogonalization_gradient_direction_preservation`):

| Metric | Without Orth | With Orth | Winner |
|--------|--------------|-----------|--------|
| Reconstruction error | 53-96% | 93-100% | No orth |
| Sign agreement | 67-82% | 64-81% | No orth |
| Cosine similarity | 0.53-0.85 | 0.42-0.82 | No orth |

**Orthogonalization helped in 0/20 test cases.**

### Experiment 2: Singular value spectrum after compression

**Test**: Analyze how compression affects singular value distribution (`test_singular_value_spectrum_after_compression`).

**Results**:

| Shape | Top-k | Condition # (orig) | Condition # (recon) | Change |
|-------|-------|-------------------|--------------------| ------|
| 64x64 | 5% | 101.5 | 4,885,288 | 48,000x |
| 64x64 | 10% | 101.5 | 4,526,313 | 44,600x |
| 64x64 | 25% | 101.5 | 3,811 | 37x |

**Key finding**: Compression makes matrices nearly rank-deficient. Small SVs go to zero (0.00), not noise. Post-decompression orthogonalization would try to "revive" dead directions.

### Experiment 3: DCT coefficient distribution vs orthogonalization

**Test**: Check if there's a relationship between SVD and DCT compressibility (`test_dct_coefficient_distribution_vs_orthogonalization`).

**Results for random Gaussian 64x64**:

| Top-k % | Energy (orig) | Energy (orth) | Difference |
|---------|---------------|---------------|------------|
| 1% | 1.3% | 1.5% | ~same |
| 5% | 5.5% | 5.5% | identical |
| 25% | 24.9% | 25.0% | identical |

**Total DCT energy**:
- Original: 4148
- Orthogonalized: 65 (64x smaller!)

**Key finding**: DCT coefficient concentration is **independent** of singular value structure. The relative distribution is identical; only the absolute scale changes dramatically.

### Experiment 4: Structured (low-rank) matrices

**Test**: Check if low-rank matrices (more like real gradients) behave differently (`test_dct_with_structured_matrix`).

**Results**:

| Effective Rank | Top-10% Energy (orig) | Top-10% Energy (orth) | Verdict |
|----------------|----------------------|----------------------|---------|
| 1 | 5.1% | 11.0% | Orth better |
| 4 | 10.8% | 10.7% | Same |
| 16 | 11.1% | 10.6% | Same |
| 64 | 11.7% | 10.9% | Same |

**Surprising finding**: Low-rank structure does NOT make DCT more compressible! Rank-1 matrix has LESS concentrated DCT energy than full-rank.

## Analysis: Why SVD and DCT Are Unrelated

| Property | DCT | SVD |
|----------|-----|-----|
| Basis | Fixed frequency basis | Data-dependent |
| What it captures | Spatial patterns (smoothness) | Linear structure (rank) |
| Compressibility from | Smoothness/periodicity | Low-rank structure |

These are **orthogonal** (pun intended) concepts. A low-rank matrix can have any DCT spectrum. A smooth matrix can have any rank.

## Why Orthogonalization Hurts Compression

Original hypothesis (WRONG):
> "Orthogonalization spreads energy uniformly across singular directions, which hurts DCT's ability to concentrate energy in few coefficients."

Actual findings:
1. **DCT concentration is unchanged** by orthogonalization (relative distribution identical)
2. **Absolute scale shrinks ~100x** (orthogonal matrices have Frobenius norm = sqrt(min(m,n)))
3. **Reconstruction compares wrong scales**: We compress the orthogonalized (small) values but compare reconstruction to original (large) values

The mechanism isn't about "spreading energy" - it's about:
- Scale mismatch between compressed and original domains
- Changing which specific coefficients are selected by top-k
- Possibly: orthogonalization creates correlations that hurt specific coefficient recovery

## Post-Decompression Orthogonalization (Also Failed)

Alternative approach: orthogonalize AFTER decompression instead of before compression.

**Why it fails**:
- Compression makes matrices rank-deficient (small SVs → 0)
- Orthogonalization tries to make all SVs = 1
- Amplifying near-zero SVs = amplifying numerical noise
- Condition numbers of 4M+ indicate near-singularity

## Conclusion

**Muon and DisTrO are fundamentally incompatible:**

| Muon's Philosophy | DisTrO's Philosophy |
|-------------------|---------------------|
| All gradient directions equally important | Some directions more important |
| Equalize singular values | Exploit spectral sparsity |
| Operates on full-precision gradients | Operates on heavily compressed signals |

Muon assumes access to full gradient information to equalize optimization across directions. Once DCT+top-k has compressed the gradient, there's not enough information left for orthogonalization to help.

## Recommendation

Remove Muon integration code:
- `use_muon` flag and related fields
- `muon_momentum` buffer (never used)
- Orthogonalization call in `generate()`

Keep the orthogonalization kernel (`kernels/orthogonalize.rs`) - it's well-tested and might be useful for other purposes (e.g., weight orthogonalization regularization).

## Future Work (If Revisiting)

1. **Test with actual LLM gradients** - Our tests used random/synthetic matrices. Real gradients may have different properties (heavy tails, specific rank structure).

2. **Different compression schemes** - Muon might work with compression that preserves more gradient structure (e.g., random projection, structured sparsity).

3. **Orthogonalize weights, not gradients** - Periodic weight orthogonalization (different from Muon) might be compatible with DisTrO.

## Test Files

All empirical tests are in `shared/modeling/src/distro.rs`:
- `test_orthogonalization_compression_quality`
- `test_sign_agreement_after_compression`
- `test_orthogonalization_gradient_direction_preservation`
- `test_singular_value_spectrum_after_compression`
- `test_dct_coefficient_distribution_vs_orthogonalization`
- `test_dct_with_structured_matrix`
