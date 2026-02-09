# MatFormer Verification

This chapter documents the test suite for verifying MatFormer heterogeneous training correctness.

## Mathematical Correctness Tests

### Zero-Tail Gradients

Verifies that exclusive weights (tail neurons) have zero gradient when training at a smaller tier:

```bash
cargo test -p psyche-modeling matformer_mlp_has_zero_tail_grads
```

**What it tests:**
- Tier-1 client's backward pass produces zeros for suffix neurons
- No gradient "leakage" from prefix to suffix
- `.narrow()` slicing preserves correct gradient flow

### Gradient Alignment: Expansion

Tests that smaller gradients are correctly expanded with zeros:

```bash
cargo test -p psyche-modeling test_align_matformer_prefix_grad_expand_gate
```

**What it tests:**
- `[512, 256]` gradient expands to `[1024, 256]`
- Prefix values preserved in positions 0-511
- Suffix positions filled with zeros

### Gradient Alignment: Slicing

Tests that larger gradients are correctly sliced:

```bash
cargo test -p psyche-modeling test_align_matformer_prefix_grad_slice_down
```

**What it tests:**
- `[256, 1024]` gradient slices to match `[256, 512]` local shape
- Correct dimension identified for `down_proj` (column slicing)

### Non-MLP Parameter Rejection

Tests that non-FFN parameters are correctly rejected:

```bash
cargo test -p psyche-modeling test_align_matformer_prefix_grad_rejects_non_mlp
```

**What it tests:**
- Attention weights return `UnsupportedParameter` error
- Embedding weights return `UnsupportedParameter` error
- Only `gate_proj`, `up_proj`, `down_proj` are accepted

## Integration Tests

### Local Testnet: Homogeneous Baseline

Establishes ground truth loss curve:

```bash
DYLD_LIBRARY_PATH=./.venv/lib/python3.12/site-packages/torch/lib \
cargo run --release -p psyche-centralized-local-testnet -- start \
  --headless \
  --headless-exit-after-secs 120 \
  --num-clients 3 \
  --config-path ./config/nanogpt-20m-run \
  --client-matformer-tiers 0,0,0 \
  --tui false
```

**Success criteria:**
- Loss decreases over training
- No errors or panics
- All clients complete all rounds

### Local Testnet: Heterogeneous (1:2 Ratio)

Tests mixed tier training:

```bash
DYLD_LIBRARY_PATH=./.venv/lib/python3.12/site-packages/torch/lib \
cargo run --release -p psyche-centralized-local-testnet -- start \
  --headless \
  --headless-exit-after-secs 120 \
  --num-clients 3 \
  --config-path ./config/nanogpt-20m-run \
  --client-matformer-tiers 0,1,1 \
  --client-matformer-helper-fractions 0,0,0 \
  --tui false
```

**Success criteria:**
- Final loss within 15% of homogeneous baseline
- No gradient explosion/NaN
- All clients complete all rounds

### Local Testnet: Extreme Ratio (1:3)

Stress tests with minimal tier-0 capacity:

```bash
DYLD_LIBRARY_PATH=./.venv/lib/python3.12/site-packages/torch/lib \
cargo run --release -p psyche-centralized-local-testnet -- start \
  --headless \
  --headless-exit-after-secs 200 \
  --num-clients 4 \
  --config-path ./config/nanogpt-20m-run \
  --client-matformer-tiers 0,1,1,1 \
  --client-matformer-helper-fractions 0,0,0,0 \
  --tui false
```

**Success criteria:**
- Training completes without errors
- Loss still decreases (may be slower)

## Code Correctness Checklist

When reviewing MatFormer changes, verify:

- [ ] `matformer_prefix_dim()` correctly identifies gate/up (dim=0) vs down (dim=1)
- [ ] `align_matformer_prefix_grad()` expands smaller gradients with zeros
- [ ] `contributing_peers` counts only successful alignments
- [ ] Normalization matches "mean" semantics from batch_decompress
- [ ] Schema hash uses tier=0 canonical form
- [ ] Double-slicing detected and rejected with clear error

## Diagnostic Commands

### Check Gradient Shapes

Add logging to verify gradient dimensions during training:

```rust
tracing::debug!(
    parameter = var.name(),
    local_shape = ?var.local_tensor().size(),
    full_shape = ?full_shape,
    "Processing gradient"
);
```

### Verify Schema Hash

Compare schema hashes from different tier configurations:

```bash
# Should produce same hash for compatible configs
cargo run -p psyche-centralized-client -- print-schema-hash \
  --checkpoint path/to/checkpoint \
  --matformer-tier 0

cargo run -p psyche-centralized-client -- print-schema-hash \
  --checkpoint path/to/checkpoint-tier1 \
  --matformer-tier 1
```

### Profile Memory Usage

Compare memory between tiers:

```bash
# Tier-0 baseline
/usr/bin/time -l cargo run --release -p psyche-centralized-client -- train \
  --matformer-tier 0 ... 2>&1 | grep "maximum resident"

# Tier-1 comparison
/usr/bin/time -l cargo run --release -p psyche-centralized-client -- train \
  --matformer-tier 1 ... 2>&1 | grep "maximum resident"
```

## Known Issues and Workarounds

### Helper Mode with Non-Zero Tiers

**Issue:** Helper mode generates indices for full FFN width, causing out-of-bounds errors with tier-1+.

**Workaround:** Disable helper mode:
```bash
--client-matformer-helper-fractions 0,0,0
```

**Status:** Auto-disabled when sliced checkpoint detected.

### Double-Slicing Detection

**Issue:** Loading sliced checkpoint with `--matformer-load-strategy universal` and `--matformer-tier > 0` would slice twice.

**Workaround:** Use `--matformer-load-strategy auto` (default) or `sliced`.

**Status:** Hard error with clear message.

### Memory Savings at Small Scale

**Issue:** Memory savings are modest (~15%) for small models because embeddings dominate.

**Expected:** Larger models (7B+) see ~32% savings from tier-1.

**Status:** Working as designed; savings scale with model size.

## Experiment Scripts

### Run Full Test Matrix

```bash
#!/bin/bash
# scripts/run_matformer_tests.sh

set -e

echo "=== Unit Tests ==="
cargo test -p psyche-modeling matformer

echo "=== Homogeneous Baseline ==="
./scripts/run_experiment.sh 0,0,0 > results/homogeneous.log 2>&1

echo "=== Heterogeneous 1:2 ==="
./scripts/run_experiment.sh 0,1,1 > results/heterogeneous_1to2.log 2>&1

echo "=== Heterogeneous 2:1 ==="
./scripts/run_experiment.sh 0,0,1 > results/heterogeneous_2to1.log 2>&1

echo "=== Analysis ==="
./scripts/analyze_results.sh results/
```

### Analyze Results

```bash
#!/bin/bash
# scripts/analyze_results.sh

RESULTS_DIR="${1:-.}"

echo "=== Final Loss Comparison ==="
for log in "$RESULTS_DIR"/*.log; do
    name=$(basename "$log" .log)
    final_loss=$(grep -oP 'loss[=:]\s*\K[0-9.]+' "$log" | tail -1)
    echo "$name: $final_loss"
done

echo ""
echo "=== Error Check ==="
for log in "$RESULTS_DIR"/*.log; do
    name=$(basename "$log" .log)
    errors=$(grep -ci 'error\|panic\|nan\|inf' "$log" 2>/dev/null || echo 0)
    echo "$name: $errors issues"
done
```

## Continuous Integration

The following tests run in CI:

1. `cargo test -p psyche-modeling` - Unit tests including MatFormer
2. `just integration-test` - Full integration test suite
3. Headless smoke test with `--headless-exit-after-secs 10`

For heterogeneous-specific CI, consider adding:

```yaml
heterogeneous-smoke-test:
  script:
    - cargo run --release -p psyche-centralized-local-testnet -- start \
        --headless --headless-exit-after-secs 60 \
        --num-clients 3 \
        --client-matformer-tiers 0,1,1 \
        --client-matformer-helper-fractions 0,0,0 \
        --config-path ./config/test-nanogpt-small
```
