# MatFormer Evolution

This chapter documents the development history of Psyche's MatFormer heterogeneous training capabilities, covering the key architectural decisions and their rationale.

## Evolution Timeline

The MatFormer implementation evolved through several distinct phases:

| Phase | Focus | Key Changes |
|-------|-------|-------------|
| 0 | Infrastructure | Metrics, DisTrO MatFormer slicing |
| 1 | Data Format | Detection for modded-nanogpt |
| 2-3 | Kernels | Polar Express orthogonalization |
| 4 | Optimizers | Muon optimizer implementation |
| 5a | Precision | Adaptive precision detection (V100/A100/H100) |
| 5b | Architecture | NanoGPT + MatFormer support |
| 6 | Helper Mode | Stochastic suffix sampling |
| 7 | Compatibility | Schema canonicalization + tier management |
| 8 | Resolution | Hub-based tier resolution |
| 9 | Testing | Testing & reliability infrastructure |
| 10 | Polish | Documentation & production readiness |

## Key Architectural Decisions

### Decision 1: Tier-0 Canonical Hash

**Problem:** V100 (tier-1) loads pre-sliced checkpoints; A100 (tier-0) loads universal checkpoints. Different configs would produce different schema hashes, causing client rejection.

**Solution:** Canonicalize all configs to tier-0 equivalent before hashing.

**Rationale:**
- The *effective model architecture* is the same across tiers
- Only the *active subset* differs during training
- Schema hash should represent model compatibility, not runtime configuration

**Implementation:** `canonicalize_config_for_schema()` in `shared/client/src/state/init.rs`

### Decision 2: Helper Mode Disabled for Non-Zero Tiers

**Problem:** Helper mode samples stochastic suffix indices for gradient diversity. These indices are generated assuming full FFN width, causing out-of-bounds access for tier-1+ clients.

**Solution:** Auto-disable helper mode when loading sliced checkpoints or using non-zero tiers.

**Rationale:**
- Sparse alignment with non-contiguous indices adds complexity
- Helper mode provides marginal benefit for small-scale training
- Clean prefix-only training is simpler to verify

**Implementation:** `should_disable_helper()` detection in model loading

### Decision 3: Contributing Peers Normalization

**Problem:** Homogeneous path uses `"mean"` reduction via `scatter_reduce`, but heterogeneous path originally used raw summation.

**Solution:** Add `contributing_peers` counter and normalize heterogeneous gradients by peer count.

**Rationale:**
- Semantically consistent with homogeneous path
- Sign-SGD mitigates but doesn't eliminate the difference
- Correct behavior if optimizer changes in future

**Implementation:** `shared/modeling/src/distro.rs:857-896`

```rust
let mut contributing_peers: usize = 0;
// ... aggregation loop ...
let normalized = if contributing_peers > 1 {
    combined / (contributing_peers as f64)
} else {
    combined
};
```

### Decision 4: Double-Slicing as Hard Error

**Problem:** Loading a sliced checkpoint with `--matformer-tier > 0` and `--matformer-load-strategy universal` would apply slicing twice, corrupting the model.

**Solution:** Detect sliced checkpoints via metadata and reject double-slicing.

**Rationale:**
- Silent corruption is worse than a clear error
- Users can fix configuration and retry
- Metadata makes sliced checkpoints self-describing

**Implementation:** `validate_no_double_slicing()` in model loading

### Decision 5: Dimension-Aware Gradient Alignment

**Problem:** FFN layers have different weight matrix orientations:
- `gate_proj`, `up_proj`: `[intermediate_size, hidden_size]` (FFN width in rows)
- `down_proj`: `[hidden_size, intermediate_size]` (FFN width in columns)

**Solution:** Use parameter names to determine alignment dimension.

**Rationale:**
- Layer names carry architectural semantics
- No need for explicit tier tracking during aggregation
- Generalizes to any FFN variant following naming conventions

**Implementation:** `matformer_prefix_dim()` in `shared/modeling/src/distro.rs`

```rust
fn matformer_prefix_dim(name: &str) -> Option<usize> {
    if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
        Some(0)  // FFN width in rows
    } else if name.ends_with("down_proj.weight") {
        Some(1)  // FFN width in columns
    } else {
        None     // Non-FFN parameter
    }
}
```

## What Made It Into Production

### Working Features

1. **Schema hash canonicalization**: Different checkpoint types can join same run
2. **Gradient shape alignment**: `align_matformer_prefix_grad()` handles shape differences
3. **Heterogeneous code path detection**: `same_shape` check routes correctly
4. **Witness/consensus system**: Tier-agnostic verification via blob hashes
5. **Self-describing checkpoints**: Tier metadata enables auto-inference
6. **Startup configuration summary**: Clear feedback on effective configuration

### Recommended Configuration

```bash
# Heterogeneous training (production-ready)
--client-matformer-tiers 0,1,1 \
--client-matformer-helper-fractions 0,0,0  # Required: disable helper mode for tier > 0
```

## Lessons Learned

### What Worked Well

1. **Incremental development**: Each phase built on verified previous work
2. **Unit tests for invariants**: `matformer_mlp_has_zero_tail_grads` caught issues early
3. **Clear error messages**: Schema mismatches now provide actionable feedback
4. **Self-describing data**: Checkpoint metadata prevents misconfiguration

### What Required Iteration

1. **Helper mode complexity**: Initially attempted sparse alignment, ultimately disabled
2. **Normalization semantics**: Required analysis to match homogeneous path
3. **Checkpoint tier inference**: Multiple fallback strategies for compatibility

### Technical Debt Addressed

1. **Parameter naming in DistroResult**: Added to enable per-parameter alignment
2. **Tier metadata in checkpoints**: Enables double-slicing detection
3. **Schema canonicalization**: Unifies different checkpoint loading paths

## Testing Strategy

The implementation uses layered testing:

1. **Unit tests**: Verify mathematical properties (zero gradients, alignment correctness)
2. **Integration tests**: Local testnet with multiple tier configurations
3. **Smoke tests**: Headless runs with loss convergence checks

See [MatFormer Verification](./matformer-verification.md) for the complete test suite.

## Future Work

### Potential Enhancements

1. **Helper mode for non-zero tiers**: Requires sparse index transmission
2. **Per-layer tier schedules**: Different tiers for different layers (Mix'n'Match)
3. **Attention head slicing**: Extend MatFormer beyond FFN
4. **Adaptive tier assignment**: Server assigns tiers based on client capabilities

### Dependencies for Production Scale

1. **Real hardware validation**: V100 + A100 mixed clusters
2. **Network resilience testing**: Latency, packet loss, client churn
3. **Statistical reproducibility**: Multiple seeds, reproducible results
