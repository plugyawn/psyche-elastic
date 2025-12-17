# MatFormer: Elastic Inference with Nested Models

MatFormer (Matryoshka Transformer) is a technique that enables training a single transformer model that supports multiple compute/quality trade-offs at inference time. Psyche implements MatFormer to allow heterogeneous clients with different hardware capabilities to participate in the same training run.

## Overview

Traditional transformers require running the full model for every inference. MatFormer introduces **nested submodels** where smaller models are strict prefixes of larger ones, sharing weights. This enables:

- **Heterogeneous training**: Clients with different VRAM/compute can train at different capacity levels
- **Elastic inference**: Deploy the same weights at multiple latency/quality points
- **Behavioral consistency**: Smaller models closely match larger model outputs due to shared weights

## How It Works

MatFormer primarily targets the **FFN (Feed-Forward Network)** blocks, which typically account for ~2/3 of transformer parameters. The technique works by:

1. **Prefix slicing**: Smaller models use the first N neurons of the FFN hidden dimension
2. **Shared weights**: The first N neurons are trained by ALL model sizes, making them "universally useful"
3. **Exponential tiers**: Widths are typically h, h/2, h/4, h/8 (powers of 2)

### Weight Structure

For a standard FFN with hidden size `h`:

```
FFN(x) = W_down · activation(W_up · x)

Where:
- W_up:   [hidden_size, embed_dim]  (projects up to hidden)
- W_down: [embed_dim, hidden_size]  (projects back down)
```

MatFormer at tier `t` uses only the first `h / 2^t` neurons:

```
FFN_tier(x) = W_down[:, :h_t] · activation(W_up[:h_t, :] · x)
```

### Tier System

| Tier | FFN Width | Use Case |
|------|-----------|----------|
| 0 | 100% (full) | Maximum quality, high-VRAM GPUs |
| 1 | 50% | Balanced quality/speed |
| 2 | 25% | Lower VRAM devices |
| 3 | 12.5% | Memory-constrained devices |

## Using MatFormer in Psyche

### Client Configuration

Clients specify their MatFormer tier via the `--matformer-tier` flag:

```bash
psyche-centralized-client train \
  --run-id "my-run" \
  --server-addr "server:8080" \
  --matformer-tier 1 \
  --device cuda
```

### Heterogeneous Training Example

A training run with clients of varying capabilities:

```
┌─────────────────────────────────────────────────────────────┐
│                     Server                                  │
└─────────────────────────────────────────────────────────────┘
          ▲              ▲              ▲
          │              │              │
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  RTX 4090   │  │  RTX 3060   │  │  M1 Mac     │
│  tier=0     │  │  tier=1     │  │  tier=2     │
│  (full)     │  │  (half)     │  │  (quarter)  │
└─────────────┘  └─────────────┘  └─────────────┘
```

### Local Testnet with Multiple Tiers

The local testnet supports heterogeneous tier assignment:

```bash
cargo run -p psyche-centralized-local-testnet -- start \
  --num-clients 3 \
  --client-matformer-tiers 0,1,2 \
  --config-path ./config/my-config
```

This assigns:
- Client 1: tier 0 (full width)
- Client 2: tier 1 (half width)
- Client 3: tier 2 (quarter width)

## Gradient Handling

When clients train at different tiers, gradients are computed only for the active prefix of weights:

- **Tier 0** clients compute gradients for ALL FFN weights
- **Tier 1** clients compute gradients for the first 50% of FFN neurons
- **Tier 2** clients compute gradients for the first 25% of FFN neurons

The shared prefix neurons receive gradient contributions from all clients, making them robust and general-purpose. Exclusive neurons (used only by larger tiers) are more specialized.

### Parameter Naming

Gradient distribution results now include parameter names to correctly aggregate gradients across heterogeneous clients:

```rust
pub struct SerializedDistroResult {
    pub parameter_name: String,  // Identifies which parameter
    pub sparse_idx: ...,
    pub sparse_val: ...,
}
```

## Implementation Details

### LLaMA FFN Implementation

The MatFormer-enabled FFN in `shared/modeling/src/models/llama.rs`:

```rust
impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let Some(matformer_hidden_size) = self.matformer_hidden_size else {
            // Full model path
            return self.down_proj.forward(
                &(self.gate_proj.forward(xs).silu() * self.up_proj.forward(xs)),
            );
        };

        // MatFormer path: use prefix slices via .narrow()
        let gate_w = self.gate_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
        let up_w = self.up_proj.linear.ws.narrow(0, 0, matformer_hidden_size);
        let down_w = self.down_proj.linear.ws.narrow(1, 0, matformer_hidden_size);

        let gate = xs.matmul(&gate_w.transpose(0, 1));
        let up = xs.matmul(&up_w.transpose(0, 1));
        let hidden = gate.silu() * up;
        hidden.matmul(&down_w.transpose(0, 1))
    }
}
```

Key implementation notes:
- `.narrow()` creates views (not copies), preserving gradient flow
- Tier is set at model load time via `from_pretrained_with_matformer_tier()`
- Tensor parallelism is not yet supported with MatFormer tiers > 0

### Client Capabilities Protocol

Clients advertise their tier in the capabilities handshake:

```rust
pub struct ClientCapabilities {
    pub device: String,        // "CUDA(0)", "MPS", "CPU"
    pub matformer_tier: u8,    // Requested tier
}

pub struct TrainingAssignment {
    pub matformer_tier: u8,    // Server-assigned tier
}
```

## Verification

A unit test verifies correct gradient isolation:

```rust
#[test]
fn matformer_mlp_has_zero_tail_grads() {
    // Verifies that exclusive weights (tail) have zero gradient
    // when training at a smaller tier
}
```

Run the test:

```bash
cargo test -p psyche-modeling matformer_mlp_has_zero_tail_grads
```

## Current Limitations

1. **Single tier per client**: Each client trains at one tier; true MatFormer trains all tiers per step
2. **No tensor parallelism**: MatFormer tiers > 0 require disabling tensor parallelism
3. **No Mix'n'Match**: Per-layer width schedules are not yet implemented
4. **Attention unchanged**: Only FFN width is elastic; attention heads remain fixed

## References

- [MatFormer: Nested Transformer for Elastic Inference](https://arxiv.org/abs/2310.07707) - Original paper
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) - Related work on nested representations
