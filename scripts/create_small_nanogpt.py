#!/usr/bin/env python3
"""Create a small NanoGPT checkpoint (~20M params) for testing."""

import json
import torch
from safetensors.torch import save_file
from transformers import GPT2Tokenizer
import os
import math

# Target: ~20M params to match llama2-20m-init
# LLaMA 20M: 256 hidden, 256 intermediate, 8 layers, 32000 vocab

# NanoGPT config to match approximately:
# - hidden_size: 256
# - intermediate_size: 1024 (4x hidden, standard GPT ratio)
# - num_layers: 8
# - vocab_size: 50257 (GPT-2 tokenizer)
# - num_attention_heads: 8

config = {
    "hidden_size": 256,
    "intermediate_size": 1024,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "vocab_size": 50257,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "use_fused_qkvo": False,
    "use_relu_squared_mlp": False,
    "mlp_bias": False,
    "use_qk_norm": False,
    "use_sa_lambdas": False,
    "use_x0_residual": False,
    "use_x0_lambdas": False,
    "use_resid_lambdas": False,
    "num_value_embeddings": 0,
    "use_block_skip": False,
    "use_smear_gate": False,
    "use_logit_softcap": False,
    "use_backout": False,
    "use_half_truncate_rope": False,
    "use_key_offset": False,
    "use_learnable_attn_scale": False,
    "use_attention_gate": False,
}
config["matformer_tier"] = 0
config["matformer_base_intermediate_size"] = config["intermediate_size"]

def count_params(config):
    """Estimate parameter count."""
    h = config["hidden_size"]
    i = config["intermediate_size"]
    n = config["num_hidden_layers"]
    v = config["vocab_size"]

    # Embeddings
    embed_params = v * h  # token embeddings

    # Per layer
    # Attention: Q, K, V, O projections
    attn_params = 4 * h * h
    # MLP: gate, up, down
    mlp_params = 3 * h * i
    # Layer norms (2 per layer)
    norm_params = 2 * h

    layer_params = attn_params + mlp_params + norm_params

    # Final layer norm + lm_head
    final_params = h + v * h

    total = embed_params + n * layer_params + final_params
    return total

print(f"Estimated params: {count_params(config):,}")

def init_weights(shape, std=0.02):
    """Initialize weights with normal distribution."""
    return torch.randn(shape) * std

def create_model_weights(config):
    """Create initialized model weights."""
    h = config["hidden_size"]
    i = config["intermediate_size"]
    n = config["num_hidden_layers"]
    v = config["vocab_size"]
    n_heads = config["num_attention_heads"]
    head_dim = h // n_heads

    weights = {}

    # Token embeddings
    weights["model.embed_tokens.weight"] = init_weights((v, h))

    # Layers
    for layer_idx in range(n):
        prefix = f"model.layers.{layer_idx}"

        # Input layer norm
        weights[f"{prefix}.input_layernorm.weight"] = torch.ones(h)

        # Self attention
        weights[f"{prefix}.self_attn.q_proj.weight"] = init_weights((h, h))
        weights[f"{prefix}.self_attn.k_proj.weight"] = init_weights((h, h))
        weights[f"{prefix}.self_attn.v_proj.weight"] = init_weights((h, h))
        weights[f"{prefix}.self_attn.o_proj.weight"] = init_weights((h, h))

        # Post attention layer norm
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(h)

        # MLP
        weights[f"{prefix}.mlp.gate_proj.weight"] = init_weights((i, h))
        weights[f"{prefix}.mlp.up_proj.weight"] = init_weights((i, h))
        weights[f"{prefix}.mlp.down_proj.weight"] = init_weights((h, i))

    # Final layer norm
    weights["model.norm.weight"] = torch.ones(h)

    # LM head
    weights["lm_head.weight"] = init_weights((v, h))

    return weights

# Create output directory
output_dir = "/tmp/nanogpt-20m-init"
os.makedirs(output_dir, exist_ok=True)

# Save config
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Create and save weights
print("Creating model weights...")
weights = create_model_weights(config)

# Count actual params
total_params = sum(w.numel() for w in weights.values())
print(f"Actual params: {total_params:,}")

print("Saving weights...")
save_file(weights, os.path.join(output_dir, "model.safetensors"))

# Copy tokenizer files from GPT-2
print("Setting up tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained(output_dir)

print(f"Checkpoint saved to {output_dir}")
print(f"Config: {config['hidden_size']} hidden, {config['intermediate_size']} intermediate, {config['num_hidden_layers']} layers")
