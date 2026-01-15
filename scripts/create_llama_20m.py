#!/usr/bin/env python3
"""Create a LLaMA 20M checkpoint matching the NanoGPT-matched architecture.

Both models will have:
- hidden_size: 256
- intermediate_size: 256
- num_hidden_layers: 8
- num_attention_heads: 16
- num_key_value_heads: 2 (GQA with 8:1 ratio)

LLaMA will use vocab_size: 32000 (LLaMA tokenizer)
"""

import json
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
import os

config = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 256,
    "intermediate_size": 256,  # Same as NanoGPT matched
    "initializer_range": 0.02,
    "max_position_embeddings": 2048,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 16,  # Same as NanoGPT matched
    "num_hidden_layers": 8,     # Same as NanoGPT matched
    "num_key_value_heads": 2,   # GQA with 8:1 ratio
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 32000,        # LLaMA tokenizer
}
config["matformer_tier"] = 0
config["matformer_base_intermediate_size"] = config["intermediate_size"]

def count_params(config):
    """Estimate parameter count with GQA."""
    h = config["hidden_size"]
    i = config["intermediate_size"]
    n = config["num_hidden_layers"]
    v = config["vocab_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = h // n_heads

    # Embeddings
    embed_params = v * h

    # Per layer attention with GQA
    kv_dim = n_kv_heads * head_dim
    attn_params = h * h + h * kv_dim + h * kv_dim + h * h  # Q, K, V, O

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
    """Create initialized model weights with GQA support."""
    h = config["hidden_size"]
    i = config["intermediate_size"]
    n = config["num_hidden_layers"]
    v = config["vocab_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = h // n_heads
    kv_dim = n_kv_heads * head_dim

    print(f"Head dim: {head_dim}, KV dim: {kv_dim}")
    print(f"Q/O shape: ({h}, {h}), K/V shape: ({kv_dim}, {h})")

    weights = {}

    # Token embeddings
    weights["model.embed_tokens.weight"] = init_weights((v, h))

    # Layers
    for layer_idx in range(n):
        prefix = f"model.layers.{layer_idx}"

        # Input layer norm
        weights[f"{prefix}.input_layernorm.weight"] = torch.ones(h)

        # Self attention with GQA
        weights[f"{prefix}.self_attn.q_proj.weight"] = init_weights((h, h))
        weights[f"{prefix}.self_attn.k_proj.weight"] = init_weights((kv_dim, h))
        weights[f"{prefix}.self_attn.v_proj.weight"] = init_weights((kv_dim, h))
        weights[f"{prefix}.self_attn.o_proj.weight"] = init_weights((h, h))

        # Post attention layer norm
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(h)

        # MLP (LLaMA style with gate, up, down)
        weights[f"{prefix}.mlp.gate_proj.weight"] = init_weights((i, h))
        weights[f"{prefix}.mlp.up_proj.weight"] = init_weights((i, h))
        weights[f"{prefix}.mlp.down_proj.weight"] = init_weights((h, i))

    # Final layer norm
    weights["model.norm.weight"] = torch.ones(h)

    # LM head
    weights["lm_head.weight"] = init_weights((v, h))

    return weights

# Create output directory
output_dir = "/tmp/llama-20m-matched"
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

# Create a minimal tokenizer config (will use LLaMA tokenizer from HF)
print("Setting up tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.environ.get("HF_TOKEN"))
    tokenizer.save_pretrained(output_dir)
    print("Saved LLaMA tokenizer")
except Exception as e:
    print(f"Could not download LLaMA tokenizer: {e}")
    print("Using GPT-2 tokenizer as fallback")
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained(output_dir)
    # Update config for GPT-2 vocab
    config["vocab_size"] = 50257
    config["bos_token_id"] = 50256
    config["eos_token_id"] = 50256
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    # Recreate weights with GPT-2 vocab
    print("Recreating weights with GPT-2 vocab size...")
    weights = create_model_weights(config)
    total_params = sum(w.numel() for w in weights.values())
    print(f"Actual params with GPT-2 vocab: {total_params:,}")
    save_file(weights, os.path.join(output_dir, "model.safetensors"))

print(f"\nCheckpoint saved to {output_dir}")
print(f"Config: {config['hidden_size']} hidden, {config['intermediate_size']} intermediate, {config['num_hidden_layers']} layers")
print(f"Attention: {config['num_attention_heads']} heads, {config['num_key_value_heads']} KV heads (GQA)")
print(f"Vocab size: {config['vocab_size']}")
