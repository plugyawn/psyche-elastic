#!/usr/bin/env python3
"""Inference script for NanoGPT/LLaMA models trained with Psyche.

Usage:
    python scripts/inference_nanogpt.py --model ./checkpoints/nanogpt-trained/nanogpt-20m-fineweb-step100
    python scripts/inference_nanogpt.py --model ./checkpoints/nanogpt-trained/nanogpt-20m-fineweb-step100 --prompt "Once upon a time"
"""

import argparse
import json
import os
import torch
from safetensors.torch import load_file
from transformers import (
    GPT2TokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    TextStreamer,
)


def load_model(checkpoint_path: str, device: str = "auto"):
    """Load a NanoGPT/LLaMA model from a Psyche checkpoint."""

    # Load config
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {checkpoint_path}")

    with open(config_path) as f:
        config = json.load(f)

    print(f"Model config:")
    print(f"  hidden_size: {config.get('hidden_size')}")
    print(f"  num_layers: {config.get('num_hidden_layers')}")
    print(f"  num_heads: {config.get('num_attention_heads')}")
    print(f"  vocab_size: {config.get('vocab_size')}")

    # Create LlamaConfig (NanoGPT uses LLaMA architecture)
    llama_config = LlamaConfig(
        vocab_size=config.get("vocab_size", 50257),
        hidden_size=config.get("hidden_size", 256),
        intermediate_size=config.get("intermediate_size", 1024),
        num_hidden_layers=config.get("num_hidden_layers", 8),
        num_attention_heads=config.get("num_attention_heads", 8),
        num_key_value_heads=config.get("num_key_value_heads", 8),
        max_position_embeddings=config.get("max_position_embeddings", 2048),
        rms_norm_eps=config.get("rms_norm_eps", 1e-5),
        rope_theta=config.get("rope_theta", 10000.0),
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    )

    # Load weights
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No model.safetensors found in {checkpoint_path}")

    print(f"\nLoading weights from {weights_path}...")
    weights = load_file(weights_path)

    # Create and load model
    model = LlamaForCausalLM(llama_config)
    missing, unexpected = model.load_state_dict(weights, strict=False)

    if missing:
        print(f"  Warning: Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {len(unexpected)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Move to device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"  Device: {device}")
    model = model.to(device)
    model.eval()

    return model, device


def load_tokenizer(checkpoint_path: str):
    """Load tokenizer from checkpoint or fall back to GPT-2."""
    tokenizer_path = os.path.join(checkpoint_path, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {checkpoint_path}")
        return GPT2TokenizerFast.from_pretrained(checkpoint_path)
    else:
        print("Using default GPT-2 tokenizer")
        return GPT2TokenizerFast.from_pretrained("gpt2")


def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stream: bool = True,
):
    """Generate text from the model."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    print("Generated: ", end="", flush=True)

    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=top_p if temperature > 0 else None,
            top_k=top_k if temperature > 0 else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    if not stream:
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated[len(prompt):])

    print(f"\n{'='*60}")

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained NanoGPT model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The meaning of life is",
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - keep prompting",
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, device = load_model(args.model, args.device)
    tokenizer = load_tokenizer(args.model)

    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\nPrompt: ")
                if prompt.lower() in ["quit", "exit", "q"]:
                    break
                generate(
                    model, tokenizer, prompt, device,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )
            except KeyboardInterrupt:
                break
    else:
        generate(
            model, tokenizer, args.prompt, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )


if __name__ == "__main__":
    main()
