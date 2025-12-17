#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM


def build_wordlevel_tokenizer(vocab_size: int) -> Tokenizer:
    special_tokens = ["<unk>", "<bos>", "<eos>", "<pad>"]
    if vocab_size < len(special_tokens):
        raise ValueError(
            f"vocab_size must be >= {len(special_tokens)} (got {vocab_size})"
        )

    vocab: dict[str, int] = {tok: i for i, tok in enumerate(special_tokens)}
    for i in range(vocab_size - len(special_tokens)):
        vocab[f"tok{i}"] = i + len(special_tokens)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a tiny local LLaMA checkpoint (safetensors + tokenizer.json) for Psyche."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/tiny-llama-local"),
        help="Output directory (will be created).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--max-position-embeddings", type=int, default=128)
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=1.0e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )

    model = LlamaForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Creating tiny LLaMA with {total_params:,} parameters at {out_dir}")

    model.save_pretrained(out_dir, safe_serialization=True)

    tokenizer = build_wordlevel_tokenizer(args.vocab_size)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Wrote {tokenizer_path}")


if __name__ == "__main__":
    main()

