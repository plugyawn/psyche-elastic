#!/usr/bin/env python3
"""
Export tier-sliced MatFormer checkpoints from a universal checkpoint.

Usage:
  python scripts/export_matformer_tiers.py --src checkpoints/matformer-test --tiers 1 2

This will write checkpoints/matformer-test-tier1 and -tier2 with:
- config.json updated to the sliced FFN width and matformer_tier=0
- sliced safetensors (single-shard only)
- tokenizer and auxiliary json files copied over
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file


def find_single_safetensors(src: Path) -> Path:
    candidates = sorted(src.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one safetensors file in {src}, found {len(candidates)}"
        )
    return candidates[0]


def slice_ffn_weights(weights: dict[str, torch.Tensor], divisor: int) -> dict[str, torch.Tensor]:
    sliced = {}
    for name, tensor in weights.items():
        if name.endswith("mlp.gate_proj.weight") or name.endswith("mlp.up_proj.weight"):
            hidden = tensor.shape[0]
            width = hidden // divisor
            sliced[name] = tensor[:width, :].contiguous()
        elif name.endswith("mlp.down_proj.weight"):
            hidden = tensor.shape[1]
            width = hidden // divisor
            sliced[name] = tensor[:, :width].contiguous()
        else:
            sliced[name] = tensor
    return sliced


def export_tier(src: Path, tier: int) -> Path:
    if tier < 1:
        raise ValueError("tier must be >=1 for slicing")
    dst = src.with_name(f"{src.name}-tier{tier}")
    dst.mkdir(parents=True, exist_ok=True)

    # Config
    cfg_path = src / "config.json"
    config = json.loads(cfg_path.read_text())
    if "intermediate_size" not in config:
        raise ValueError("config.json missing intermediate_size")
    divisor = 1 << tier
    config["intermediate_size"] = config["intermediate_size"] // divisor
    config["matformer_tier"] = 0
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    # Weights (single-shard only)
    src_weights_path = find_single_safetensors(src)
    weights = load_file(str(src_weights_path))
    sliced = slice_ffn_weights(weights, divisor)
    save_file(sliced, str(dst / src_weights_path.name))

    # Copy tokenizer and aux files
    for pattern in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "*.py",
    ]:
        for file in src.glob(pattern):
            shutil.copy(file, dst / file.name)

    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MatFormer tier checkpoints.")
    parser.add_argument("--src", required=True, type=Path, help="Path to universal checkpoint dir")
    parser.add_argument(
        "--tiers",
        required=True,
        type=int,
        nargs="+",
        help="Tier numbers to export (e.g., 1 2 for /2 and /4 FFN widths)",
    )
    args = parser.parse_args()

    src: Path = args.src
    tiers: Iterable[int] = args.tiers

    for tier in tiers:
        dst = export_tier(src, tier)
        print(f"[export] wrote {dst}")


if __name__ == "__main__":
    main()
