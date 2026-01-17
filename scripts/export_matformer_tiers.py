#!/usr/bin/env python3
"""
Export tier-sliced MatFormer checkpoints from a universal checkpoint.

Usage:
  python scripts/export_matformer_tiers.py --src checkpoints/matformer-test --tiers 1 2

This will write checkpoints/matformer-test-tier1 and -tier2 with:
- config.json updated to the sliced FFN width
- matformer_tier set to the tier number
- matformer_base_intermediate_size recorded for auto-detection
- sliced safetensors (single-shard only)
- tokenizer and auxiliary json files copied over
- matformer_manifest.json in the source directory describing tier files
"""
from __future__ import annotations

import argparse
import json
import shutil
from hashlib import sha256
from os.path import relpath
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file

COMMON_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "*.py",
)
MANIFEST_NAME = "matformer_manifest.json"


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
            if hidden % divisor != 0:
                raise ValueError(
                    f"{name} hidden size {hidden} not divisible by divisor {divisor}"
                )
            width = hidden // divisor
            sliced[name] = tensor[:width, :].contiguous()
        elif name.endswith("mlp.down_proj.weight"):
            hidden = tensor.shape[1]
            if hidden % divisor != 0:
                raise ValueError(
                    f"{name} hidden size {hidden} not divisible by divisor {divisor}"
                )
            width = hidden // divisor
            sliced[name] = tensor[:, :width].contiguous()
        else:
            sliced[name] = tensor
    return sliced


def collect_weight_files(root: Path) -> list[Path]:
    weights = sorted(root.glob("*.safetensors"))
    weights += sorted(root.glob("*.safetensors.index.json"))
    return [p for p in weights if p.is_file()]


def collect_common_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in COMMON_PATTERNS:
        files.extend(sorted(root.glob(pattern)))
    return [p for p in files if p.is_file()]


def rel_to_manifest(path: Path, manifest_dir: Path) -> str:
    return Path(relpath(path, start=manifest_dir)).as_posix()


def file_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_manifest(
    manifest_dir: Path,
    base_intermediate_size: int,
    tiers: Iterable[int],
) -> None:
    manifest_path = manifest_dir / MANIFEST_NAME
    common_files = collect_common_files(manifest_dir)
    common_rel = [rel_to_manifest(path, manifest_dir) for path in common_files]

    tier_entries: list[dict[str, object]] = []
    sha_map: dict[str, str] = {}

    for path in common_files:
        sha_map[rel_to_manifest(path, manifest_dir)] = file_sha256(path)

    # Tier 0 entry (universal checkpoint)
    base_files = [manifest_dir / "config.json"] + collect_weight_files(manifest_dir)
    base_files = [p for p in base_files if p.is_file()]
    base_rel = [rel_to_manifest(path, manifest_dir) for path in base_files]
    for path in base_files:
        sha_map[rel_to_manifest(path, manifest_dir)] = file_sha256(path)
    tier_entries.append(
        {
            "tier": 0,
            "intermediate_size": base_intermediate_size,
            "files": base_rel,
        }
    )

    for tier in tiers:
        divisor = 1 << tier
        if base_intermediate_size % divisor != 0:
            raise ValueError(
                f"base intermediate_size {base_intermediate_size} not divisible by 2**{tier}"
            )
        tier_dir = manifest_dir.with_name(f"{manifest_dir.name}-tier{tier}")
        tier_config = tier_dir / "config.json"
        if not tier_config.is_file():
            raise FileNotFoundError(f"tier {tier} config.json missing at {tier_config}")
        tier_files = [tier_config] + collect_weight_files(tier_dir)
        tier_files = [p for p in tier_files if p.is_file()]
        tier_rel = [rel_to_manifest(path, manifest_dir) for path in tier_files]
        for path in tier_files:
            sha_map[rel_to_manifest(path, manifest_dir)] = file_sha256(path)
        tier_entries.append(
            {
                "tier": tier,
                "intermediate_size": base_intermediate_size // divisor,
                "files": tier_rel,
            }
        )

    manifest = {
        "schema_version": 1,
        "matformer_base_intermediate_size": base_intermediate_size,
        "common_files": common_rel,
        "tiers": tier_entries,
        "sha256": sha_map,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


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
    if config.get("matformer_tier", 0) not in (0, None):
        raise ValueError("source checkpoint appears tier-sliced; expected universal checkpoint")
    divisor = 1 << tier
    base_intermediate_size = config.get(
        "matformer_base_intermediate_size", config["intermediate_size"]
    )
    if base_intermediate_size % divisor != 0:
        raise ValueError(
            f"base intermediate_size {base_intermediate_size} not divisible by 2**{tier}"
        )
    config["intermediate_size"] = base_intermediate_size // divisor
    config["matformer_tier"] = tier
    config["matformer_base_intermediate_size"] = base_intermediate_size
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

    config = json.loads((src / "config.json").read_text())
    if "intermediate_size" not in config:
        raise ValueError("config.json missing intermediate_size")
    base_intermediate_size = config.get(
        "matformer_base_intermediate_size", config["intermediate_size"]
    )

    for tier in tiers:
        dst = export_tier(src, tier)
        print(f"[export] wrote {dst}")

    write_manifest(src, base_intermediate_size, tiers)
    print(f"[export] wrote {src / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
