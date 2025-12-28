#!/usr/bin/env python3
"""
Download pre-tokenized FineWeb-10B dataset from HuggingFace.

This dataset contains GPT-2 BPE tokenized text from FineWeb, used by modded-nanogpt.
Each file contains ~100M tokens as raw uint16 little-endian values.

Dataset: https://huggingface.co/datasets/kjj0/fineweb10B-gpt2

Usage:
    python scripts/download_fineweb10B.py                    # Download all (103 train + 1 val)
    python scripts/download_fineweb10B.py --num-chunks 1     # Download just validation + 1 train chunk
    python scripts/download_fineweb10B.py --val-only         # Download only validation set
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


REPO_ID = "kjj0/fineweb10B-gpt2"
REPO_TYPE = "dataset"

# File naming convention from modded-nanogpt
VAL_FILE = "fineweb_val_000000.bin"
TRAIN_FILE_TEMPLATE = "fineweb_train_{:06d}.bin"
NUM_TRAIN_CHUNKS = 103  # 000001 to 000103
TOKENS_PER_CHUNK = 100_000_000  # ~100M tokens per file


def download_file(filename: str, local_dir: Path, verbose: bool = True) -> Path:
    """Download a single file from the HuggingFace dataset."""
    if verbose:
        print(f"Downloading {filename}...", end=" ", flush=True)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    if verbose:
        size_mb = Path(local_path).stat().st_size / (1024 * 1024)
        print(f"done ({size_mb:.1f} MB)")

    return Path(local_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download FineWeb-10B GPT-2 tokenized dataset from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/fineweb10B"),
        help="Output directory (default: data/fineweb10B)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=NUM_TRAIN_CHUNKS,
        help=f"Number of training chunks to download, 1-{NUM_TRAIN_CHUNKS} (default: all {NUM_TRAIN_CHUNKS})",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Download only the validation file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List available files in the repository and exit",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # List files mode
    if args.list_files:
        print(f"Files in {REPO_ID}:")
        for f in sorted(list_repo_files(REPO_ID, repo_type=REPO_TYPE)):
            print(f"  {f}")
        return 0

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []
    total_tokens = 0

    # Always download validation file
    if verbose:
        print(f"Downloading to {out_dir.resolve()}")
        print()

    val_path = download_file(VAL_FILE, out_dir, verbose)
    downloaded_files.append(VAL_FILE)
    # Validation set has ~10M tokens (smaller than train chunks)
    val_tokens = val_path.stat().st_size // 2  # uint16 = 2 bytes
    total_tokens += val_tokens

    if verbose:
        print(f"  Validation tokens: {val_tokens:,}")
        print()

    # Download training chunks
    if not args.val_only:
        num_chunks = min(args.num_chunks, NUM_TRAIN_CHUNKS)
        if verbose:
            print(f"Downloading {num_chunks} training chunk(s)...")
            print()

        for i in range(1, num_chunks + 1):
            filename = TRAIN_FILE_TEMPLATE.format(i)
            train_path = download_file(filename, out_dir, verbose)
            downloaded_files.append(filename)
            chunk_tokens = train_path.stat().st_size // 2
            total_tokens += chunk_tokens

    # Write metadata
    meta = {
        "format": "fineweb10B-gpt2",
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "tokenizer": "gpt2",
        "vocab_size": 50257,
        "token_size_in_bytes": 2,
        "files": downloaded_files,
        "total_tokens": total_tokens,
        "notes": "Raw uint16 little-endian GPT-2 BPE token IDs",
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    if verbose:
        print()
        print(f"Download complete!")
        print(f"  Directory: {out_dir.resolve()}")
        print(f"  Files: {len(downloaded_files)}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Metadata: {meta_path}")
        print()
        print("To use with Psyche, set data_location to this directory.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
