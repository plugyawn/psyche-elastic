#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
import sys
import urllib.request
from pathlib import Path

FALLBACK_TINY_SHAKESPEARE = """\
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thy self thy foe, to thy sweet self too cruel:
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And tender churl mak'st waste in niggarding:
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.
"""


def download_bytes(url: str, timeout_secs: float) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "psyche/prepare_tinyshakespeare (python urllib)"},
    )
    with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
        return resp.read()


def build_byte_vocab(data: bytes, vocab_size: int) -> dict[int, int]:
    # 0..3 are reserved in the tiny-llama-local checkpoint (<unk>, <bos>, <eos>, <pad>).
    start_token_id = 4
    max_symbols = vocab_size - start_token_id
    uniq = sorted(set(data))
    if len(uniq) > max_symbols:
        raise ValueError(
            f"Text contains {len(uniq)} unique bytes, but vocab_size={vocab_size} "
            f"only leaves room for {max_symbols} non-special tokens."
        )
    return {b: (start_token_id + i) for i, b in enumerate(uniq)}


def encode_bytes(data: bytes, byte_to_token_id: dict[int, int]) -> list[int]:
    unk = 0
    return [byte_to_token_id.get(b, unk) for b in data]


def write_u16_le(tokens: list[int], path: Path) -> None:
    # LocalDataProvider expects little-endian u16 when token_size=TwoBytes.
    out = bytearray()
    for t in tokens:
        if not (0 <= t <= 0xFFFF):
            raise ValueError(f"Token out of range for u16: {t}")
        out += int(t).to_bytes(2, byteorder="little", signed=False)
    path.write_bytes(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a tiny tokenized 'Shakespeare-like' dataset as a single .bin token stream.\n"
            "This is meant to be consumed by Psyche's LocalDataProvider (token_size=TwoBytes)."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/tinyshakespeare-bin"),
        help="Output directory for dataset (default: data/tinyshakespeare-bin).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        help="URL to fetch tiny Shakespeare text from.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help=(
            "If the download fails, fall back to a tiny embedded sample text. "
            "Not recommended; primarily for offline smoke tests."
        ),
    )
    parser.add_argument(
        "--timeout-secs",
        type=float,
        default=10.0,
        help="Download timeout in seconds.",
    )
    parser.add_argument(
        "--repeat-to-chars",
        type=int,
        default=1_000_000,
        help="Repeat the text until at least this many bytes (default: 1,000,000).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
        help="Vocabulary size (must match the checkpoint's vocab_size).",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw: bytes
    try:
        raw = download_bytes(args.url, args.timeout_secs)
        (out_dir / "source.txt").write_bytes(raw)
        source = {"type": "url", "url": args.url}
    except Exception as e:
        if not args.allow_fallback:
            print(f"[dataset] download failed: {e}", file=sys.stderr)
            print(
                "[dataset] Refusing to continue without --allow-fallback to avoid silently training on tiny fallback text.",
                file=sys.stderr,
            )
            return 2
        raw = FALLBACK_TINY_SHAKESPEARE.encode("utf-8")
        (out_dir / "source.txt").write_bytes(raw)
        source = {"type": "fallback", "error": str(e)}
        (out_dir / "FALLBACK_USED").write_text(
            "Download failed; generated dataset from embedded fallback sample.\n"
        )
        print(
            f"[dataset] WARNING: download failed; using fallback text because --allow-fallback was set: {e}",
            file=sys.stderr,
        )
        print(
            "[dataset] NOTE: Psyche will refuse to train on this dataset unless PSYCHE_ALLOW_FALLBACK_DATASET=1 is set.",
            file=sys.stderr,
        )

    if args.repeat_to_chars and len(raw) < args.repeat_to_chars:
        # Repeat with a newline separator to avoid accidental token concatenation.
        sep = b"\n"
        reps = (args.repeat_to_chars + len(raw) - 1) // max(1, len(raw))
        raw = (raw + sep) * reps
        raw = raw[: args.repeat_to_chars]

    byte_to_token_id = build_byte_vocab(raw, args.vocab_size)
    tokens = encode_bytes(raw, byte_to_token_id)

    # One contiguous stream; LocalDataProvider will slice into overlapping (seq_len+1) sequences.
    train_bin = out_dir / "train.bin"
    write_u16_le(tokens, train_bin)

    sha256 = hashlib.sha256(raw).hexdigest()
    meta = {
        "format": "psyche-local-binary-v1",
        "token_size_in_bytes": 2,
        "vocab_size": args.vocab_size,
        "num_bytes": len(raw),
        "sha256": sha256,
        "num_tokens": len(tokens),
        "unique_bytes": len(byte_to_token_id),
        "byte_to_token_id": {str(k): v for k, v in byte_to_token_id.items()},
        "source": source,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[dataset] wrote {train_bin} ({train_bin.stat().st_size} bytes)")
    print(f"[dataset] wrote {out_dir / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
