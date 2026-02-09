# Tiny LLaMA (local) smoke-test config

This config uses a tiny local Hugging Face-style LLaMA checkpoint (under `./checkpoints/tiny-llama-local`) and a dummy data provider.

## Create the checkpoint

```bash
python3 scripts/make_tiny_llama_checkpoint.py --out checkpoints/tiny-llama-local
```

## Run (single machine)

```bash
bash scripts/psyche-centralized-server.sh run --state config/test-tiny-llama/state.toml --server-port 20000 --tui false --logs json --write-log logs/server.jsonl
```

In another terminal:

```bash
RAW_IDENTITY_SECRET_KEY="$(printf '%064x' 1)" RUST_LOG="warn,psyche_client=info" \
  bash scripts/psyche-centralized-client.sh train --run-id test-tiny-llama --server-addr localhost:20000 --logs json --device mps --matformer-tier 0 --iroh-discovery local --iroh-relay disabled --write-log logs/client0.jsonl
```

## MatFormer tiers

The checkpoint defaults to `intermediate_size=256`, so:
- tier `0` => 256
- tier `1` => 128
- tier `2` => 64
