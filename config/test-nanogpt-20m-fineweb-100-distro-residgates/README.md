# NanoGPT 20M FineWeb (100-step, DisTrO) smoke-test config

This config runs a small NanoGPT (~20M params) on local GPT-2-tokenized FineWeb data (`data/fineweb10B`)
using the DisTrO optimizer so multiple clients can contribute updates.

## Run (local-testnet, 2 clients: tiers 0 and 1)

```bash
source scripts/psyche-env.sh

# Ensure release binaries exist (headless local-testnet uses them):
cargo build --release -p psyche-centralized-server -p psyche-centralized-client

cargo run -p psyche-centralized-local-testnet -- start \
  --headless --headless-exit-after-secs 900 \
  --num-clients 2 \
  --config-path config/test-nanogpt-20m-fineweb-100-distro \
  --client-matformer-tiers 0,1 \
  --client-device cpu \
  --server-port 21010 \
  --tui=false \
  --write-log
```

Notes:
- `data/fineweb10B` is expected to contain `fineweb_val_000000.bin` (uint16 GPT-2 tokens).
- Distillation is *disabled* by default. Use local-testnet flags to enable it.

