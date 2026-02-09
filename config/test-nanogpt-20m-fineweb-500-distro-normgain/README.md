# NanoGPT 20M FineWeb (500-step, DisTrO, norm tier gain) local-test config

Same as `config/test-nanogpt-20m-fineweb-500-distro`, but uses a checkpoint config that enables
MatFormer tier-conditioned RMSNorm gain (`matformer_stabilization.norm_tier_gain`).

## Run (local-testnet, 2 clients: tiers 0 and 1)

```bash
source scripts/psyche-env.sh
cargo build --release -p psyche-centralized-server -p psyche-centralized-client

cargo run -p psyche-centralized-local-testnet -- start \
  --headless --headless-exit-after-secs 3600 \
  --num-clients 2 \
  --config-path config/test-nanogpt-20m-fineweb-500-distro-normgain \
  --client-matformer-tiers 0,1 \
  --client-device cpu \
  --server-port 21011 \
  --tui=false \
  --write-log
```
