# Psyche Centralized Server and Client

## Local Testing

You can use the `psyche-centralized-local-testnet` binary in `/architectures/centralized/local-testnet`, which automates the process of launching a centralized server and multiple clients using tmux.

### Prerequisites

`nix develop` OR

- tmux (unless using `--headless`)
- nvtop (optional; tmux mode only, skip with `--no-nvtop`)

### Usage

```
cargo run -p psyche-centralized-local-testnet -- --help
```

### Example Invocations

#### Headless smoke test (no tmux)

```bash
cargo run -p psyche-centralized-local-testnet -- start --headless --headless-exit-after-secs 10 --num-clients 2 --config-path ./config/test --tui false
```

#### Tiny LLaMA on MPS (macOS)

```bash
.venv/bin/python scripts/make_tiny_llama_checkpoint.py --out checkpoints/tiny-llama-local
cargo run -p psyche-centralized-local-testnet -- start --headless --headless-exit-after-secs 25 --num-clients 2 --config-path ./config/test-tiny-llama --tui false --client-device mps
```

#### Demo

```bash
cargo run -p psyche-centralized-local-testnet start --num-clients 3 --config-path ./config/llama2-20m-dolma-noverify-no-checkpointer --write-distro-data ./distro-data/llama2-20m-noverify --tui false
```

This command launches a server and 3 clients, using the configuration in `/path/to/config`, writing gradient data, and disabling the TUI for clients.

#### Heterogeneous MatFormer Training

Test with clients at different MatFormer tiers (different FFN widths):

```bash
cargo run -p psyche-centralized-local-testnet -- start \
    --headless \
    --headless-exit-after-secs 60 \
    --num-clients 3 \
    --config-path ./config/test \
    --client-matformer-tiers 0,1,2 \
    --tui false
```

This assigns:
- Client 1: tier 0 (full FFN width, 100%)
- Client 2: tier 1 (half FFN width, 50%)
- Client 3: tier 2 (quarter FFN width, 25%)

Useful for simulating heterogeneous hardware where devices have different VRAM capacities.

## Multi-Machine (macOS)

For running a centralized server + clients across multiple Macs, use the wrapper scripts in `scripts/` (they set up `LIBTORCH_USE_PYTORCH`, `PYO3_PYTHON`, and the torch dylib path via `scripts/psyche-env.sh`).

To build a copyable bundle (release binaries + scripts), run:

```bash
bash scripts/package-centralized-mac.sh
```

### Start a server (pick one Mac)

```bash
RUST_LOG="warn,psyche_centralized_server=info" \
  bash scripts/psyche-centralized-server.sh run \
    --state config/test-tiny-llama/state.toml \
    --server-port 20000 \
    --tui false \
    --logs json \
    --write-log logs/server.jsonl
```

### Start clients (on each Mac)

```bash
RUST_LOG="warn,psyche_client=info,psyche_centralized_client=info" \
  bash scripts/psyche-centralized-client.sh train \
    --run-id test-tiny-llama \
    --server-addr <SERVER_LAN_IP>:20000 \
    --logs json \
    --write-log logs/client.jsonl \
    --device mps \
    --matformer-tier 2 \
    --iroh-discovery local \
    --iroh-relay disabled
```

### Spawn multiple clients per machine (optional)

```bash
bash scripts/psyche-centralized-spawn-clients.sh \
  --run-id test-tiny-llama \
  --server-addr <SERVER_LAN_IP>:20000 \
  --num-clients 2 \
  --device mps \
  --tiers 0,2 \
  --logs json \
  --rust-log "warn,psyche_client=info"
```

### What to look for in logs

- Clients log `loaded_model` with `matformer_tier` and (for LLaMA) `intermediate_size_active`.
- Clients log `client_loss` with `matformer_tier`, `trained_batches`, and `loss` (when training happened that round).
- Server logs `received_witness_metadata` with `loss`/`tokens_per_sec` from witnesses (a quick sanity signal for convergence/health).

#### Testing against node crashes

```bash
just psyche-centralized-local-testnet --num-clients 3 --config-path ./config/kill-test-short-epoch-checkpoint/ --random-kill-num 1 --allowed-to-kill 2,3 --first-client-checkpoint bug-free-chainsaw/tiny-local-20m --hf-token xxxxxxxxxxxxx --write-log
```

This command launches a server with 3 clients, using the config "kill-test-short-epoch-checkpoint".
It randomly kills either client 2 or 3 every 120 seconds (the default interval).
Client 1 is doing checkpointing, so we don't kill it.
Client 1 is set to checkpoint to the HF repo `bug-free-chainsaw/tiny-local-20m`, and we pass an HF token for auth. We also enable logging to disk.
