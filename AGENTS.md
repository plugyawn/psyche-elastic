# Repository Guidelines

## Scope And Sources Of Truth

Start with:
- `CLAUDE.md` for practical build/run/test commands.
- `psyche-book/` (mdBook) for system design and detailed how-tos.
- `architectures/centralized/README.md` and `docker/README.md` for operator-level workflows.
- If you touch Solana account layouts, read `architectures/decentralized/solana-coordinator/BREAKING-CHANGES.md` before coding.

This repo is a Rust workspace that implements distributed training over the internet in two deployment modes:
- **Centralized (off-chain)**: a TCP server ticks the coordinator and clients train + gossip results.
- **Decentralized (on-chain)**: Solana programs hold coordinator state and clients tick via RPC.

## Setup Notes (Nix, macOS, Linux)

- Nix is the "golden path": it pins toolchains and large deps (Torch, Anchor, etc). See `psyche-book/src/development/setup.md`.
- Optional speedup: enable the Garnix binary cache (also described in `psyche-book/src/development/setup.md`).
- Platform differences:
  - Linux: CUDA/NCCL paths are expected for multi-GPU features.
  - macOS: MPS is supported; data-parallel features may be disabled depending on config/build.

If you are not using Nix, prefer the wrapper `scripts/psyche-env.sh` so `tch-rs` can find libtorch and your build artifacts land in `.cargo-target/` (repo-local).

## Project Structure & Module Organization

- `architectures/centralized/`: `server/`, `client/`, `local-testnet/` (tmux/headless runner), and `testing/` integration tests.
- `architectures/decentralized/`: Solana programs (`solana-coordinator/`, `solana-authorizer/`, `solana-treasurer/`), `solana-client/`, and `testing/`.
- `shared/`: reusable crates (coordinator state machine, client state machine, networking, modeling, data providers, metrics, watcher, TUI).
- `python/`: PyO3-based extension + optional Python "sidecar" modeling path (see `psyche-book/src/development/python.md`).
- `config/`: run configs (typically `config/<name>/state.toml` plus optional `data.toml` for TCP data servers).
- `scripts/`: reproducible wrappers and utilities (macOS libtorch env, spawn clients, dataset prep, checkpoint slicing, stress tests).
- `docker/`, `telemetry/`: dockerized infra and local observability stack.
- `psyche-book/`: docs; generated CLI help lives in `psyche-book/generated/cli/`.

## Architecture Overview (Mental Model)

1. **Coordinator state machine** (`shared/coordinator/`): phases like `WaitingForMembers` -> `Warmup` -> `RoundTrain` -> `RoundWitness` -> `Cooldown`, plus epoch/round bookkeeping.
2. **Backend drives ticks**:
   - Centralized: `architectures/centralized/server/` owns the coordinator and broadcasts state over TCP.
   - Decentralized: Solana program stores the coordinator account; clients read via subscriptions and call `tick`.
3. **Clients execute rounds** (`shared/client/`):
   - Resolve run config + checkpoint, load model, set up a data provider.
   - For each round, deterministically select assigned batches from the coordinator seed.
   - Train locally via `shared/modeling/` and emit compressed updates (DisTrO) and commitments.
4. **P2P result propagation** (`shared/network/`): clients exchange results over iroh gossip/blob transport; witnesses attest to completeness.
5. **Model sharing** (`psyche-book/src/explain/model-sharing.md`): new joiners sync in `Warmup` via P2P checkpointing or a designated HuggingFace checkpointer.

Key "heterogeneous training" axis:
- **MatFormer** (nested FFN widths) lets clients run at different tiers (`--matformer-tier`) while sharing prefix weights. See `psyche-book/src/explain/matformer.md` and the verification checklist in `psyche-book/src/development/matformer-verification.md`.

## Build, Test, And Development Commands

### With Nix (preferred)

```bash
nix develop

just --list
just local-testnet -- --headless --headless-exit-after-secs 60 --num-clients 2 --config-path ./config/test --tui false

# CI-equivalent checks (Garnix + Nix checks)
just check            # if available in your `just` version
just nix check        # module form used elsewhere in this repo

nix fmt               # preferred formatter entrypoint (Rust + Nix)
just fmt              # deprecated wrapper (prints a warning); prefer `nix fmt`
```

### Without Nix (common on macOS)

Use the env wrapper to locate torch's `lib/` dir and set `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH`, plus a repo-local `CARGO_TARGET_DIR`:

```bash
# If you're already in bash:
source scripts/psyche-env.sh
cargo build --release -p psyche-centralized-server -p psyche-centralized-client

# If you're in zsh (macOS default) and don't want to switch shells:
bash -lc 'source scripts/psyche-env.sh && cargo build --release -p psyche-centralized-server -p psyche-centralized-client'
```

To create a pinned Python 3.12 runtime in `.venv/` (used by `tch-rs` via PyTorch libs):

```bash
bash scripts/bootstrap-python-runtime.sh
```

If `scripts/bootstrap-python-runtime.sh` is not present in your branch/checkout, see `psyche-book/src/development/setup.md` for the manual non-Nix setup steps.

### Docs (mdBook)

```bash
just serve_book         # dev server for docs
just build_book         # build into an output dir
nix build .#psyche-book  # Nix build of the book (outputs to ./result/)
```

If you change CLI flags or help text, regenerate the embedded CLI docs:
- `just generate_cli_docs` (writes `psyche-book/generated/cli/*.md`)

### Python Integration (Sidecar)

Python support is experimental (see `psyche-book/src/development/python.md`). The most reliable workflow is to run commands through the dev shell so the extension is rebuilt as needed:

```bash
nix develop .#dev-python --command cargo run --features python --example train -- --help
```

## Running Locally (Centralized / Off-Chain)

### Local testnet (server + N clients)

```bash
cargo run --release -p psyche-centralized-local-testnet -- start \
  --headless --headless-exit-after-secs 60 \
  --num-clients 3 \
  --config-path ./config/test \
  --client-matformer-tiers 0,1,2 \
  --tui false
```

Notes:
- Non-headless local-testnet uses tmux panes; headless mode is best for CI and quick smoke tests.
- Some local setups include a convenience wrapper `scripts/run_local_testnet.sh` for quick headless runs (expects `.venv/`).

For multi-machine (or when you want explicit logs), run server/clients directly via wrappers:

```bash
# Server
RUST_LOG="warn,psyche_centralized_server=info" \
  bash scripts/psyche-centralized-server.sh run \
    --state config/test-tiny-llama/state.toml \
    --server-port 20000 --tui false --logs json --write-log logs/server.jsonl

# Client
RAW_IDENTITY_SECRET_KEY="$(printf '%064x' 1)" \
RUST_LOG="warn,psyche_client=info,psyche_centralized_client=info" \
  bash scripts/psyche-centralized-client.sh train \
    --run-id test-tiny-llama \
    --server-addr <SERVER_IP>:20000 \
    --device mps \
    --matformer-tier 0 \
    --iroh-discovery local --iroh-relay disabled \
    --logs json --write-log logs/client0.jsonl
```

To spawn many clients per machine with stable identities/ports:
- Use `scripts/psyche-centralized-spawn-clients.sh` (writes per-client `logs/client-*.jsonl`).

## Running Locally (Decentralized / Solana)

- Build programs with Anchor, deploy to localnet/devnet, create a run, then start clients.
- Read `psyche-book/src/development/running-onchain.md` and `docker/README.md`.
- If you modify on-chain account layouts, you must preserve ABI or write migrations; see `architectures/decentralized/solana-coordinator/BREAKING-CHANGES.md`.

## Testing Guidelines

- Rust unit tests: `cargo test`
- Centralized integration tests: `just integration-test [test_name]`
- Decentralized integration tests: `just decentralized-integration-tests [test_name]`
- Decentralized integration tests with Python: `USE_PYTHON=1 just decentralized-integration-tests [test_name]`
- Decentralized chaos tests: `just decentralized-chaos-integration-test [test_name]`

Some decentralized tests require Docker images + Anchor builds:
- `just setup_test_infra` (or `just setup_python_test_infra`)
- `just run_test_infra <num_clients>`
- `just stop_test_infra`

## Observability (Telemetry + Metrics)

Local stack:

```bash
docker compose -f telemetry/docker-compose.yml up
```

Psyche uses "OLTP"/OpenTelemetry flags and env vars (note the transposed spelling in the CLI):
- Env: `OLTP_METRICS_URL`, `OLTP_TRACING_URL`, `OLTP_LOGS_URL`, `OLTP_AUTH_HEADER`
- Flags: `--oltp-metrics-url`, `--oltp-tracing-url`, `--oltp-logs-url`, `--oltp-auth-header`

For quick local runs, the `just local-testnet` recipe pre-populates local collector endpoints.

## Logs, Plots, And Experiment Artifacts

By convention:
- Run outputs go under `logs/` (gitignored).
- Checkpoints go under `checkpoints/` (gitignored).
- Local datasets go under `data/` (gitignored).

### What To Look For In Logs

Useful markers (grep-friendly):
- Client: `integration_test_log_marker=loaded_model`, `state_change`, `witness_elected`, `client_loss`
- Server: `received_witness_metadata`, `schema_hash_match`

Prefer structured logs in debugging runs:
- `--logs json --write-log logs/<name>.jsonl`

If you need deterministic identities or per-client metrics ports (especially when comparing runs):
- Set `RAW_IDENTITY_SECRET_KEY` (hex-encoded 32 bytes).
- Set `METRICS_LOCAL_PORT` (or use `scripts/psyche-centralized-spawn-clients.sh` which assigns ports).

If you need per-step loss comparisons across runs (avoid stepwise diffs from swapped batch assignments):
- Use `--aligned-batches` on the centralized server/local-testnet.
- Optionally pin `--aligned-batches-seed <u64>` so different `run_id`s still get identical batch schedules.

For gradient-level debugging:
- Client supports `--write-gradients-dir <dir>` to dump received/shared gradients.

### Plotting Loss Curves

Some branches/local workflows include ad-hoc plotting helpers (for example `scripts/plot_loss_curves.py`) that parse coordinator/client logs and write PNGs under `plots/`.
Treat `plots/` as local analysis output unless a PR explicitly intends to ship plots.

### DisTrO / Gradient Diagnostics

- Local testnet supports writing optimizer artifacts (for offline analysis) via `--write-distro-data <dir>`.
- Historical experiment notes and reproduction commands live in `logs/experiment-log.md`.

### Stress Tests And Fault Injection

There are convenience runners in `scripts/`:
- `scripts/stress_test.sh basic|latency|packet-loss|combined`
- `scripts/stress_test_matrix.sh` (runs a grid and writes `stress_test_results/<timestamp>/`)

Fault injection is plumbed via CLI flags on local-testnet/client (see `--help` for exact names):
- latency ranges, packet loss, bandwidth caps, and seeded reproducibility.

### Live Tracking Dashboard (Optional)

Some local workflows also include a small "baseline vs experiment" dashboard under `tracking/` (log-to-JSON conversion + a static HTML page). If present, it is useful for live comparisons during long runs.

## Checkpoints And MatFormer Tiers

- `config/test-tiny-llama/README.md` is the recommended "first run" smoke test.
- To generate a tiny local dataset for smoke tests, use:

```bash
python3 scripts/prepare_tinyshakespeare_bin_dataset.py --out-dir data/tinyshakespeare-bin
```

If `scripts/prepare_tinyshakespeare_bin_dataset.py` is not present in your branch/checkout, use any equivalent tokenization pipeline that matches the configured `token_size`/`vocab_size` for the run.

- To pre-slice a universal checkpoint into tiered directories (for smaller clients), use:

```bash
python3 scripts/export_matformer_tiers.py --src ./checkpoints/<ckpt> --tiers 1 2
```

- Helper mode is a common footgun for tiers > 0; most heterogeneous test commands disable it explicitly via `--client-matformer-helper-fractions 0,0,0,...`. See `psyche-book/src/development/matformer-verification.md`.

## Releases (Decentralized Client)

When cutting a new Psyche Solana client release, update both in lockstep (see `architectures/decentralized/RELEASES.md`):
- The Docker image tag in `docker.nix`
- `CLIENT_VERSION` in `architectures/decentralized/solana-client/src/app.rs`

## Coding Style & Naming Conventions

- Rust:
  - Format: `cargo fmt` (or `nix fmt`).
  - Lint: keep `cargo clippy` clean for touched code paths.
  - Naming: standard Rust conventions (`snake_case` fns/modules, `CamelCase` types).
- Nix: use `nix fmt` (repo formatter config lives in `nix/`).
- Shell: prefer `bash` + `set -euo pipefail` (match existing scripts in `scripts/`).

## Commit & Pull Request Guidelines

- Maintain a clean linear history (prefer rebase; avoid merge commits). See `psyche-book/src/development/contributing.md`.
- Keep commits meaningful and independently buildable; avoid "oops/fixup" commits in the final PR series.
- Commit subjects are short/imperative; optional scoped prefixes are common (for example `matformer:`).
- PRs should include:
  - What/why + risk notes (especially for consensus/aggregation logic, networking, and Solana account changes).
  - Exact commands run (tests, repro, benchmarks).
  - Logs/plots when changing training behavior (attach or link to generated artifacts and the config used).

## Security & Configuration Tips

- Don't commit credentials/keys. Use `agenix` for repo-managed secrets (see `psyche-book/src/development/agenix.md`) and keep `.env` files local (`config/client/.env` is intentionally excluded).
- Local dataset generation may create a sentinel `FALLBACK_USED`; the code refuses to train on fallback data unless `PSYCHE_ALLOW_FALLBACK_DATASET=1` is set (see `scripts/prepare_tinyshakespeare_bin_dataset.py`).
- When you need a local, reproducible build output directory for sharing binaries/logs, prefer `.cargo-target/` (set by `scripts/psyche-env.sh`) and bundle via `scripts/package-centralized-mac.sh` for macOS.
