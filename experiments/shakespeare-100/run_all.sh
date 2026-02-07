#!/bin/bash
# TinyShakespeare 100-step heterogeneous training experiments
# Run from psyche root directory

set -e
cd /Users/progyan/psyche
source .venv/bin/activate

export LIBTORCH_BYPASS_VERSION_CHECK=1
export DYLD_LIBRARY_PATH="./.venv/lib/python3.12/site-packages/torch/lib"

CONFIG="./config/shakespeare-100-run"
OUTDIR="./experiments/shakespeare-100"

echo "=========================================="
echo "TinyShakespeare 100-step Experiments"
echo "=========================================="

# Experiment 1: (0,0,1,1) - 2 tier-0, 2 tier-1
echo ""
echo "[Exp1] Running (0,0,1,1) - 2 tier-0, 2 tier-1..."
cargo run --release -p psyche-centralized-local-testnet -- start \
  --headless \
  --num-clients 4 \
  --config-path "$CONFIG" \
  --client-matformer-tiers 0,0,1,1 \
  --client-matformer-helper-fractions 0,0,0,0 \
  --tui false \
  2>&1 | tee "$OUTDIR/exp1_0011.log"

# Experiment 2: (0,1,1,1) - 1 tier-0, 3 tier-1
echo ""
echo "[Exp2] Running (0,1,1,1) - 1 tier-0, 3 tier-1..."
cargo run --release -p psyche-centralized-local-testnet -- start \
  --headless \
  --num-clients 4 \
  --config-path "$CONFIG" \
  --client-matformer-tiers 0,1,1,1 \
  --client-matformer-helper-fractions 0,0,0,0 \
  --tui false \
  2>&1 | tee "$OUTDIR/exp2_0111.log"

# Experiment 3: (0,0,0,1) - 3 tier-0, 1 tier-1
echo ""
echo "[Exp3] Running (0,0,0,1) - 3 tier-0, 1 tier-1..."
cargo run --release -p psyche-centralized-local-testnet -- start \
  --headless \
  --num-clients 4 \
  --config-path "$CONFIG" \
  --client-matformer-tiers 0,0,0,1 \
  --client-matformer-helper-fractions 0,0,0,0 \
  --tui false \
  2>&1 | tee "$OUTDIR/exp3_0001.log"

# Experiment 4: (0) - 1 tier-0 only (baseline, 1/4 data throughput)
echo ""
echo "[Exp4] Running (0) - 1 tier-0 only (1/4 data throughput baseline)..."
cargo run --release -p psyche-centralized-local-testnet -- start \
  --headless \
  --num-clients 1 \
  --config-path "$CONFIG" \
  --client-matformer-tiers 0 \
  --client-matformer-helper-fractions 0 \
  --tui false \
  2>&1 | tee "$OUTDIR/exp4_solo_0.log"

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Logs in: $OUTDIR"
echo "=========================================="
