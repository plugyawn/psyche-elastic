#!/bin/bash
# Local testnet runner for Psyche heterogeneous training tests
# Usage: ./scripts/run_local_testnet.sh [duration_secs] [num_clients] [config_path]

DURATION=${1:-30}
NUM_CLIENTS=${2:-2}
CONFIG_PATH=${3:-./config/test}
LOG_FILE="logs/testnet-$(date +%Y%m%d-%H%M%S).log"

mkdir -p logs

echo "Starting local testnet..."
echo "  Duration: ${DURATION}s"
echo "  Clients: ${NUM_CLIENTS}"
echo "  Config: ${CONFIG_PATH}"
echo "  Log: ${LOG_FILE}"

source .venv/bin/activate

LIBTORCH_BYPASS_VERSION_CHECK=1 \
DYLD_LIBRARY_PATH=./.venv/lib/python3.12/site-packages/torch/lib \
cargo run --release -p psyche-centralized-local-testnet -- start \
  --headless \
  --headless-exit-after-secs "${DURATION}" \
  --num-clients "${NUM_CLIENTS}" \
  --config-path "${CONFIG_PATH}" \
  --tui false \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "Log saved to: ${LOG_FILE}"
