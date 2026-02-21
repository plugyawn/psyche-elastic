#!/bin/bash
# Train NanoGPT and run inference on the trained model
set -e

# Configuration
CONFIG_PATH="${1:-./config/nanogpt-20m-run}"
CHECKPOINT_DIR="./checkpoints/nanogpt-trained"
TRAINING_STEPS="${2:-100}"
TIMEOUT_SECS="${3:-300}"

echo "==================================================="
echo "NanoGPT Training + Inference Pipeline"
echo "==================================================="
echo "Config: $CONFIG_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Steps: $TRAINING_STEPS"
echo "Timeout: ${TIMEOUT_SECS}s"
echo "==================================================="

# Set up environment
export DYLD_LIBRARY_PATH="${PWD}/.venv/lib/python3.12/site-packages/torch/lib"

# Clean up any existing processes
pkill -f "psyche-centralized" 2>/dev/null || true
sleep 2

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Start server
echo ""
echo "[1/4] Starting server..."
cargo run --release -p psyche-centralized-server -- run \
    --state "${CONFIG_PATH}/state.toml" \
    --server-port 20000 \
    --tui false &
SERVER_PID=$!
sleep 3

# Start client with checkpoint saving
echo ""
echo "[2/4] Starting client with checkpoint saving..."
RUN_ID=$(grep 'run_id' "${CONFIG_PATH}/state.toml" | head -1 | cut -d'"' -f2)

timeout ${TIMEOUT_SECS} cargo run --release -p psyche-centralized-client -- train \
    --run-id "$RUN_ID" \
    --server-addr "localhost:20000" \
    --logs console \
    --device auto \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --keep-steps 1 \
    --matformer-tier 0 \
    --iroh-discovery local \
    --iroh-relay disabled \
    2>&1 | tee /tmp/training_output.log || true

echo ""
echo "[3/4] Training complete. Stopping server..."
kill $SERVER_PID 2>/dev/null || true
sleep 2

# Find the latest checkpoint
echo ""
echo "[4/4] Looking for saved checkpoints..."
LATEST_CHECKPOINT=$(ls -td ${CHECKPOINT_DIR}/${RUN_ID}-step* 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in $CHECKPOINT_DIR"
    echo "Contents:"
    ls -la "$CHECKPOINT_DIR"
    exit 1
fi

echo "Found checkpoint: $LATEST_CHECKPOINT"
echo ""
echo "Checkpoint contents:"
ls -la "$LATEST_CHECKPOINT"

echo ""
echo "==================================================="
echo "Training complete! Checkpoint saved to:"
echo "$LATEST_CHECKPOINT"
echo ""
echo "To run inference:"
echo "python scripts/inference_nanogpt.py --model $LATEST_CHECKPOINT"
echo "==================================================="
