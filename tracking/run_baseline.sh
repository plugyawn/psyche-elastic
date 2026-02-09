#!/bin/bash
export DYLD_LIBRARY_PATH=/Users/progyan/psyche/.venv/lib/python3.12/site-packages/torch/lib:${DYLD_LIBRARY_PATH:-}
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
cd /Users/progyan/psyche
exec /Users/progyan/psyche/target/release/psyche-centralized-local-testnet start \
  --headless --headless-exit-after-secs 1200 \
  --num-clients 4 \
  --config-path /Users/progyan/psyche/config/test-fineweb-sssl-40 \
  --client-matformer-tiers 0,0,0,0 \
  --client-matformer-helper-fractions 0,0,0,0 \
  --tui false
