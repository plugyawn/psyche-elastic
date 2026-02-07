#!/bin/bash
# Experiment: LSSS-same (1 tier-0 + 3 tier-2, same batch via PSYCHE_FORCE_SAME_BATCH=1)
# Run this on the L server after LSSS-diff completes

set -e

echo "=== Starting LSSS-same experiment ==="
date

# Clean up all servers
pkill -9 psyche 2>/dev/null || true
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
sleep 3

cd /root/psyche
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib

# Start VRAM monitors
nohup bash -c 'while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 5; done' > /tmp/vram-LSSS-same-L.log 2>&1 &
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-LSSS-same-A6000.log 2>&1 &" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-LSSS-same-S1.log 2>&1 &" 2>/dev/null || true

# Start coordinator
echo "Starting coordinator..."
nohup ./target/release/psyche-centralized-server run --state /opt/psyche/config/exp-LSSS-same.toml --server-port 17892 --tui false --logs console > /tmp/coord-LSSS-same.log 2>&1 &
sleep 5

# Start tier-0 client on L WITH SAME_BATCH
echo "Starting tier-0 on L (SAME_BATCH)..."
export PSYCHE_FORCE_SAME_BATCH=1
nohup ./target/release/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 127.0.0.1:17892 --device cuda:0 --matformer-tier 0 --logs console > /tmp/client-LSSS-same-t0.log 2>&1 &

# Start tier-2 clients on A6000 (2 GPUs) WITH SAME_BATCH
echo "Starting tier-2 on A6000 (SAME_BATCH)..."
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
export PSYCHE_FORCE_SAME_BATCH=1
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 2 --logs console > /tmp/client-LSSS-same-t2-0.log 2>&1 &
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 216.81.248.128:17892 --device cuda:1 --matformer-tier 2 --logs console > /tmp/client-LSSS-same-t2-1.log 2>&1 &
" 2>/dev/null || true

# Start tier-2 client on S1 WITH SAME_BATCH
echo "Starting tier-2 on S1 (SAME_BATCH)..."
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib
export PSYCHE_FORCE_SAME_BATCH=1
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 2 --logs console > /tmp/client-LSSS-same-t2.log 2>&1 &
" 2>/dev/null || true

echo "All clients started at $(date)"
echo "Waiting for 10 minutes..."
sleep 600

echo "=== LSSS-same Final Results ==="
echo "L tier-0 steps:"
grep 'received_witness_metadata.*matformer_tier=0' /tmp/coord-LSSS-same.log | wc -l
echo "Last 5 L tier-0 losses:"
grep 'received_witness_metadata.*matformer_tier=0' /tmp/coord-LSSS-same.log | tail -5 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE 'step=[0-9]+.*loss=[0-9.]+'
echo ""
echo "S tier-2 steps:"
grep 'received_witness_metadata.*matformer_tier=2' /tmp/coord-LSSS-same.log | wc -l

# Stop everything
pkill -9 psyche 2>/dev/null || true
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
pkill -f 'while true.*nvidia-smi' 2>/dev/null || true

echo "=== Experiment Complete ==="
date
