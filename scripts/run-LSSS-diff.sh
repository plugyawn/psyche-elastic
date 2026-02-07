#!/bin/bash
# Experiment: LSSS-diff (1 tier-0 + 3 tier-2, different batches)
# Run this on the L server after L-only completes

set -e

echo "=== Starting LSSS-diff experiment ==="
date

# Clean up all servers
pkill -9 psyche 2>/dev/null || true
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
sleep 3

cd /root/psyche
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib

# Start VRAM monitors
nohup bash -c 'while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 5; done' > /tmp/vram-LSSS-diff-L.log 2>&1 &
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-LSSS-diff-A6000.log 2>&1 &" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-LSSS-diff-S1.log 2>&1 &" 2>/dev/null || true

# Start coordinator
echo "Starting coordinator..."
nohup ./target/release/psyche-centralized-server run --state /opt/psyche/config/exp-LSSS-diff.toml --server-port 17892 --tui false --logs console > /tmp/coord-LSSS-diff.log 2>&1 &
sleep 5

# Start tier-0 client on L
echo "Starting tier-0 on L..."
nohup ./target/release/psyche-centralized-client train --run-id exp-LSSS-diff --server-addr 127.0.0.1:17892 --device cuda:0 --matformer-tier 0 --logs console > /tmp/client-LSSS-diff-t0.log 2>&1 &

# Start tier-2 clients on A6000 (2 GPUs)
echo "Starting tier-2 on A6000..."
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-diff --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 2 --logs console > /tmp/client-LSSS-diff-t2-0.log 2>&1 &
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-diff --server-addr 216.81.248.128:17892 --device cuda:1 --matformer-tier 2 --logs console > /tmp/client-LSSS-diff-t2-1.log 2>&1 &
" 2>/dev/null || true

# Start tier-2 client on S1
echo "Starting tier-2 on S1..."
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-diff --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 2 --logs console > /tmp/client-LSSS-diff-t2.log 2>&1 &
" 2>/dev/null || true

echo "All clients started at $(date)"
echo "Waiting for 10 minutes..."
sleep 600

echo "=== LSSS-diff Final Results ==="
echo "L tier-0 steps:"
grep 'received_witness_metadata.*matformer_tier=0' /tmp/coord-LSSS-diff.log | wc -l
echo "Last 5 L tier-0 losses:"
grep 'received_witness_metadata.*matformer_tier=0' /tmp/coord-LSSS-diff.log | tail -5 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE 'step=[0-9]+.*loss=[0-9.]+'
echo ""
echo "S tier-2 steps:"
grep 'received_witness_metadata.*matformer_tier=2' /tmp/coord-LSSS-diff.log | wc -l

# Stop everything
pkill -9 psyche 2>/dev/null || true
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
pkill -f 'while true.*nvidia-smi' 2>/dev/null || true

echo "=== Experiment Complete ==="
date
