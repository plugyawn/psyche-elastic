#!/bin/bash
# Experiment: L-only (4 tier-0 clients) with batch=64 for VRAM analysis
# Run this on the L server

set -e

EXPERIMENT="L-only-batch64"
DURATION=600  # 10 minutes

echo "=== Starting $EXPERIMENT experiment ==="
date

# Clean up all servers
pkill -9 psyche 2>/dev/null || true
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
sleep 3

cd /root/psyche
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib

# Create log directory
LOG_DIR="/tmp/exp-$EXPERIMENT-$(date +%Y%m%d-%H%M%S)"
mkdir -p $LOG_DIR
echo "Logging to: $LOG_DIR"

# Start VRAM monitors
nohup bash -c "while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done" > $LOG_DIR/vram-L.csv 2>&1 &
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-$EXPERIMENT-A6000.csv 2>&1 &" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-$EXPERIMENT-S1.csv 2>&1 &" 2>/dev/null || true

# Start coordinator
echo "Starting coordinator..."
nohup ./target/release/psyche-centralized-server run \
    --state /opt/psyche/config/exp-L-only.toml \
    --server-port 17892 \
    --tui false \
    --logs console > $LOG_DIR/coordinator.log 2>&1 &
sleep 5

# Start 4 tier-0 clients distributed across servers
echo "Starting tier-0 on L (local)..."
nohup ./target/release/psyche-centralized-client train \
    --run-id exp-L-only \
    --server-addr 127.0.0.1:17892 \
    --device cuda:0 \
    --matformer-tier 0 \
    --logs console > $LOG_DIR/client-t0-L.log 2>&1 &

echo "Starting tier-0 on A6000 (2 GPUs)..."
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
nohup /opt/psyche-centralized-client train --run-id exp-L-only --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 0 --logs console > /tmp/client-$EXPERIMENT-t0-A0.log 2>&1 &
nohup /opt/psyche-centralized-client train --run-id exp-L-only --server-addr 216.81.248.128:17892 --device cuda:1 --matformer-tier 0 --logs console > /tmp/client-$EXPERIMENT-t0-A1.log 2>&1 &
" 2>/dev/null || true

echo "Starting tier-0 on S1..."
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib
nohup /opt/psyche-centralized-client train --run-id exp-L-only --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 0 --logs console > /tmp/client-$EXPERIMENT-t0-S1.log 2>&1 &
" 2>/dev/null || true

echo "All 4 tier-0 clients started at $(date)"
echo "Waiting for $DURATION seconds..."
sleep $DURATION

echo ""
echo "=== $EXPERIMENT Final Results ==="

# Extract per-step losses for plotting
echo "Extracting per-step losses..."
grep 'received_witness_metadata.*matformer_tier=0' $LOG_DIR/coordinator.log | \
    sed 's/\x1b\[[0-9;]*m//g' | \
    grep -oE 'step=[0-9]+.*loss=[0-9.]+' | \
    sed 's/step=//;s/.*loss=/,/' > $LOG_DIR/losses-L.csv

echo "step,loss" > $LOG_DIR/losses-L-header.csv
cat $LOG_DIR/losses-L.csv >> $LOG_DIR/losses-L-header.csv
mv $LOG_DIR/losses-L-header.csv $LOG_DIR/losses-L.csv

# Summary
TOTAL_STEPS=$(wc -l < $LOG_DIR/losses-L.csv)
echo "Total L steps: $((TOTAL_STEPS - 1))"  # exclude header
echo ""
echo "Last 10 losses:"
tail -10 $LOG_DIR/losses-L.csv
echo ""

# Collect VRAM logs from remote
scp -i /root/.ssh/id_rsa ubuntu@64.247.196.12:/tmp/vram-$EXPERIMENT-A6000.csv $LOG_DIR/ 2>/dev/null || true
scp -i /root/.ssh/id_rsa -P 10014 root@203.57.40.132:/tmp/vram-$EXPERIMENT-S1.csv $LOG_DIR/ 2>/dev/null || true

# VRAM summary
echo "VRAM Usage Summary:"
echo "L server (tier-0):"
awk -F',' '{sum+=$2; count++} END {print "  Average: " sum/count " MiB"}' $LOG_DIR/vram-L.csv 2>/dev/null || echo "  No data"
echo "A6000 (tier-0):"
awk -F',' '{sum+=$2; count++} END {print "  Average: " sum/count " MiB"}' $LOG_DIR/vram-$EXPERIMENT-A6000.csv 2>/dev/null || echo "  No data"

# Stop everything
pkill -9 psyche 2>/dev/null || true
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
pkill -f 'while true.*nvidia-smi' 2>/dev/null || true

echo ""
echo "=== Experiment Complete ==="
echo "Logs saved to: $LOG_DIR"
date
