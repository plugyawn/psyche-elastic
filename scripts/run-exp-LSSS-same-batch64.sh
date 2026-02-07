#!/bin/bash
# Experiment: LSSS-same (1 tier-0 + 3 tier-2, same batch) with batch=64 for VRAM analysis
# Run this on the L server

set -e

EXPERIMENT="LSSS-same-batch64"
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
    --state /opt/psyche/config/exp-LSSS-same.toml \
    --server-port 17892 \
    --tui false \
    --logs console > $LOG_DIR/coordinator.log 2>&1 &
sleep 5

# Start tier-0 client on L WITH SAME_BATCH
echo "Starting tier-0 on L (SAME_BATCH)..."
export PSYCHE_FORCE_SAME_BATCH=1
nohup ./target/release/psyche-centralized-client train \
    --run-id exp-LSSS-same \
    --server-addr 127.0.0.1:17892 \
    --device cuda:0 \
    --matformer-tier 0 \
    --logs console > $LOG_DIR/client-t0-L.log 2>&1 &

# Start tier-2 clients on A6000 (2 GPUs) WITH SAME_BATCH
echo "Starting tier-2 on A6000 (SAME_BATCH)..."
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
export PSYCHE_FORCE_SAME_BATCH=1
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 2 --logs console > /tmp/client-$EXPERIMENT-t2-A0.log 2>&1 &
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 216.81.248.128:17892 --device cuda:1 --matformer-tier 2 --logs console > /tmp/client-$EXPERIMENT-t2-A1.log 2>&1 &
" 2>/dev/null || true

# Start tier-2 client on S1 WITH SAME_BATCH
echo "Starting tier-2 on S1 (SAME_BATCH)..."
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib
export PSYCHE_FORCE_SAME_BATCH=1
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 2 --logs console > /tmp/client-$EXPERIMENT-t2-S1.log 2>&1 &
" 2>/dev/null || true

echo "1 tier-0 + 3 tier-2 clients started at $(date)"
echo "Waiting for $DURATION seconds..."
sleep $DURATION

echo ""
echo "=== $EXPERIMENT Final Results ==="

# Extract per-step losses for L (tier-0) and S (tier-2) separately
echo "Extracting per-step losses..."

# Tier-0 (L) losses
grep 'received_witness_metadata.*matformer_tier=0' $LOG_DIR/coordinator.log | \
    sed 's/\x1b\[[0-9;]*m//g' | \
    grep -oE 'step=[0-9]+.*loss=[0-9.]+' | \
    sed 's/step=//;s/.*loss=/,/' > $LOG_DIR/losses-L-raw.csv
echo "step,loss" > $LOG_DIR/losses-L.csv
cat $LOG_DIR/losses-L-raw.csv >> $LOG_DIR/losses-L.csv
rm $LOG_DIR/losses-L-raw.csv

# Tier-2 (S) losses - will have 3 entries per step, need to average
grep 'received_witness_metadata.*matformer_tier=2' $LOG_DIR/coordinator.log | \
    sed 's/\x1b\[[0-9;]*m//g' | \
    grep -oE 'step=[0-9]+.*loss=[0-9.]+' | \
    sed 's/step=//;s/.*loss=/,/' > $LOG_DIR/losses-S-raw.csv
echo "step,loss" > $LOG_DIR/losses-S-all.csv
cat $LOG_DIR/losses-S-raw.csv >> $LOG_DIR/losses-S-all.csv
rm $LOG_DIR/losses-S-raw.csv

# Summary
L_STEPS=$(wc -l < $LOG_DIR/losses-L.csv)
S_ENTRIES=$(wc -l < $LOG_DIR/losses-S-all.csv)
echo "L (tier-0) steps: $((L_STEPS - 1))"  # exclude header
echo "S (tier-2) entries: $((S_ENTRIES - 1))"  # 3 per step
echo ""
echo "Last 10 L losses:"
tail -10 $LOG_DIR/losses-L.csv
echo ""
echo "Last 10 S losses (raw):"
tail -10 $LOG_DIR/losses-S-all.csv

# Collect VRAM logs from remote
scp -i /root/.ssh/id_rsa ubuntu@64.247.196.12:/tmp/vram-$EXPERIMENT-A6000.csv $LOG_DIR/ 2>/dev/null || true
scp -i /root/.ssh/id_rsa -P 10014 root@203.57.40.132:/tmp/vram-$EXPERIMENT-S1.csv $LOG_DIR/ 2>/dev/null || true

# VRAM summary
echo ""
echo "VRAM Usage Summary:"
echo "L server (tier-0):"
awk -F',' '{sum+=$2; count++} END {print "  Average: " sum/count " MiB"}' $LOG_DIR/vram-L.csv 2>/dev/null || echo "  No data"
echo "A6000 (tier-2):"
awk -F',' '{sum+=$2; count++} END {print "  Average: " sum/count " MiB"}' $LOG_DIR/vram-$EXPERIMENT-A6000.csv 2>/dev/null || echo "  No data"
echo "S1 (tier-2):"
awk -F',' '{sum+=$2; count++} END {print "  Average: " sum/count " MiB"}' $LOG_DIR/vram-$EXPERIMENT-S1.csv 2>/dev/null || echo "  No data"

# Stop everything
pkill -9 psyche 2>/dev/null || true
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
pkill -f 'while true.*nvidia-smi' 2>/dev/null || true

echo ""
echo "=== Experiment Complete ==="
echo "Logs saved to: $LOG_DIR"
echo ""
echo "To generate plots, copy logs and run:"
echo "  python scripts/plot_experiment_losses.py $LOG_DIR"
date
