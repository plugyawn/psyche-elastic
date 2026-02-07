#!/bin/bash
# Combined experiment runner: L-only vs LSSS-same with batch=64
# Same wallclock time (10 minutes each) for fair comparison
#
# Usage (on the L server):
#   ./scripts/run-vram-experiment-batch64.sh
#
# Prerequisites:
#   1. Copy config files to /opt/psyche/config/:
#      scp config/experiment-L-only/state.toml L:/opt/psyche/config/exp-L-only.toml
#      scp config/experiment-LSSS-same/state.toml L:/opt/psyche/config/exp-LSSS-same.toml
#
#   2. Ensure all servers have the psyche-centralized-client binary at /opt/
#   3. Ensure tiny shakespeare data at /opt/psyche/data/tinyshakespeare/
#
# Output:
#   - Logs in /tmp/exp-*/ directories
#   - Comparison plot: experiment_comparison.png

set -e

DURATION=600  # 10 minutes per experiment
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BASE_LOG_DIR="/tmp/exp-batch64-$TIMESTAMP"
mkdir -p $BASE_LOG_DIR

echo "======================================================"
echo "MatFormer VRAM Experiment: L-only vs LSSS-same (batch=64)"
echo "======================================================"
echo "Duration per experiment: $DURATION seconds"
echo "Base log directory: $BASE_LOG_DIR"
echo ""

# Function to clean up all processes
cleanup() {
    echo "Cleaning up..."
    pkill -9 psyche 2>/dev/null || true
    ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
    ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "pkill -9 psyche 2>/dev/null || true" 2>/dev/null || true
    pkill -f 'while true.*nvidia-smi' 2>/dev/null || true
    ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "pkill -f 'while true.*nvidia-smi' 2>/dev/null || true" 2>/dev/null || true
    ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "pkill -f 'while true.*nvidia-smi' 2>/dev/null || true" 2>/dev/null || true
    sleep 3
}

# Initial cleanup
cleanup

cd /root/psyche
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib

###############################################
# EXPERIMENT 1: L-only (4 tier-0 clients)
###############################################
echo ""
echo "======================================================"
echo "EXPERIMENT 1: L-only (4 tier-0 clients)"
echo "======================================================"
echo "Start time: $(date)"

L_ONLY_DIR="$BASE_LOG_DIR/L-only"
mkdir -p $L_ONLY_DIR

# Start VRAM monitors
nohup bash -c "while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done" > $L_ONLY_DIR/vram-L.csv 2>&1 &
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-L-only-A6000.csv 2>&1 &" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-L-only-S1.csv 2>&1 &" 2>/dev/null || true

# Start coordinator
echo "Starting coordinator..."
nohup ./target/release/psyche-centralized-server run \
    --state /opt/psyche/config/exp-L-only.toml \
    --server-port 17892 \
    --tui false \
    --logs console > $L_ONLY_DIR/coordinator.log 2>&1 &
sleep 5

# Start 4 tier-0 clients
echo "Starting 4 tier-0 clients..."
nohup ./target/release/psyche-centralized-client train --run-id exp-L-only --server-addr 127.0.0.1:17892 --device cuda:0 --matformer-tier 0 --logs console > $L_ONLY_DIR/client-t0-L.log 2>&1 &

ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
nohup /opt/psyche-centralized-client train --run-id exp-L-only --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 0 --logs console > /tmp/client-L-only-t0-A0.log 2>&1 &
nohup /opt/psyche-centralized-client train --run-id exp-L-only --server-addr 216.81.248.128:17892 --device cuda:1 --matformer-tier 0 --logs console > /tmp/client-L-only-t0-A1.log 2>&1 &
" 2>/dev/null || true

ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib
nohup /opt/psyche-centralized-client train --run-id exp-L-only --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 0 --logs console > /tmp/client-L-only-t0-S1.log 2>&1 &
" 2>/dev/null || true

echo "Waiting $DURATION seconds..."
sleep $DURATION

# Collect VRAM logs
scp -i /root/.ssh/id_rsa ubuntu@64.247.196.12:/tmp/vram-L-only-A6000.csv $L_ONLY_DIR/ 2>/dev/null || true
scp -i /root/.ssh/id_rsa -P 10014 root@203.57.40.132:/tmp/vram-L-only-S1.csv $L_ONLY_DIR/ 2>/dev/null || true

# Cleanup
cleanup

echo "L-only experiment complete at $(date)"

###############################################
# EXPERIMENT 2: LSSS-same (1 tier-0 + 3 tier-2)
###############################################
echo ""
echo "======================================================"
echo "EXPERIMENT 2: LSSS-same (1 tier-0 + 3 tier-2, same batch)"
echo "======================================================"
echo "Start time: $(date)"

LSSS_DIR="$BASE_LOG_DIR/LSSS-same"
mkdir -p $LSSS_DIR

# Start VRAM monitors
nohup bash -c "while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done" > $LSSS_DIR/vram-L.csv 2>&1 &
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-LSSS-A6000.csv 2>&1 &" 2>/dev/null || true
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "nohup bash -c 'while true; do echo \"\$(date +%s),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)\"; sleep 5; done' > /tmp/vram-LSSS-S1.csv 2>&1 &" 2>/dev/null || true

# Start coordinator
echo "Starting coordinator..."
nohup ./target/release/psyche-centralized-server run \
    --state /opt/psyche/config/exp-LSSS-same.toml \
    --server-port 17892 \
    --tui false \
    --logs console > $LSSS_DIR/coordinator.log 2>&1 &
sleep 5

# Start tier-0 on L with SAME_BATCH
echo "Starting 1 tier-0 + 3 tier-2 clients (SAME_BATCH)..."
export PSYCHE_FORCE_SAME_BATCH=1
nohup ./target/release/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 127.0.0.1:17892 --device cuda:0 --matformer-tier 0 --logs console > $LSSS_DIR/client-t0-L.log 2>&1 &

# Start tier-2 on A6000 (2 GPUs) with SAME_BATCH
ssh -i /root/.ssh/id_rsa ubuntu@64.247.196.12 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
export PSYCHE_FORCE_SAME_BATCH=1
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 2 --logs console > /tmp/client-LSSS-t2-A0.log 2>&1 &
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 216.81.248.128:17892 --device cuda:1 --matformer-tier 2 --logs console > /tmp/client-LSSS-t2-A1.log 2>&1 &
" 2>/dev/null || true

# Start tier-2 on S1 with SAME_BATCH
ssh -i /root/.ssh/id_rsa -p 10014 root@203.57.40.132 "
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib
export PSYCHE_FORCE_SAME_BATCH=1
nohup /opt/psyche-centralized-client train --run-id exp-LSSS-same --server-addr 216.81.248.128:17892 --device cuda:0 --matformer-tier 2 --logs console > /tmp/client-LSSS-t2-S1.log 2>&1 &
" 2>/dev/null || true

echo "Waiting $DURATION seconds..."
sleep $DURATION

# Collect VRAM logs
scp -i /root/.ssh/id_rsa ubuntu@64.247.196.12:/tmp/vram-LSSS-A6000.csv $LSSS_DIR/ 2>/dev/null || true
scp -i /root/.ssh/id_rsa -P 10014 root@203.57.40.132:/tmp/vram-LSSS-S1.csv $LSSS_DIR/ 2>/dev/null || true

# Cleanup
cleanup

echo "LSSS-same experiment complete at $(date)"

###############################################
# RESULTS
###############################################
echo ""
echo "======================================================"
echo "RESULTS SUMMARY"
echo "======================================================"

echo ""
echo "=== L-only VRAM Usage ==="
echo "L server (tier-0):"
awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) print "  Average: " sum/count " MiB, samples: " count}' $L_ONLY_DIR/vram-L.csv 2>/dev/null || echo "  No data"
echo "A6000 (tier-0):"
awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) print "  Average: " sum/count " MiB, samples: " count}' $L_ONLY_DIR/vram-L-only-A6000.csv 2>/dev/null || echo "  No data"
echo "S1 (tier-0):"
awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) print "  Average: " sum/count " MiB, samples: " count}' $L_ONLY_DIR/vram-L-only-S1.csv 2>/dev/null || echo "  No data"

echo ""
echo "=== LSSS-same VRAM Usage ==="
echo "L server (tier-0):"
awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) print "  Average: " sum/count " MiB, samples: " count}' $LSSS_DIR/vram-L.csv 2>/dev/null || echo "  No data"
echo "A6000 (tier-2):"
awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) print "  Average: " sum/count " MiB, samples: " count}' $LSSS_DIR/vram-LSSS-A6000.csv 2>/dev/null || echo "  No data"
echo "S1 (tier-2):"
awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) print "  Average: " sum/count " MiB, samples: " count}' $LSSS_DIR/vram-LSSS-S1.csv 2>/dev/null || echo "  No data"

echo ""
echo "=== Loss Summary ==="
# Count steps for each experiment
L_ONLY_STEPS=$(grep -c 'received_witness_metadata' $L_ONLY_DIR/coordinator.log 2>/dev/null || echo 0)
LSSS_T0_STEPS=$(grep 'received_witness_metadata.*matformer_tier=0' $LSSS_DIR/coordinator.log 2>/dev/null | wc -l || echo 0)
LSSS_T2_STEPS=$(grep 'received_witness_metadata.*matformer_tier=2' $LSSS_DIR/coordinator.log 2>/dev/null | wc -l || echo 0)

echo "L-only total witness reports: $L_ONLY_STEPS"
echo "LSSS tier-0 (L) reports: $LSSS_T0_STEPS"
echo "LSSS tier-2 (S) reports: $LSSS_T2_STEPS"

# Show last few losses
echo ""
echo "L-only final losses (last 5):"
grep 'received_witness_metadata' $L_ONLY_DIR/coordinator.log | tail -5 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE 'step=[0-9]+.*loss=[0-9.]+' 2>/dev/null || echo "  No data"

echo ""
echo "LSSS tier-0 (L) final losses (last 5):"
grep 'received_witness_metadata.*matformer_tier=0' $LSSS_DIR/coordinator.log | tail -5 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE 'step=[0-9]+.*loss=[0-9.]+' 2>/dev/null || echo "  No data"

echo ""
echo "LSSS tier-2 (S) final losses (last 5):"
grep 'received_witness_metadata.*matformer_tier=2' $LSSS_DIR/coordinator.log | tail -5 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE 'step=[0-9]+.*loss=[0-9.]+' 2>/dev/null || echo "  No data"

echo ""
echo "======================================================"
echo "Logs saved to: $BASE_LOG_DIR"
echo ""
echo "To generate comparison plot locally:"
echo "  scp -r L:$BASE_LOG_DIR ./exp-results/"
echo "  python scripts/plot_experiment_losses.py ./exp-results/L-only ./exp-results/LSSS-same"
echo "======================================================"
echo "Experiment completed at $(date)"
