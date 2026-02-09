#!/bin/bash
# Heterogeneous MatFormer Training Experiments
# Run from the psyche root directory: ./experiments/run_experiments.sh

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR"
CONFIG_PATH="./config/nanogpt-20m-run"
TIMEOUT=300  # seconds per experiment

# Set library path for libtorch
export DYLD_LIBRARY_PATH="$PROJECT_DIR/.venv/lib/python3.12/site-packages/torch/lib"

cd "$PROJECT_DIR"

echo "==============================================================="
echo "   HETEROGENEOUS MATFORMER TRAINING EXPERIMENTS"
echo "==============================================================="
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Config path: $CONFIG_PATH"
echo "Timeout per experiment: ${TIMEOUT}s"
echo ""

# Build once
echo "Building release binary..."
cargo build --release -p psyche-centralized-local-testnet
echo ""

run_experiment() {
    local name="$1"
    local num_clients="$2"
    local tiers="$3"
    local helper_fractions="$4"
    local log_file="$RESULTS_DIR/$name.log"

    echo "─────────────────────────────────────────────────────────────"
    echo "Running: $name"
    echo "  Clients: $num_clients"
    echo "  Tiers: $tiers"
    echo "  Helper fractions: $helper_fractions"
    echo "  Log: $log_file"
    echo "─────────────────────────────────────────────────────────────"

    cargo run --release -p psyche-centralized-local-testnet -- start \
        --headless \
        --headless-exit-after-secs "$TIMEOUT" \
        --num-clients "$num_clients" \
        --config-path "$CONFIG_PATH" \
        --client-matformer-tiers "$tiers" \
        --client-matformer-helper-fractions "$helper_fractions" \
        --tui false \
        2>&1 | tee "$log_file"

    echo ""
    echo "Completed: $name"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Homogeneous Baseline (3x tier-0)
# ═══════════════════════════════════════════════════════════════════
run_experiment "exp1_homogeneous" 3 "0,0,0" "0,0,0"

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Heterogeneous 1:2 (1x tier-0, 2x tier-1)
# ═══════════════════════════════════════════════════════════════════
run_experiment "exp2_heterogeneous_1to2" 3 "0,1,1" "0,0,0"

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Heterogeneous 2:1 (2x tier-0, 1x tier-1)
# ═══════════════════════════════════════════════════════════════════
run_experiment "exp3_heterogeneous_2to1" 3 "0,0,1" "0,0,0"

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Extended Stability (200 steps, requires config change)
# ═══════════════════════════════════════════════════════════════════
# Uncomment to run (requires modifying config or longer timeout):
# run_experiment "exp4_stability_200steps" 3 "0,1,1" "0,0,0"

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Extreme Ratio 1:3 (1x tier-0, 3x tier-1)
# ═══════════════════════════════════════════════════════════════════
run_experiment "exp5_heterogeneous_1to3" 4 "0,1,1,1" "0,0,0,0"

# ═══════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "==============================================================="
echo "   EXPERIMENT RESULTS ANALYSIS"
echo "==============================================================="
echo ""

analyze_experiment() {
    local log_file="$1"
    local name=$(basename "$log_file" .log)

    if [ ! -f "$log_file" ]; then
        echo "$name: (log not found)"
        return
    fi

    # Strip ANSI codes and extract metrics
    local final_loss=$(sed 's/\x1b\[[0-9;]*m//g' "$log_file" | \
        grep "trained_batches=1.*loss=" | tail -3 | \
        grep -oE "loss=[0-9.]+" | cut -d= -f2 | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.2f", sum/n; else print "N/A"}')

    local max_step=$(sed 's/\x1b\[[0-9;]*m//g' "$log_file" | \
        grep -oE "step=[0-9]+" | sed 's/step=//' | sort -n | tail -1)

    local panics=$(grep -ci "panic" "$log_file" 2>/dev/null || echo "0")

    printf "%-25s │ loss=%-6s │ steps=%-4s │ panics=%s\n" \
        "$name" "$final_loss" "$max_step" "$panics"
}

echo "┌─────────────────────────┬────────────┬───────────┬──────────┐"
echo "│ Experiment              │ Final Loss │ Steps     │ Panics   │"
echo "├─────────────────────────┼────────────┼───────────┼──────────┤"
for log in "$RESULTS_DIR"/exp*.log; do
    if [ -f "$log" ]; then
        analyze_experiment "$log"
    fi
done
echo "└─────────────────────────┴────────────┴───────────┴──────────┘"

echo ""
echo "==============================================================="
echo "   ALL EXPERIMENTS COMPLETE"
echo "==============================================================="
echo ""
echo "Log files saved to: $RESULTS_DIR/"
echo ""
echo "To re-run analysis only:"
echo "  ./experiments/analyze.sh"
echo ""
