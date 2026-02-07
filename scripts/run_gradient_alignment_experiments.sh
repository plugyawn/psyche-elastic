#!/bin/bash
# Gradient Alignment Investigation Experiments
# Purpose: Determine if ~55% sign agreement in heterogeneous training is problematic
#
# Experiments:
# 1. Homogeneous disjoint baseline (4x tier-0, different batches)
# 2. Tier-2 only disjoint baseline (4x tier-2, different batches)
# 3. Same-batch heterogeneous (1 tier-0 + 3 tier-2, same batch)
# 4. Convergence comparison (extended runs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DURATION=${1:-180}
CONFIG_PATH=${2:-./config/nanogpt-20m-run}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="logs/gradient-alignment-${TIMESTAMP}"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Gradient Alignment Investigation Experiments"
echo "=============================================="
echo "Duration per experiment: ${DURATION}s"
echo "Config: ${CONFIG_PATH}"
echo "Output directory: ${LOG_DIR}"
echo ""

# Activate venv and set up environment
source .venv/bin/activate 2>/dev/null || true

export LIBTORCH_BYPASS_VERSION_CHECK=1
export DYLD_LIBRARY_PATH="${PROJECT_DIR}/.venv/lib/python3.12/site-packages/torch/lib"

# Build first to avoid repeated compilation
echo "Building release binary..."
cargo build --release -p psyche-centralized-local-testnet 2>&1 | tail -5

BINARY="${PROJECT_DIR}/target/release/psyche-centralized-local-testnet"

# Check if binary exists, otherwise try .cargo-target
if [ ! -f "$BINARY" ]; then
    BINARY="${PROJECT_DIR}/.cargo-target/release/psyche-centralized-local-testnet"
fi

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Release binary not found. Please build first:"
    echo "  cargo build --release -p psyche-centralized-local-testnet"
    exit 1
fi

run_experiment() {
    local name="$1"
    local tiers="$2"
    local helpers="$3"
    local same_batch="$4"
    local port="$5"

    local log_file="${LOG_DIR}/${name}.log"
    local distro_dir="${LOG_DIR}/${name}-distro"

    echo ""
    echo "----------------------------------------------"
    echo "Experiment: ${name}"
    echo "  Tiers: ${tiers}"
    echo "  Helpers: ${helpers}"
    echo "  Same batch: ${same_batch}"
    echo "  Port: ${port}"
    echo "  Log: ${log_file}"
    echo "----------------------------------------------"

    local cmd="${BINARY} start \
        --headless \
        --headless-exit-after-secs ${DURATION} \
        --server-port ${port} \
        --num-clients 4 \
        --config-path ${CONFIG_PATH} \
        --client-matformer-tiers ${tiers} \
        --client-matformer-helper-fractions ${helpers} \
        --write-distro-data ${distro_dir} \
        --tui false"

    if [ "$same_batch" = "true" ]; then
        PSYCHE_FORCE_SAME_BATCH=1 $cmd 2>&1 | tee "$log_file"
    else
        $cmd 2>&1 | tee "$log_file"
    fi

    # Extract gradient alignment metrics from log
    echo ""
    echo "Gradient alignment summary for ${name}:"
    grep "gradient_alignment" "$log_file" | tail -20 || echo "  (no gradient_alignment entries found)"
    echo ""
}

# ============================================
# Experiment 1: Homogeneous Disjoint Baseline
# ============================================
# Purpose: What's "normal" sign agreement with same architecture, different batches?
# Expected: If ~55-60%, then data variance alone explains observed heterogeneous results
run_experiment "exp1-homo-disjoint" "0,0,0,0" "0,0,0,0" "false" "20201"

# ============================================
# Experiment 2: Tier-2 Only Disjoint Baseline
# ============================================
# Purpose: What's sign agreement between tier-2 clients on different batches?
# Expected: Should be comparable to homogeneous tier-0 baseline
run_experiment "exp2-tier2-disjoint" "2,2,2,2" "0,0,0,0" "false" "20202"

# ============================================
# Experiment 3: Same-Batch Heterogeneous
# ============================================
# Purpose: Isolate tier-mismatch from data variance
# Expected: If ~58% (matches DCT baseline), tier mismatch is fine
run_experiment "exp3-hetero-samebatch" "0,2,2,2" "0,0,0,0" "true" "20203"

# ============================================
# Experiment 4: Heterogeneous Disjoint (Production Config)
# ============================================
# Purpose: Actual production scenario for comparison
# Expected: Similar to exp1 if tier variance doesn't add significant degradation
run_experiment "exp4-hetero-disjoint" "0,2,2,2" "0,0,0,0" "false" "20204"

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
echo ""
echo "Results saved to: ${LOG_DIR}/"
echo ""
echo "Quick analysis:"
echo "  - If exp1 (homo-disjoint) shows ~55-60% sign agreement:"
echo "    Data variance alone explains the observed heterogeneous results."
echo "    Sign agreement is NOT the issue."
echo ""
echo "  - If exp1 shows >70% sign agreement:"
echo "    Tier variance contributes significantly."
echo "    Check exp3 (same-batch hetero) to isolate DCT vs tier effects."
echo ""
echo "  - If exp3 (same-batch hetero) shows ~58%:"
echo "    DCT compression is the bottleneck, not tier mismatch."
echo ""
echo "  - If exp3 shows <50%:"
echo "    There may be a bug in tier alignment."
echo ""

# Summary extraction
echo "Extracting final metrics..."
for exp in exp1-homo-disjoint exp2-tier2-disjoint exp3-hetero-samebatch exp4-hetero-disjoint; do
    log_file="${LOG_DIR}/${exp}.log"
    if [ -f "$log_file" ]; then
        echo ""
        echo "=== ${exp} ==="
        # Extract last 10 gradient_alignment entries (MLP params are most interesting)
        grep "gradient_alignment" "$log_file" | grep "mlp" | tail -5 || echo "(no mlp entries)"
        # Count total steps
        steps=$(grep -c "gradient_alignment" "$log_file" 2>/dev/null || echo "0")
        echo "Total gradient_alignment log entries: ${steps}"
    fi
done
