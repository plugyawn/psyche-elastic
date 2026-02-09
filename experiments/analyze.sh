#!/bin/bash
# Analyze experiment results without re-running
# Usage: ./experiments/analyze.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================================================="
echo "   HETEROGENEOUS MATFORMER TRAINING - RESULTS ANALYSIS"
echo "==============================================================="
echo ""

analyze_experiment() {
    local log_file="$1"
    local name=$(basename "$log_file" .log)

    if [ ! -f "$log_file" ]; then
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
    local nans=$(grep -ci "nan" "$log_file" 2>/dev/null || echo "0")

    printf "│ %-23s │ %-10s │ %-9s │ %-8s │\n" \
        "$name" "$final_loss" "$max_step" "$panics"
}

echo "┌─────────────────────────┬────────────┬───────────┬──────────┐"
echo "│ Experiment              │ Final Loss │ Steps     │ Panics   │"
echo "├─────────────────────────┼────────────┼───────────┼──────────┤"
for log in "$SCRIPT_DIR"/exp*.log; do
    if [ -f "$log" ]; then
        analyze_experiment "$log"
    fi
done
echo "└─────────────────────────┴────────────┴───────────┴──────────┘"

echo ""
echo "=== DETAILED LOSS PROGRESSION ==="
echo ""

for log in "$SCRIPT_DIR"/exp*.log; do
    if [ -f "$log" ]; then
        name=$(basename "$log" .log)
        echo "--- $name ---"
        # Show loss at steps 1, 10, 20, 30, 40, 50, 60
        for step in 1 10 20 30 40 50 60; do
            loss=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | \
                grep -E "step=$step\b.*loss=" | head -1 | \
                grep -oE "loss=[0-9.]+" | cut -d= -f2)
            if [ -n "$loss" ]; then
                printf "  step %2d: %.2f\n" "$step" "$loss"
            fi
        done
        echo ""
    fi
done

echo "=== COMPARISON VS BASELINE ==="
echo ""

baseline_loss=$(sed 's/\x1b\[[0-9;]*m//g' "$SCRIPT_DIR/exp1_homogeneous.log" 2>/dev/null | \
    grep "trained_batches=1.*loss=" | tail -3 | \
    grep -oE "loss=[0-9.]+" | cut -d= -f2 | \
    awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print 0}')

if [ -n "$baseline_loss" ] && [ "$baseline_loss" != "0" ]; then
    echo "Baseline (exp1_homogeneous) final loss: $baseline_loss"
    echo ""
    for log in "$SCRIPT_DIR"/exp*.log; do
        if [ -f "$log" ]; then
            name=$(basename "$log" .log)
            if [ "$name" != "exp1_homogeneous" ]; then
                exp_loss=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | \
                    grep "trained_batches=1.*loss=" | tail -3 | \
                    grep -oE "loss=[0-9.]+" | cut -d= -f2 | \
                    awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print 0}')
                if [ -n "$exp_loss" ] && [ "$exp_loss" != "0" ]; then
                    diff=$(echo "scale=2; (($exp_loss - $baseline_loss) / $baseline_loss) * 100" | bc)
                    printf "  %s: %.2f (%+.1f%% vs baseline)\n" "$name" "$exp_loss" "$diff"
                fi
            fi
        fi
    done
fi

echo ""
echo "==============================================================="
