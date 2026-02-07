#!/usr/bin/env python3
"""
Plot 100-step comparison: L-only vs LSSS-diff vs LSSS-same
"""

import sys
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_content, tier_filter=None):
    """Extract step and loss from log content."""
    steps = []
    losses = []

    for line in log_content.split('\n'):
        if 'received_witness_metadata' not in line:
            continue

        # Filter by tier if specified
        if tier_filter is not None:
            tier_match = re.search(r'matformer_tier=(\d+)', line)
            if tier_match and int(tier_match.group(1)) != tier_filter:
                continue

        step_match = re.search(r'step=(\d+)', line)
        loss_match = re.search(r'loss=([0-9.]+)', line)

        if step_match and loss_match:
            step = int(step_match.group(1))
            loss = float(loss_match.group(1))
            if loss < 100:  # Filter out inf
                steps.append(step)
                losses.append(loss)

    return steps, losses

def average_by_step(steps, losses):
    """Average losses per step."""
    step_losses = {}
    for s, l in zip(steps, losses):
        if s not in step_losses:
            step_losses[s] = []
        step_losses[s].append(l)

    avg_steps = sorted(step_losses.keys())
    avg_losses = [np.mean(step_losses[s]) for s in avg_steps]
    return avg_steps, avg_losses

def main():
    if len(sys.argv) < 4:
        print("Usage: python plot_100step_comparison.py <L-only.log> <LSSS-diff.log> <LSSS-same.log> [output.png]")
        sys.exit(1)

    l_only_file = sys.argv[1]
    lsss_diff_file = sys.argv[2]
    lsss_same_file = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else 'comparison_100step.png'

    # Read logs
    with open(l_only_file) as f:
        l_only_log = f.read()
    with open(lsss_diff_file) as f:
        lsss_diff_log = f.read()
    with open(lsss_same_file) as f:
        lsss_same_log = f.read()

    # Parse L-only (tier-0 only)
    l_steps, l_losses = parse_log(l_only_log, tier_filter=0)
    l_steps, l_losses = average_by_step(l_steps, l_losses)

    # Parse LSSS-diff tier-0
    diff_t0_steps, diff_t0_losses = parse_log(lsss_diff_log, tier_filter=0)
    diff_t0_steps, diff_t0_losses = average_by_step(diff_t0_steps, diff_t0_losses)

    # Parse LSSS-diff tier-1 (averaged)
    diff_t1_steps, diff_t1_losses = parse_log(lsss_diff_log, tier_filter=1)
    diff_t1_steps, diff_t1_losses = average_by_step(diff_t1_steps, diff_t1_losses)

    # Parse LSSS-same tier-0
    same_t0_steps, same_t0_losses = parse_log(lsss_same_log, tier_filter=0)
    same_t0_steps, same_t0_losses = average_by_step(same_t0_steps, same_t0_losses)

    # Parse LSSS-same tier-1 (averaged)
    same_t1_steps, same_t1_losses = parse_log(lsss_same_log, tier_filter=1)
    same_t1_steps, same_t1_losses = average_by_step(same_t1_steps, same_t1_losses)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot L-only
    ax.plot(l_steps, l_losses, 'b-', linewidth=2, label='L-only (1 tier-0)', alpha=0.9)

    # Plot LSSS-diff
    ax.plot(diff_t0_steps, diff_t0_losses, 'r-', linewidth=2, label='LSSS-diff tier-0', alpha=0.9)
    ax.plot(diff_t1_steps, diff_t1_losses, 'r--', linewidth=2, label='LSSS-diff tier-1 (avg)', alpha=0.7)

    # Plot LSSS-same
    ax.plot(same_t0_steps, same_t0_losses, 'g-', linewidth=2, label='LSSS-same tier-0', alpha=0.9)
    ax.plot(same_t1_steps, same_t1_losses, 'g--', linewidth=2, label='LSSS-same tier-1 (avg)', alpha=0.7)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('100-Step Comparison: L-only vs LSSS (diff batch) vs LSSS (same batch)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Print summary
    print("\n=== Final Losses (step 99) ===")
    if l_losses:
        print(f"L-only:           {l_losses[-1]:.4f}")
    if diff_t0_losses:
        print(f"LSSS-diff tier-0: {diff_t0_losses[-1]:.4f}")
    if diff_t1_losses:
        print(f"LSSS-diff tier-1: {diff_t1_losses[-1]:.4f}")
    if same_t0_losses:
        print(f"LSSS-same tier-0: {same_t0_losses[-1]:.4f}")
    if same_t1_losses:
        print(f"LSSS-same tier-1: {same_t1_losses[-1]:.4f}")

if __name__ == '__main__':
    main()
