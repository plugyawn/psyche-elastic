#!/usr/bin/env python3
"""
Plot per-step losses from Psyche experiments.
Supports L-only and LSSS experiments with per-tier loss tracking.

Usage:
    python scripts/plot_experiment_losses.py <log_dir_L_only> <log_dir_LSSS>
    python scripts/plot_experiment_losses.py <single_log_dir>
"""

import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_coordinator_log(log_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse coordinator log to extract per-step losses for tier-0 and tier-2.
    Returns (tier0_df, tier2_df) with columns [step, loss].
    """
    tier0_data = []
    tier2_data = []

    with open(log_path, 'r') as f:
        for line in f:
            # Remove ANSI escape codes
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)

            if 'received_witness_metadata' not in line:
                continue

            # Extract matformer_tier
            tier_match = re.search(r'matformer_tier=(\d+)', line)
            if not tier_match:
                continue
            tier = int(tier_match.group(1))

            # Extract step
            step_match = re.search(r'step=(\d+)', line)
            if not step_match:
                continue
            step = int(step_match.group(1))

            # Extract loss
            loss_match = re.search(r'loss=([0-9.]+)', line)
            if not loss_match:
                continue
            loss = float(loss_match.group(1))

            if tier == 0:
                tier0_data.append({'step': step, 'loss': loss})
            elif tier == 2:
                tier2_data.append({'step': step, 'loss': loss})

    tier0_df = pd.DataFrame(tier0_data)
    tier2_df = pd.DataFrame(tier2_data)

    return tier0_df, tier2_df


def average_tier2_losses(tier2_df: pd.DataFrame) -> pd.DataFrame:
    """Average tier-2 losses per step (since there are 3 S clients)."""
    if tier2_df.empty:
        return tier2_df
    return tier2_df.groupby('step').agg({'loss': 'mean'}).reset_index()

def average_losses_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """Average losses per step (useful when there are multiple witnesses per tier)."""
    if df.empty:
        return df
    return df.groupby('step').agg({'loss': 'mean'}).reset_index()


def smooth(values, window=10):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')


def plot_comparison(l_only_dir: str, lsss_dir: str, output_path: str = None):
    """
    Plot L-only vs LSSS comparison.
    - L-only: all tier-0 losses
    - LSSS: tier-0 (L) loss and averaged tier-2 (S) loss
    """
    # Parse L-only logs
    l_only_log = os.path.join(l_only_dir, 'coordinator.log')
    l_only_t0, _ = parse_coordinator_log(l_only_log)
    l_only_t0 = average_losses_by_step(l_only_t0)

    # Parse LSSS logs
    lsss_log = os.path.join(lsss_dir, 'coordinator.log')
    lsss_t0, lsss_t2 = parse_coordinator_log(lsss_log)
    lsss_t0 = average_losses_by_step(lsss_t0)
    lsss_t2_avg = average_tier2_losses(lsss_t2)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Raw losses
    ax1 = axes[0]
    ax1.set_title('Per-Step Loss (Raw)', fontsize=12)

    if not l_only_t0.empty:
        ax1.plot(l_only_t0['step'], l_only_t0['loss'],
                 label='L-only (4x tier-0)', alpha=0.7, linewidth=1)

    if not lsss_t0.empty:
        ax1.plot(lsss_t0['step'], lsss_t0['loss'],
                 label='LSSS L (tier-0)', alpha=0.7, linewidth=1)

    if not lsss_t2_avg.empty:
        ax1.plot(lsss_t2_avg['step'], lsss_t2_avg['loss'],
                 label='LSSS S avg (tier-2)', alpha=0.7, linewidth=1)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Smoothed losses
    ax2 = axes[1]
    ax2.set_title('Per-Step Loss (Smoothed, window=20)', fontsize=12)

    window = 20
    if not l_only_t0.empty and len(l_only_t0) > window:
        smoothed = smooth(l_only_t0['loss'].values, window)
        steps = l_only_t0['step'].values[window-1:]
        ax2.plot(steps, smoothed, label='L-only (4x tier-0)', linewidth=2)

    if not lsss_t0.empty and len(lsss_t0) > window:
        smoothed = smooth(lsss_t0['loss'].values, window)
        steps = lsss_t0['step'].values[window-1:]
        ax2.plot(steps, smoothed, label='LSSS L (tier-0)', linewidth=2)

    if not lsss_t2_avg.empty and len(lsss_t2_avg) > window:
        smoothed = smooth(lsss_t2_avg['loss'].values, window)
        steps = lsss_t2_avg['step'].values[window-1:]
        ax2.plot(steps, smoothed, label='LSSS S avg (tier-2)', linewidth=2)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if not l_only_t0.empty:
        print(f"L-only: {len(l_only_t0)} steps, final loss: {l_only_t0['loss'].iloc[-1]:.4f}")
    if not lsss_t0.empty:
        print(f"LSSS L (tier-0): {len(lsss_t0)} steps, final loss: {lsss_t0['loss'].iloc[-1]:.4f}")
    if not lsss_t2_avg.empty:
        print(f"LSSS S (tier-2 avg): {len(lsss_t2_avg)} steps, final loss: {lsss_t2_avg['loss'].iloc[-1]:.4f}")


def plot_single_experiment(log_dir: str, output_path: str = None):
    """Plot losses from a single experiment directory."""
    log_path = os.path.join(log_dir, 'coordinator.log')
    tier0_df, tier2_df = parse_coordinator_log(log_path)
    tier0_df = average_losses_by_step(tier0_df)
    tier2_avg = average_tier2_losses(tier2_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot raw
    ax1 = axes[0]
    ax1.set_title('Per-Step Loss (Raw)', fontsize=12)

    if not tier0_df.empty:
        ax1.plot(tier0_df['step'], tier0_df['loss'],
                 label='L (tier-0)', alpha=0.7, linewidth=1)

    if not tier2_avg.empty:
        ax1.plot(tier2_avg['step'], tier2_avg['loss'],
                 label='S avg (tier-2)', alpha=0.7, linewidth=1)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot smoothed
    ax2 = axes[1]
    ax2.set_title('Per-Step Loss (Smoothed, window=20)', fontsize=12)

    window = 20
    if not tier0_df.empty and len(tier0_df) > window:
        smoothed = smooth(tier0_df['loss'].values, window)
        steps = tier0_df['step'].values[window-1:]
        ax2.plot(steps, smoothed, label='L (tier-0)', linewidth=2)

    if not tier2_avg.empty and len(tier2_avg) > window:
        smoothed = smooth(tier2_avg['loss'].values, window)
        steps = tier2_avg['step'].values[window-1:]
        ax2.plot(steps, smoothed, label='S avg (tier-2)', linewidth=2)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    # Print summary
    print("\n=== Summary Statistics ===")
    if not tier0_df.empty:
        print(f"L (tier-0): {len(tier0_df)} steps, final loss: {tier0_df['loss'].iloc[-1]:.4f}")
    if not tier2_avg.empty:
        print(f"S (tier-2 avg): {len(tier2_avg)} steps, final loss: {tier2_avg['loss'].iloc[-1]:.4f}")


def main():
    if len(sys.argv) == 2:
        # Single experiment
        log_dir = sys.argv[1]
        output_path = os.path.join(log_dir, 'losses_plot.png')
        plot_single_experiment(log_dir, output_path)
    elif len(sys.argv) == 3:
        # Comparison of L-only vs LSSS
        l_only_dir = sys.argv[1]
        lsss_dir = sys.argv[2]
        output_path = 'experiment_comparison.png'
        plot_comparison(l_only_dir, lsss_dir, output_path)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
