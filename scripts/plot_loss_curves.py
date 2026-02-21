#!/usr/bin/env python3
"""
Plot loss curves from Psyche client/coordinator logs.

This is intentionally tolerant to:
- ANSI color codes
- occasional NUL bytes and non-UTF8 sequences
- multiple clients reporting per step (we can aggregate mean/median/first)

Outputs PNGs into plots/rendered/.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    tier: int = 0
    # How to aggregate multiple reports for the same (step,tier) within a log:
    # - first: first seen loss per step
    # - mean: average across all seen for that step
    # - median: median across all seen for that step
    agg: str = "first"


def _safe_float(s: str) -> Optional[float]:
    s = s.strip()
    if s.lower() in ("inf", "+inf", "-inf", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_losses_by_step(log_path: Path, tier: int) -> Dict[int, List[float]]:
    losses: Dict[int, List[float]] = {}
    if not log_path.exists():
        return losses

    with log_path.open("rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore")
            line = ANSI_ESCAPE.sub("", line)

            if "matformer_tier=" not in line or "loss=" not in line or "step=" not in line:
                continue

            tier_m = re.search(r"\bmatformer_tier=(\d+)\b", line)
            if not tier_m or int(tier_m.group(1)) != tier:
                continue

            step_m = re.search(r"\bstep=(\d+)\b", line)
            loss_m = re.search(r"\bloss=([0-9.]+|inf|nan)\b", line)
            if not step_m or not loss_m:
                continue

            step = int(step_m.group(1))
            loss = _safe_float(loss_m.group(1))
            if loss is None:
                continue

            losses.setdefault(step, []).append(loss)

    return losses


def aggregate_losses(
    losses_by_step: Dict[int, List[float]],
    agg: str,
) -> Tuple[np.ndarray, np.ndarray]:
    steps = np.array(sorted(losses_by_step.keys()), dtype=np.int64)
    ys: List[float] = []
    for s in steps:
        vals = losses_by_step[int(s)]
        if not vals:
            ys.append(float("nan"))
            continue
        if agg == "first":
            ys.append(vals[0])
        elif agg == "mean":
            ys.append(float(np.mean(vals)))
        elif agg == "median":
            ys.append(float(np.median(vals)))
        else:
            raise ValueError(f"unknown agg: {agg}")
    return steps, np.array(ys, dtype=np.float64)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()

    # Simple causal moving average; keep alignment simple for blog plots.
    out = np.full_like(y, np.nan, dtype=np.float64)
    for i in range(len(y)):
        start = max(0, i - window + 1)
        chunk = y[start : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            out[i] = float(np.mean(chunk))
    return out


def plot_runs(
    runs: List[RunSpec],
    out_path: Path,
    title: str,
    smooth_window: int = 20,
    xlim: Optional[Tuple[int, int]] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    ax_raw, ax_smooth = axes

    for run in runs:
        losses_by_step = extract_losses_by_step(run.path, tier=run.tier)
        steps, y = aggregate_losses(losses_by_step, agg=run.agg)
        if steps.size == 0:
            continue

        ax_raw.plot(steps, y, linewidth=1.3, label=run.label)

        y_s = moving_average(y, smooth_window)
        ax_smooth.plot(steps, y_s, linewidth=2.0, label=run.label)

    ax_raw.set_title("Per-Step Loss (Raw)")
    ax_raw.set_xlabel("Step")
    ax_raw.set_ylabel("Loss")
    ax_raw.grid(True, alpha=0.25)

    ax_smooth.set_title(f"Per-Step Loss (Smoothed, window={smooth_window})")
    ax_smooth.set_xlabel("Step")
    ax_smooth.set_ylabel("Loss")
    ax_smooth.grid(True, alpha=0.25)

    for ax in axes:
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

    # Put legend in the smoothed plot (usually fewer overlaps).
    ax_smooth.legend(loc="best", frameon=True)

    fig.suptitle(title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="plots/rendered", help="output directory for PNGs")
    ap.add_argument("--smooth-window", type=int, default=20)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = (repo / args.out_dir).resolve()

    # 500-step baselines: tier-0 client logs captured for one representative client.
    runs_500 = [
        RunSpec("L-only (500; client0)", repo / "plots/input/exp-500-L-only_client0.log", tier=0, agg="first"),
        RunSpec("L-solo (500)", repo / "plots/input/exp-500-L-solo_client0.log", tier=0, agg="first"),
        RunSpec("LSSS-diff (500)", repo / "plots/input/exp-500-LSSS-diff_client0.log", tier=0, agg="first"),
        RunSpec("LSSS-same (500)", repo / "plots/input/exp-500-LSSS-same_client0.log", tier=0, agg="first"),
        # Intentionally exclude LSSS-same-aligned (known-bug run).
    ]
    plot_runs(
        runs_500,
        out_dir / "tier0_loss_500_baselines.png",
        title="Tier-0 Loss Curves (500-step baselines)",
        smooth_window=args.smooth_window,
    )

    # 100-step coordinator logs: multiple clients report; use mean per step for tier-0.
    runs_100 = [
        RunSpec("L-only (100, nosign; tier0 mean)", repo / "plots/input/exp-100-L-only-nosign-20260206-075024/coordinator.log", tier=0, agg="mean"),
        RunSpec("LSSS-diff (100, nosign; tier0 mean)", repo / "plots/input/exp-100-LSSS-diff-nosign-20260206-080932/coordinator.log", tier=0, agg="mean"),
        RunSpec("LSSS-diff (100, nosign+pcgrad; tier0 mean)", repo / "plots/input/exp-100-LSSS-diff-nosign-pcgrad-20260206-082243/coordinator.log", tier=0, agg="mean"),
        RunSpec("LSSS-diff (100, nosign+topk16; tier0 mean)", repo / "plots/input/exp-100-LSSS-diff-nosign-tk16-20260206-085023/coordinator.log", tier=0, agg="mean"),
        RunSpec("LSSS-diff (100, topk16+pcgrad; tier0 mean)", repo / "plots/input/exp-100-LSSS-tk16pcg-20260206-090642/coordinator.log", tier=0, agg="mean"),
        RunSpec("LSSS (100, mlp-topk16; tier0 mean)", repo / "plots/input/exp-100-LSSS-mlptk16-20260206-092502/coordinator.log", tier=0, agg="mean"),
    ]
    plot_runs(
        runs_100,
        out_dir / "tier0_loss_100_variants.png",
        title="Tier-0 Loss Curves (100-step variants)",
        smooth_window=args.smooth_window,
    )

    # Core-objective schedule sweeps (tier-0 mean from coordinator logs).
    runs_core = [
        RunSpec("core-const (tier0 mean)", repo / "plots/input/coreobj/coord-exp-100-LSSS-core-const.log", tier=0, agg="mean"),
        RunSpec("core-linear (tier0 mean)", repo / "plots/input/coreobj/coord-exp-100-LSSS-core-linear.log", tier=0, agg="mean"),
        RunSpec("core-cosine (tier0 mean)", repo / "plots/input/coreobj/coord-exp-100-LSSS-core-cosine.log", tier=0, agg="mean"),
    ]
    plot_runs(
        runs_core,
        out_dir / "tier0_loss_core_objective_schedules.png",
        title="Tier-0 Loss: Core Objective Schedule Sweeps",
        smooth_window=max(10, args.smooth_window // 2),
    )

    # Suffix gate run: show tier-0 vs tier-2 avg from client logs (more direct than coordinator aggregation).
    suf_dir = repo / "plots/input/exp-100-LSSS-sufgate-20260206-162629"
    runs_suf = [
        RunSpec("LSSS L (tier0)", suf_dir / "client-t0-L.log", tier=0, agg="first"),
        RunSpec("LSSS S avg (tier2; A0)", suf_dir / "exp100-LSSS-sufgate-A0.log", tier=2, agg="first"),
        RunSpec("LSSS S avg (tier2; A1)", suf_dir / "exp100-LSSS-sufgate-A1.log", tier=2, agg="first"),
        RunSpec("LSSS S avg (tier2; S1)", suf_dir / "exp100-LSSS-sufgate-S1.log", tier=2, agg="first"),
    ]
    plot_runs(
        runs_suf,
        out_dir / "loss_100_sufgate_tier0_and_tier2.png",
        title="Loss Curves (Suffix Gate Run)",
        smooth_window=args.smooth_window,
    )

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()

