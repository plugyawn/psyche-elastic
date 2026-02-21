#!/usr/bin/env python3
"""
Delta-to-baseline plots for A100 2000-step experiments (2026-02-16).

This complements `scripts/plot_a100_runs_20260216.py` by making small differences
visible: it plots (variant - sign_baseline) for tier0 and tier1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    tier: int


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    out = np.full_like(y, np.nan, dtype=np.float64)
    for i in range(len(y)):
        start = max(0, i - window + 1)
        chunk = y[start : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            out[i] = float(np.mean(chunk))
    return out


def _extract_losses(path: Path, tier: int) -> Dict[int, float]:
    losses: Dict[int, float] = {}
    if not path.exists():
        return losses

    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("integration_test_log_marker") != "loss":
                continue
            if int(obj.get("matformer_tier", -1)) != tier:
                continue

            step = obj.get("step")
            loss = obj.get("loss")
            if not isinstance(step, int):
                continue
            if not isinstance(loss, (int, float)):
                continue

            # Keep the first observed loss per step (consistent with previous plotting).
            losses.setdefault(step, float(loss))

    return losses


def _to_series(losses_by_step: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    steps = np.array(sorted(losses_by_step.keys()), dtype=np.int64)
    y = np.array([losses_by_step[int(s)] for s in steps], dtype=np.float64)
    return steps, y


def _align(steps_a: np.ndarray, y_a: np.ndarray, steps_b: np.ndarray, y_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return steps, a, b aligned on common steps."""
    set_a = set(map(int, steps_a))
    set_b = set(map(int, steps_b))
    common = sorted(set_a.intersection(set_b))
    if not common:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    idx_a = {int(s): i for i, s in enumerate(steps_a)}
    idx_b = {int(s): i for i, s in enumerate(steps_b)}
    aa = np.array([y_a[idx_a[s]] for s in common], dtype=np.float64)
    bb = np.array([y_b[idx_b[s]] for s in common], dtype=np.float64)
    return np.array(common, dtype=np.int64), aa, bb


def _plot_delta(ax: plt.Axes, baseline: RunSpec, variants: List[RunSpec], smooth_window: int) -> None:
    b_steps, b_y = _to_series(_extract_losses(baseline.path, tier=baseline.tier))
    b_y_s = _moving_average(b_y, smooth_window)

    for v in variants:
        v_steps, v_y = _to_series(_extract_losses(v.path, tier=v.tier))
        v_y_s = _moving_average(v_y, smooth_window)

        steps, b_al, v_al = _align(b_steps, b_y_s, v_steps, v_y_s)
        if steps.size == 0:
            continue
        ax.plot(steps, v_al - b_al, linewidth=2.0, label=v.label)

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss delta (variant - baseline)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    inp = repo / "plots/input/a100/20260216"
    out = repo / "plots/rendered/a100_2000_runs_delta_to_sign_20260216.png"

    smooth_window = 25

    b0 = RunSpec("sign baseline", inp / "sign_baseline_tier0.console.log", tier=0)
    b1 = RunSpec("sign baseline", inp / "sign_baseline_tier1.console.log", tier=1)

    variants0 = [
        RunSpec("sign samebatch500", inp / "sign_samebatch500_tier0.console.log", tier=0),
        RunSpec("rawgeom_v2_p05", inp / "rawgeom_v2_p05_tier0.console.log", tier=0),
        RunSpec("apply0", inp / "apply0_tier0.console.log", tier=0),
        RunSpec("L-solo", inp / "lsolo_tier0.console.log", tier=0),
    ]
    variants1 = [
        RunSpec("sign samebatch500", inp / "sign_samebatch500_tier1.console.log", tier=1),
        RunSpec("rawgeom_v2_p05", inp / "rawgeom_v2_p05_tier1.console.log", tier=1),
        RunSpec("apply0", inp / "apply0_tier1.console.log", tier=1),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=160)
    axes[0].set_title(f"Tier0 delta to sign baseline (smoothed window={smooth_window})")
    _plot_delta(axes[0], b0, variants0, smooth_window=smooth_window)
    axes[1].set_title(f"Tier1 delta to sign baseline (smoothed window={smooth_window})")
    _plot_delta(axes[1], b1, variants1, smooth_window=smooth_window)

    fig.suptitle("A100 runs (2000 steps): delta-to-sign-baseline")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

