#!/usr/bin/env python3
"""
Plot loss curves from the A100 2000-step experiments (2026-02-16).

Inputs live under:
  plots/input/a100/20260216/*.console.log

These files are JSON log streams (stdout) that include structured "loss" events:
  {"integration_test_log_marker":"loss","step":...,"loss":...,"matformer_tier":...}
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


def _extract_losses(path: Path, tier: int) -> Dict[int, List[float]]:
    losses: Dict[int, List[float]] = {}
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

            losses.setdefault(step, []).append(float(loss))

    return losses


def _aggregate(losses_by_step: Dict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    steps = np.array(sorted(losses_by_step.keys()), dtype=np.int64)
    ys: List[float] = []
    for s in steps:
        vals = losses_by_step[int(s)]
        ys.append(float(vals[0]) if vals else float("nan"))
    return steps, np.array(ys, dtype=np.float64)


def _plot_one_axis(ax: plt.Axes, runs: List[RunSpec], title: str, smooth_window: int) -> None:
    for run in runs:
        by_step = _extract_losses(run.path, tier=run.tier)
        steps, y = _aggregate(by_step)
        if steps.size == 0:
            continue

        y_s = _moving_average(y, smooth_window)
        ax.plot(steps, y_s, linewidth=2.0, label=run.label)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (moving avg)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    inp = repo / "plots/input/a100/20260216"
    out = repo / "plots/rendered/a100_2000_runs_20260216.png"

    runs_tier0 = [
        RunSpec("sign baseline (tier0)", inp / "sign_baseline_tier0.console.log", tier=0),
        RunSpec("sign samebatch500 (tier0)", inp / "sign_samebatch500_tier0.console.log", tier=0),
        RunSpec("rawgeom_v2_p05 (tier0)", inp / "rawgeom_v2_p05_tier0.console.log", tier=0),
        RunSpec("apply0 (tier0)", inp / "apply0_tier0.console.log", tier=0),
        RunSpec("L-solo (tier0)", inp / "lsolo_tier0.console.log", tier=0),
    ]

    runs_tier1 = [
        RunSpec("sign baseline (tier1)", inp / "sign_baseline_tier1.console.log", tier=1),
        RunSpec("sign samebatch500 (tier1)", inp / "sign_samebatch500_tier1.console.log", tier=1),
        RunSpec("rawgeom_v2_p05 (tier1)", inp / "rawgeom_v2_p05_tier1.console.log", tier=1),
        RunSpec("apply0 (tier1)", inp / "apply0_tier1.console.log", tier=1),
    ]

    smooth_window = 25

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=160)
    _plot_one_axis(axes[0], runs_tier0, title="A100 FineWeb 2000-step: Tier0", smooth_window=smooth_window)
    _plot_one_axis(axes[1], runs_tier1, title="A100 FineWeb 2000-step: Tier1", smooth_window=smooth_window)

    fig.suptitle(f"A100 runs (2000 steps) - smoothed window={smooth_window}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

