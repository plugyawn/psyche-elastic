#!/usr/bin/env python3
"""
Plot the two stalled A100 5000-step runs:
  - L-solo b=200 (tier0)
  - L+S inner=4 (tier0, tier1 optional)

Filters out untrained tail points (trained_batches <= 0).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Series:
    steps: np.ndarray
    loss: np.ndarray
    ma: np.ndarray


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.astype(np.float64, copy=True)
    out = np.full_like(y, np.nan, dtype=np.float64)
    for i in range(len(y)):
        j = max(0, i - window + 1)
        out[i] = float(np.mean(y[j : i + 1]))
    return out


def parse_loss(path: Path, tier: int) -> Tuple[np.ndarray, np.ndarray]:
    by_step: Dict[int, float] = {}
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
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
            trained_batches = obj.get("trained_batches")
            if not isinstance(step, int):
                continue
            if not (isinstance(trained_batches, int) and trained_batches > 0):
                continue
            if not isinstance(loss, (int, float)):
                continue
            by_step.setdefault(step, float(loss))
    if not by_step:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    steps = np.array(sorted(by_step.keys()), dtype=np.int64)
    loss = np.array([by_step[int(s)] for s in steps], dtype=np.float64)
    return steps, loss


def load_series(path: Path, tier: int, smooth_window: int) -> Series:
    steps, loss = parse_loss(path, tier=tier)
    if steps.size == 0:
        raise SystemExit(f"No valid loss points in {path} for tier={tier}")
    return Series(steps=steps, loss=loss, ma=moving_average(loss, smooth_window))


def trim_to_step(s: Series, max_step: int) -> Series:
    mask = s.steps <= max_step
    return Series(steps=s.steps[mask], loss=s.loss[mask], ma=s.ma[mask])


def annotate_end(ax: plt.Axes, label: str, s: Series, dx: float = 6.0, dy: float = 0.0) -> None:
    ax.scatter([s.steps[-1]], [s.ma[-1]], s=90, edgecolors="black", linewidths=0.6, zorder=6)
    ax.annotate(
        f"{label}: s{s.steps[-1]} ma={s.ma[-1]:.4f}",
        xy=(s.steps[-1], s.ma[-1]),
        xytext=(s.steps[-1] + dx, s.ma[-1] + dy),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "lw": 0.8},
        va="center",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsolo-tier0-log", type=Path, required=True)
    ap.add_argument("--ls-inner4-tier0-log", type=Path, required=True)
    ap.add_argument("--ls-inner4-tier1-log", type=Path, default=None)
    ap.add_argument("--smooth-window", type=int, default=25)
    ap.add_argument(
        "--out-overlap",
        type=Path,
        default=Path("plots/rendered/a100_5000_stalled_overlap_tier0.png"),
    )
    ap.add_argument(
        "--out-full",
        type=Path,
        default=Path("plots/rendered/a100_5000_stalled_full_tier0.png"),
    )
    args = ap.parse_args()

    lsolo = load_series(args.lsolo_tier0_log.expanduser(), tier=0, smooth_window=max(1, args.smooth_window))
    ls_t0 = load_series(args.ls_inner4_tier0_log.expanduser(), tier=0, smooth_window=max(1, args.smooth_window))
    ls_t1 = None
    if args.ls_inner4_tier1_log is not None:
        ls_t1 = load_series(args.ls_inner4_tier1_log.expanduser(), tier=1, smooth_window=max(1, args.smooth_window))

    overlap_end = int(min(lsolo.steps[-1], ls_t0.steps[-1]))
    lsolo_overlap = trim_to_step(lsolo, overlap_end)
    ls_t0_overlap = trim_to_step(ls_t0, overlap_end)
    ls_t1_overlap = trim_to_step(ls_t1, overlap_end) if ls_t1 is not None else None

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.5), dpi=180)
    ax.plot(lsolo_overlap.steps, lsolo_overlap.loss, alpha=0.18, linewidth=1.0, label="L-solo b=200 tier0 (raw)")
    ax.plot(lsolo_overlap.steps, lsolo_overlap.ma, linewidth=2.0, label="L-solo b=200 tier0 (ma)")
    ax.plot(ls_t0_overlap.steps, ls_t0_overlap.loss, alpha=0.18, linewidth=1.0, label="L+S inner=4 tier0 (raw)")
    ax.plot(ls_t0_overlap.steps, ls_t0_overlap.ma, linewidth=2.0, label="L+S inner=4 tier0 (ma)")
    if ls_t1_overlap is not None and ls_t1_overlap.steps.size:
        ax.plot(ls_t1_overlap.steps, ls_t1_overlap.ma, linewidth=1.8, linestyle="--", label="L+S inner=4 tier1 (ma)")

    annotate_end(ax, "L-solo", lsolo_overlap, dx=6.0, dy=-0.03)
    annotate_end(ax, "L+S tier0", ls_t0_overlap, dx=6.0, dy=0.03)
    if ls_t1_overlap is not None and ls_t1_overlap.steps.size:
        annotate_end(ax, "L+S tier1", ls_t1_overlap, dx=6.0, dy=-0.08)

    ax.set_title(f"A100 stalled 5000 runs: overlap view (step <= {overlap_end})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, overlap_end + 20)
    fig.tight_layout()
    args.out_overlap.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_overlap.expanduser())
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.5), dpi=180)
    ax.plot(lsolo.steps, lsolo.loss, alpha=0.15, linewidth=1.0, label="L-solo b=200 tier0 (raw)")
    ax.plot(lsolo.steps, lsolo.ma, linewidth=2.0, label="L-solo b=200 tier0 (ma)")
    ax.plot(ls_t0.steps, ls_t0.loss, alpha=0.15, linewidth=1.0, label="L+S inner=4 tier0 (raw)")
    ax.plot(ls_t0.steps, ls_t0.ma, linewidth=2.0, label="L+S inner=4 tier0 (ma)")
    if ls_t1 is not None and ls_t1.steps.size:
        ax.plot(ls_t1.steps, ls_t1.ma, linewidth=1.8, linestyle="--", label="L+S inner=4 tier1 (ma)")

    annotate_end(ax, "L-solo", lsolo, dx=10.0, dy=-0.03)
    annotate_end(ax, "L+S tier0", ls_t0, dx=10.0, dy=0.03)
    if ls_t1 is not None and ls_t1.steps.size:
        annotate_end(ax, "L+S tier1", ls_t1, dx=10.0, dy=-0.08)

    ax.set_title("A100 stalled 5000 runs: full observed range")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, max(int(lsolo.steps[-1]), int(ls_t0.steps[-1])) + 30)
    fig.tight_layout()
    args.out_full.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_full.expanduser())
    plt.close(fig)

    print(f"L-solo b=200 tier0 last trained step: {int(lsolo.steps[-1])}, ma={float(lsolo.ma[-1]):.6f}")
    print(f"L+S inner4 tier0 last trained step: {int(ls_t0.steps[-1])}, ma={float(ls_t0.ma[-1]):.6f}")
    if ls_t1 is not None and ls_t1.steps.size:
        print(f"L+S inner4 tier1 last trained step: {int(ls_t1.steps[-1])}, ma={float(ls_t1.ma[-1]):.6f}")
    print(f"overlap_end_step={overlap_end}")
    print(args.out_overlap.expanduser())
    print(args.out_full.expanduser())


if __name__ == "__main__":
    main()
