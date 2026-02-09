#!/usr/bin/env python3
"""
Generic "batch hardness" plotter for Psyche client logs.

Hardness is a de-trended per-step loss:
  hardness(step) = loss(step) - moving_avg(loss, window)

This is useful for verifying that stepwise spikes come from data difficulty /
batch assignment rather than optimizer regressions.

Usage:
  python3 scripts/plot_batch_hardness.py \\
    --tier0-log logs/.../client-tier0.console.log \\
    --tier1-log logs/.../client-tier1.console.log \\
    --out plots/rendered/batch_hardness.png
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class StepInfo:
    loss: float
    batch_str: str


_BATCH_RE = re.compile(r"B\[[^\]]+\]")


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    out = np.full_like(y, np.nan, dtype=np.float64)
    for i in range(len(y)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(y[start : i + 1]))
    return out


def _extract_steps(log_path: Path, tier: int) -> Dict[int, StepInfo]:
    """
    Return step -> {loss, assigned_batches_str}.

    We join:
      - integration_test_log_marker=="loss" for the loss
      - integration_test_log_marker=="witness_elected" for assigned_batches
    """
    loss_by_step: Dict[int, float] = {}
    batch_by_step: Dict[int, str] = {}

    for line in log_path.read_text(errors="ignore").splitlines():
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        if int(obj.get("matformer_tier", -1)) != tier:
            continue

        step = obj.get("step")
        if not isinstance(step, int):
            continue

        marker = obj.get("integration_test_log_marker")
        if marker == "loss":
            loss = obj.get("loss")
            if isinstance(loss, (int, float)):
                loss_by_step.setdefault(step, float(loss))
        elif marker == "witness_elected":
            assigned = obj.get("assigned_batches")
            if isinstance(assigned, str):
                m = _BATCH_RE.search(assigned)
                batch_by_step.setdefault(step, m.group(0) if m else assigned)

    out: Dict[int, StepInfo] = {}
    for step, loss in loss_by_step.items():
        out[step] = StepInfo(loss=loss, batch_str=batch_by_step.get(step, ""))
    return out


def _make_series(step_info: Dict[int, StepInfo]) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    steps = np.array(sorted(step_info.keys()), dtype=np.int64)
    losses = np.array([step_info[int(s)].loss for s in steps], dtype=np.float64)
    batch_strs = {int(s): step_info[int(s)].batch_str for s in steps}
    return steps, losses, batch_strs


def _annotate_topk(ax: plt.Axes, steps: np.ndarray, hardness: np.ndarray, batch_strs: Dict[int, str], k: int) -> None:
    if steps.size == 0:
        return
    idx = np.argsort(hardness)[::-1][:k]
    for i in idx:
        s = int(steps[i])
        h = float(hardness[i])
        b = batch_strs.get(s, "")
        label = f"step {s}\\n{b}" if b else f"step {s}"
        ax.annotate(
            label,
            xy=(s, h),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )


def _plot(ax: plt.Axes, steps: np.ndarray, hardness: np.ndarray, title: str, batch_strs: Dict[int, str], annotate_k: int) -> None:
    ax.plot(steps, hardness, linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Hardness = loss - moving_avg(loss)")
    ax.grid(True, alpha=0.25)
    _annotate_topk(ax, steps, hardness, batch_strs=batch_strs, k=annotate_k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier0-log", required=True, help="tier0 client console log (JSON lines)")
    ap.add_argument("--tier1-log", default="", help="tier1 client console log (JSON lines)")
    ap.add_argument("--window", type=int, default=25, help="moving average window")
    ap.add_argument("--annotate-k", type=int, default=8, help="annotate top-k hardest steps")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--title", default="Batch hardness", help="plot title")
    args = ap.parse_args()

    p0 = Path(args.tier0_log)
    p1 = Path(args.tier1_log) if args.tier1_log else None
    out = Path(args.out)

    s0 = _extract_steps(p0, tier=0)
    steps0, loss0, batches0 = _make_series(s0)
    hard0 = loss0 - _moving_average(loss0, args.window)

    if p1 is not None and p1.exists():
        s1 = _extract_steps(p1, tier=1)
        steps1, loss1, batches1 = _make_series(s1)
        hard1 = loss1 - _moving_average(loss1, args.window)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=160)
        _plot(axes[0], steps0, hard0, title=f"Tier0 batch hardness (window={args.window})", batch_strs=batches0, annotate_k=args.annotate_k)
        _plot(axes[1], steps1, hard1, title=f"Tier1 batch hardness (window={args.window})", batch_strs=batches1, annotate_k=args.annotate_k)
        fig.suptitle(args.title)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=160)
        _plot(ax, steps0, hard0, title=f"Tier0 batch hardness (window={args.window})", batch_strs=batches0, annotate_k=args.annotate_k)
        fig.suptitle(args.title)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
