#!/usr/bin/env python3
"""
Batch hardness plot for the A100 FineWeb 2000-step run (2026-02-16).

Goal: show that the stepwise spikes are strongly data-driven under aligned batches
by plotting a de-trended per-step loss ("hardness") and annotating the hardest
steps with their assigned BatchIds.

Input:
  plots/input/a100/20260216/sign_baseline_tier0.console.log
  plots/input/a100/20260216/sign_baseline_tier1.console.log

Output:
  plots/rendered/a100_2000_batch_hardness_sign_baseline_20260216.png
"""

from __future__ import annotations

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

    with log_path.open("rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore").strip()
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
                    # Keep it short but informative: "[B[2, 3]]" -> "B[2, 3]"
                    m = _BATCH_RE.search(assigned)
                    batch_by_step.setdefault(step, m.group(0) if m else assigned)

    out: Dict[int, StepInfo] = {}
    for step, loss in loss_by_step.items():
        batch_str = batch_by_step.get(step, "")
        out[step] = StepInfo(loss=loss, batch_str=batch_str)
    return out


def _make_series(step_info: Dict[int, StepInfo]) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    steps = np.array(sorted(step_info.keys()), dtype=np.int64)
    losses = np.array([step_info[int(s)].loss for s in steps], dtype=np.float64)
    batch_strs = {int(s): step_info[int(s)].batch_str for s in steps}
    return steps, losses, batch_strs


def _annotate_topk(ax: plt.Axes, steps: np.ndarray, hardness: np.ndarray, batch_strs: Dict[int, str], k: int) -> None:
    # Pick top-k hardest (largest positive hardness).
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


def _plot(ax: plt.Axes, steps: np.ndarray, hardness: np.ndarray, title: str, batch_strs: Dict[int, str]) -> None:
    ax.plot(steps, hardness, linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Step (aligned batches)")
    ax.set_ylabel("Hardness = loss - moving_avg(loss)")
    ax.grid(True, alpha=0.25)
    _annotate_topk(ax, steps, hardness, batch_strs=batch_strs, k=8)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    inp = repo / "plots/input/a100/20260216"
    out = repo / "plots/rendered/a100_2000_batch_hardness_sign_baseline_20260216.png"

    p0 = inp / "sign_baseline_tier0.console.log"
    p1 = inp / "sign_baseline_tier1.console.log"
    if not p0.exists() or not p1.exists():
        raise SystemExit(f"Missing inputs under {inp}. Expected sign_baseline_tier0/1.console.log")

    s0 = _extract_steps(p0, tier=0)
    s1 = _extract_steps(p1, tier=1)

    steps0, loss0, batches0 = _make_series(s0)
    steps1, loss1, batches1 = _make_series(s1)

    window = 25
    hard0 = loss0 - _moving_average(loss0, window)
    hard1 = loss1 - _moving_average(loss1, window)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=160)
    _plot(axes[0], steps0, hard0, title=f"Tier0 batch hardness (window={window})", batch_strs=batches0)
    _plot(axes[1], steps1, hard1, title=f"Tier1 batch hardness (window={window})", batch_strs=batches1)

    fig.suptitle("A100 FineWeb 2000-step: batch hardness (sign baseline)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
