#!/usr/bin/env python3
"""
Enhanced comparison plot for A100 500-step matrix runs.

Focus:
- clearer endpoint differences
- larger end markers
- zoomed endpoint panels (default: steps >= 400)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunSeries:
    label: str
    path: Path
    tier0_steps: np.ndarray
    tier0_loss: np.ndarray
    tier1_steps: np.ndarray
    tier1_loss: np.ndarray
    tier0_heldout: Optional[float]
    tier1_heldout: Optional[float]


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.astype(np.float64, copy=True)
    out = np.full_like(y, np.nan, dtype=np.float64)
    total = 0.0
    buf: List[float] = []
    for i, v in enumerate(y):
        fv = float(v)
        buf.append(fv)
        total += fv
        if len(buf) > window:
            total -= buf.pop(0)
        out[i] = total / len(buf)
    return out


def load_loss_series(path: Path, tier: int) -> Tuple[np.ndarray, np.ndarray]:
    by_step: Dict[int, float] = {}
    if not path.exists():
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    with path.open("r", errors="ignore") as f:
        for line in f:
            if '"integration_test_log_marker":"loss"' not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if int(obj.get("matformer_tier", -1)) != tier:
                continue
            step = obj.get("step")
            loss = obj.get("loss")
            if not isinstance(step, int):
                continue
            if not isinstance(loss, (int, float)):
                continue
            by_step.setdefault(step, float(loss))
    if not by_step:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    steps = np.array(sorted(by_step.keys()), dtype=np.int64)
    vals = np.array([by_step[int(s)] for s in steps], dtype=np.float64)
    return steps, vals


def load_heldout(path: Path, tier: int) -> Optional[float]:
    if not path.exists():
        return None
    last = None
    with path.open("r", errors="ignore") as f:
        for line in f:
            if '"integration_test_log_marker":"heldout_eval"' not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if int(obj.get("matformer_tier", -1)) != tier:
                continue
            v = obj.get("heldout_loss")
            if isinstance(v, (int, float)):
                last = float(v)
    return last


def load_run(label: str, run_dir: Path) -> RunSeries:
    t0_log = run_dir / "client-tier0.console.log"
    t1_log = run_dir / "client-tier1.console.log"
    t0_steps, t0_loss = load_loss_series(t0_log, tier=0)
    t1_steps, t1_loss = load_loss_series(t1_log, tier=1)
    t0_heldout = load_heldout(t0_log, tier=0)
    t1_heldout = load_heldout(t1_log, tier=1)
    return RunSeries(
        label=label,
        path=run_dir,
        tier0_steps=t0_steps,
        tier0_loss=t0_loss,
        tier1_steps=t1_steps,
        tier1_loss=t1_loss,
        tier0_heldout=t0_heldout,
        tier1_heldout=t1_heldout,
    )


def add_endpoint_labels(
    ax: plt.Axes,
    endpoints: List[Tuple[str, int, float, Optional[float], str]],
    x_offset: float,
) -> None:
    if not endpoints:
        return
    ys = [y for _, _, y, _, _ in endpoints]
    span = max(0.12, max(ys) - min(ys))
    for i, (label, x, y, held, color) in enumerate(sorted(endpoints, key=lambda v: v[2])):
        offset = (i - (len(endpoints) - 1) / 2.0) * (0.08 * span)
        text = f"{label}: {y:.3f}"
        if held is not None:
            text += f" | val {held:.3f}"
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x + x_offset, y + offset),
            fontsize=8.5,
            color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": color},
            va="center",
        )


def _set_zoom_ylim(
    ax: plt.Axes,
    y_zoom_values: List[np.ndarray],
    pad_ratio: float = 0.18,
    min_span: float = 0.08,
) -> None:
    ys: List[float] = []
    for arr in y_zoom_values:
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size:
            ys.extend(float(v) for v in finite.tolist())
    if not ys:
        return
    y_min = min(ys)
    y_max = max(ys)
    span = max(min_span, y_max - y_min)
    pad = max(0.03, span * pad_ratio)
    ax.set_ylim(y_min - pad, y_max + pad)


def plot_enhanced(
    runs: List[RunSeries],
    out_path: Path,
    smooth_window: int,
    zoom_start_step: int,
    end_marker_size: float,
) -> None:
    colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9), dpi=170)
    # Row 0: tier0, row 1: tier1. Col 0: full, col 1: zoom.
    t0_full, t0_zoom = axes[0, 0], axes[0, 1]
    t1_full, t1_zoom = axes[1, 0], axes[1, 1]

    max_step_seen = 0
    t0_endpoints: List[Tuple[str, int, float, Optional[float], str]] = []
    t1_endpoints: List[Tuple[str, int, float, Optional[float], str]] = []
    t0_zoom_y: List[np.ndarray] = []
    t1_zoom_y: List[np.ndarray] = []

    for idx, run in enumerate(runs):
        color = colors(idx % 10)
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
        )

        if run.tier0_steps.size:
            y0 = moving_average(run.tier0_loss, smooth_window)
            x0 = run.tier0_steps
            max_step_seen = max(max_step_seen, int(x0[-1]))
            t0_full.plot(x0, y0, lw=2.0, color=color, label=run.label)
            t0_zoom.plot(x0, y0, lw=2.0, color=color, label=run.label)
            t0_full.scatter(
                [x0[-1]],
                [y0[-1]],
                s=end_marker_size,
                color=[color],
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            t0_zoom.scatter(
                [x0[-1]],
                [y0[-1]],
                s=end_marker_size * 1.1,
                color=[color],
                edgecolors="black",
                linewidths=0.6,
                zorder=6,
            )
            t0_endpoints.append(
                (run.label, int(x0[-1]), float(y0[-1]), run.tier0_heldout, color_hex)
            )
            zoom_mask0 = x0 >= zoom_start_step
            if np.any(zoom_mask0):
                t0_zoom_y.append(y0[zoom_mask0])

        if run.tier1_steps.size:
            y1 = moving_average(run.tier1_loss, smooth_window)
            x1 = run.tier1_steps
            max_step_seen = max(max_step_seen, int(x1[-1]))
            t1_full.plot(x1, y1, lw=2.0, color=color, label=run.label)
            t1_zoom.plot(x1, y1, lw=2.0, color=color, label=run.label)
            t1_full.scatter(
                [x1[-1]],
                [y1[-1]],
                s=end_marker_size,
                color=[color],
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            t1_zoom.scatter(
                [x1[-1]],
                [y1[-1]],
                s=end_marker_size * 1.1,
                color=[color],
                edgecolors="black",
                linewidths=0.6,
                zorder=6,
            )
            t1_endpoints.append(
                (run.label, int(x1[-1]), float(y1[-1]), run.tier1_heldout, color_hex)
            )
            zoom_mask1 = x1 >= zoom_start_step
            if np.any(zoom_mask1):
                t1_zoom_y.append(y1[zoom_mask1])

    for ax in [t0_full, t0_zoom, t1_full, t1_zoom]:
        ax.grid(True, alpha=0.28)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

    t0_full.set_title("Tier0 loss (full)")
    t0_zoom.set_title(f"Tier0 loss (zoom: step >= {zoom_start_step})")
    t1_full.set_title("Tier1 loss (full)")
    t1_zoom.set_title(f"Tier1 loss (zoom: step >= {zoom_start_step})")

    if max_step_seen <= 0:
        max_step_seen = 500
    t0_zoom.set_xlim(zoom_start_step, max_step_seen + 18)
    t1_zoom.set_xlim(zoom_start_step, max_step_seen + 18)
    _set_zoom_ylim(t0_zoom, t0_zoom_y)
    _set_zoom_ylim(t1_zoom, t1_zoom_y)

    t0_full.axvline(zoom_start_step, color="gray", ls="--", lw=1.0, alpha=0.7)
    t1_full.axvline(zoom_start_step, color="gray", ls="--", lw=1.0, alpha=0.7)

    add_endpoint_labels(t0_zoom, t0_endpoints, x_offset=6.0)
    add_endpoint_labels(t1_zoom, t1_endpoints, x_offset=6.0)

    t0_full.legend(loc="upper right", fontsize=8.5)
    t1_full.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        f"A100 500-step matrix (moving avg window={smooth_window})\n"
        "Large endpoint markers + callouts show final train and held-out losses",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-b400-dir", required=True, type=Path)
    ap.add_argument("--lsolo-b400-dir", required=True, type=Path)
    ap.add_argument("--lsolo-b200-dir", required=True, type=Path)
    ap.add_argument("--inner3-dir", required=True, type=Path)
    ap.add_argument("--inner4-dir", required=True, type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("plots/a100_matrix_500_heldout_20260220_enhanced.png"),
    )
    ap.add_argument("--smooth-window", type=int, default=25)
    ap.add_argument("--zoom-start-step", type=int, default=400)
    ap.add_argument("--end-marker-size", type=float, default=85.0)
    args = ap.parse_args()

    runs = [
        load_run("L+S b=400", args.ls_b400_dir.expanduser()),
        load_run("L-solo b=400", args.lsolo_b400_dir.expanduser()),
        load_run("L-solo b=200", args.lsolo_b200_dir.expanduser()),
        load_run("inner=3", args.inner3_dir.expanduser()),
        load_run("inner=4", args.inner4_dir.expanduser()),
    ]

    plot_enhanced(
        runs=runs,
        out_path=args.out.expanduser(),
        smooth_window=max(1, args.smooth_window),
        zoom_start_step=max(0, args.zoom_start_step),
        end_marker_size=max(10.0, args.end_marker_size),
    )
    print(args.out.expanduser())


if __name__ == "__main__":
    main()
