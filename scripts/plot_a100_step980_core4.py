#!/usr/bin/env python3
"""
Plot Core-4 A100 tier0 loss curves truncated to a max step.

Inputs are JSON console logs containing lines like:
  {"integration_test_log_marker":"loss","step":...,"loss":...,"trained_batches":...}

This script writes:
  - full plot with raw + MA25
  - end-focus zoom plot
  - TSV export of step/loss/ma25 for each run
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunInput:
    label: str
    path: Path


@dataclass(frozen=True)
class RunSeries:
    label: str
    steps: np.ndarray
    loss: np.ndarray
    ma25: np.ndarray


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.astype(np.float64, copy=True)
    out = np.full_like(y, np.nan, dtype=np.float64)
    for i in range(len(y)):
        j = max(0, i - window + 1)
        out[i] = float(np.mean(y[j : i + 1]))
    return out


def parse_tier0_loss(path: Path, max_step: int) -> Tuple[np.ndarray, np.ndarray]:
    by_step: Dict[int, float] = {}
    if not path.exists():
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

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
            if int(obj.get("matformer_tier", -1)) != 0:
                continue

            step = obj.get("step")
            loss = obj.get("loss")
            trained_batches = obj.get("trained_batches")
            if not isinstance(step, int) or step > max_step:
                continue
            # Exclude post-failure rounds where no real training happened.
            if isinstance(trained_batches, int) and trained_batches <= 0:
                continue
            if not isinstance(loss, (int, float)):
                continue
            by_step.setdefault(step, float(loss))

    if not by_step:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    steps = np.array(sorted(by_step.keys()), dtype=np.int64)
    loss = np.array([by_step[int(s)] for s in steps], dtype=np.float64)
    return steps, loss


def load_runs(run_inputs: List[RunInput], max_step: int, smooth_window: int) -> List[RunSeries]:
    out: List[RunSeries] = []
    for run in run_inputs:
        steps, loss = parse_tier0_loss(run.path, max_step=max_step)
        if steps.size == 0:
            continue
        ma = moving_average(loss, smooth_window)
        out.append(RunSeries(label=run.label, steps=steps, loss=loss, ma25=ma))
    return out


def add_endpoint_labels(ax: plt.Axes, runs: List[RunSeries], x_offset: float) -> None:
    endpoints: List[Tuple[str, int, float, float]] = []
    for r in runs:
        endpoints.append((r.label, int(r.steps[-1]), float(r.loss[-1]), float(r.ma25[-1])))
    if not endpoints:
        return

    ys = [y for _, _, y, _ in endpoints]
    span = max(0.12, max(ys) - min(ys))
    ordered = sorted(endpoints, key=lambda x: x[2])
    for i, (label, step, loss, ma25) in enumerate(ordered):
        offset = (i - (len(ordered) - 1) / 2.0) * (0.08 * span)
        txt = f"{label}: s{step} loss={loss:.3f} ma25={ma25:.3f}"
        ax.annotate(
            txt,
            xy=(step, loss),
            xytext=(step + x_offset, loss + offset),
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.8},
            va="center",
        )


def set_zoom_ylim(ax: plt.Axes, ys: List[np.ndarray]) -> None:
    vals: List[float] = []
    for arr in ys:
        if arr.size:
            vals.extend(arr.tolist())
    if not vals:
        return
    y_min = min(vals)
    y_max = max(vals)
    span = max(0.08, y_max - y_min)
    pad = max(0.03, 0.18 * span)
    ax.set_ylim(y_min - pad, y_max + pad)


def plot_full(runs: List[RunSeries], out_path: Path, smooth_window: int, max_step: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(13, 7), dpi=180)
    for r in runs:
        ax.plot(r.steps, r.loss, alpha=0.18, linewidth=1.2, label=f"{r.label} (raw)")
        ax.plot(r.steps, r.ma25, linewidth=2.2, label=f"{r.label} (ma{smooth_window})")
        ax.scatter(
            [r.steps[-1]],
            [r.ma25[-1]],
            s=80,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    add_endpoint_labels(ax, runs, x_offset=8.0)
    ax.set_title(f"A100 Core-4 Tier0 Loss (step <= {max_step}, MA window={smooth_window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, max_step + 25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_endfocus(
    runs: List[RunSeries],
    out_path: Path,
    smooth_window: int,
    max_step: int,
    zoom_start: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(13, 7), dpi=180)
    zoom_y: List[np.ndarray] = []
    for r in runs:
        mask = r.steps >= zoom_start
        if np.any(mask):
            ax.plot(r.steps[mask], r.loss[mask], alpha=0.18, linewidth=1.2, label=f"{r.label} (raw)")
            ax.plot(r.steps[mask], r.ma25[mask], linewidth=2.2, label=f"{r.label} (ma{smooth_window})")
            zoom_y.append(r.ma25[mask])
        ax.scatter(
            [r.steps[-1]],
            [r.ma25[-1]],
            s=90,
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
        )

    add_endpoint_labels(ax, runs, x_offset=6.0)
    ax.set_title(f"A100 Core-4 Tier0 End Focus (steps {zoom_start}..{max_step})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(zoom_start, max_step + 18)
    set_zoom_ylim(ax, zoom_y)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_tsv(runs: List[RunSeries], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["run", "step", "loss", "ma25"])
        for r in runs:
            for step, loss, ma in zip(r.steps.tolist(), r.loss.tolist(), r.ma25.tolist()):
                w.writerow([r.label, int(step), f"{loss:.8f}", f"{ma:.8f}"])


def print_summary(runs: List[RunSeries]) -> None:
    print("Summary (tier0, filtered to trained_batches>0 and non-null loss):")
    for r in runs:
        print(
            f"- {r.label}: last_step={int(r.steps[-1])}, "
            f"last_loss={float(r.loss[-1]):.6f}, "
            f"last_ma25={float(r.ma25[-1]):.6f}, "
            f"best_ma25={float(np.min(r.ma25)):.6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner4-log", type=Path, required=True)
    ap.add_argument("--ls-inner1-log", type=Path, required=True)
    ap.add_argument("--lsolo-b400-log", type=Path, required=True)
    ap.add_argument("--lsolo-b200-log", type=Path, required=True)
    ap.add_argument("--max-step", type=int, default=980)
    ap.add_argument("--smooth-window", type=int, default=25)
    ap.add_argument("--zoom-start", type=int, default=700)
    ap.add_argument(
        "--out-main",
        type=Path,
        default=Path("plots/rendered/a100_step980_core4_tier0.png"),
    )
    ap.add_argument(
        "--out-endfocus",
        type=Path,
        default=Path("plots/rendered/a100_step980_core4_tier0_endfocus.png"),
    )
    ap.add_argument(
        "--out-tsv",
        type=Path,
        default=Path("plots/data/a100_step980_core4_tier0.tsv"),
    )
    args = ap.parse_args()

    runs = load_runs(
        [
            RunInput(label="L+S inner=4", path=args.inner4_log.expanduser()),
            RunInput(label="L+S inner=1", path=args.ls_inner1_log.expanduser()),
            RunInput(label="L-solo b=400", path=args.lsolo_b400_log.expanduser()),
            RunInput(label="L-solo b=200", path=args.lsolo_b200_log.expanduser()),
        ],
        max_step=max(0, args.max_step),
        smooth_window=max(1, args.smooth_window),
    )

    if not runs:
        raise SystemExit("No valid run data found after filtering.")

    plot_full(
        runs=runs,
        out_path=args.out_main.expanduser(),
        smooth_window=max(1, args.smooth_window),
        max_step=max(0, args.max_step),
    )
    plot_endfocus(
        runs=runs,
        out_path=args.out_endfocus.expanduser(),
        smooth_window=max(1, args.smooth_window),
        max_step=max(0, args.max_step),
        zoom_start=max(0, args.zoom_start),
    )
    write_tsv(runs=runs, out_path=args.out_tsv.expanduser())
    print_summary(runs)
    print(args.out_main.expanduser())
    print(args.out_endfocus.expanduser())
    print(args.out_tsv.expanduser())


if __name__ == "__main__":
    main()
