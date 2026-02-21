#!/usr/bin/env python3
"""
Plot "cosine mixer shadow" telemetry from client logs.

This expects JSON logs containing entries like:
  {"message":"DisTrO cosine mixer stats (MatFormer prefix params)", "step":..., "mean_cos":..., "would_drop_frac":...}

Usage:
  python3 scripts/plot_cosine_shadow_stats.py --in <client.console.log> --out plots/rendered/<name>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def extract(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps: List[int] = []
    mean_cos: List[float] = []
    drop_frac: List[float] = []

    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        if obj.get("message") != "DisTrO cosine mixer stats (MatFormer prefix params)":
            continue
        step = obj.get("step")
        mc = obj.get("mean_cos")
        df = obj.get("would_drop_frac")
        if not isinstance(step, int):
            continue
        if not isinstance(mc, (int, float)):
            continue
        if not isinstance(df, (int, float)):
            continue

        steps.append(step)
        mean_cos.append(float(mc))
        drop_frac.append(float(df))

    if not steps:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Ensure sorted.
    order = np.argsort(np.array(steps, dtype=np.int64))
    steps = np.array(steps, dtype=np.int64)[order]
    mean_cos = np.array(mean_cos, dtype=np.float64)[order]
    drop_frac = np.array(drop_frac, dtype=np.float64)[order]
    return steps, mean_cos, drop_frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="client console log (JSON lines)")
    ap.add_argument("--out", dest="out", required=True, help="output PNG path")
    ap.add_argument("--smooth-window", type=int, default=25)
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    steps, mean_cos, drop_frac = extract(inp)
    if steps.size == 0:
        raise SystemExit(f"No cosine-shadow stats found in: {inp}")

    mc_s = _moving_average(mean_cos, args.smooth_window)
    df_s = _moving_average(drop_frac, max(5, args.smooth_window // 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=160)
    ax0, ax1 = axes

    ax0.plot(steps, mc_s, linewidth=2.0)
    ax0.set_title(f"Mean Cosine (smoothed window={args.smooth_window})")
    ax0.set_xlabel("Step")
    ax0.set_ylabel("mean_cos")
    ax0.grid(True, alpha=0.25)

    ax1.plot(steps, df_s, linewidth=2.0)
    ax1.set_title("Would-Drop Fraction (smoothed)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("would_drop_frac")
    ax1.set_ylim(0.0, min(1.0, max(0.2, float(np.nanmax(df_s) * 1.3))))
    ax1.grid(True, alpha=0.25)

    fig.suptitle(f"Cosine Mixer Shadow Telemetry: {inp.name}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

