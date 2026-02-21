#!/usr/bin/env python3
"""
Plot helper for the A100 NanoGPT-20m batch-scaling experiment (500 steps).

Inputs are run directories containing:
  client-tier0.console.log
  client-tier1.console.log (optional)

These are JSON log streams (stdout) that include:
  {"integration_test_log_marker":"loss","step":...,"loss":...,"matformer_tier":...}
and (for L+S runs with cosine shadow enabled):
  {"message":"DisTrO cosine mixer stats (MatFormer prefix params)", "step":..., "mean_cos":..., "would_drop_frac":...}

Outputs:
  <out_dir>/<out_prefix>_tier0_loss.png
  <out_dir>/<out_prefix>_global_loss.png
  <out_dir>/<out_prefix>_cosine_shadow.png
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class LossSeries:
    steps: np.ndarray
    loss: np.ndarray


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
    for line in path.read_text(errors="ignore").splitlines():
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
        losses.setdefault(step, float(loss))
    return losses


def _to_series(losses_by_step: Dict[int, float]) -> LossSeries:
    steps = np.array(sorted(losses_by_step.keys()), dtype=np.int64)
    y = np.array([losses_by_step[int(s)] for s in steps], dtype=np.float64)
    return LossSeries(steps=steps, loss=y)


def _align(a: LossSeries, b: LossSeries) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_a = set(map(int, a.steps))
    set_b = set(map(int, b.steps))
    common = sorted(set_a.intersection(set_b))
    if not common:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    idx_a = {int(s): i for i, s in enumerate(a.steps)}
    idx_b = {int(s): i for i, s in enumerate(b.steps)}
    aa = np.array([a.loss[idx_a[s]] for s in common], dtype=np.float64)
    bb = np.array([b.loss[idx_b[s]] for s in common], dtype=np.float64)
    return np.array(common, dtype=np.int64), aa, bb


def _extract_cosine_shadow(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps: List[int] = []
    mean_cos: List[float] = []
    drop_frac: List[float] = []
    if not path.exists():
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
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
    order = np.argsort(np.array(steps, dtype=np.int64))
    steps = np.array(steps, dtype=np.int64)[order]
    mean_cos = np.array(mean_cos, dtype=np.float64)[order]
    drop_frac = np.array(drop_frac, dtype=np.float64)[order]
    return steps, mean_cos, drop_frac


def _mean_window(series: LossSeries, lo: int, hi: int) -> float:
    xs = [v for s, v in zip(series.steps.tolist(), series.loss.tolist()) if lo <= int(s) <= hi]
    return float(statistics.mean(xs)) if xs else float("nan")


def _find_console(run_dir: Path, tier: int) -> Path:
    return run_dir / f"client-tier{tier}.console.log"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsolo-b200", type=str, required=True, help="run dir for L-solo b=200 (min_clients=1)")
    ap.add_argument("--ls-b400", type=str, required=True, help="run dir for L+S b=400 (min_clients=2)")
    ap.add_argument("--lsolo-b400", type=str, default="", help="optional run dir for L-solo b=400 (min_clients=1)")
    ap.add_argument("--out-dir", type=str, default="plots/rendered", help="output directory")
    ap.add_argument("--out-prefix", type=str, default="a100_batchscale_500", help="output filename prefix")
    ap.add_argument("--smooth-window", type=int, default=25)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    d_lsolo_b200 = Path(args.lsolo_b200).expanduser()
    d_ls_b400 = Path(args.ls_b400).expanduser()
    d_lsolo_b400 = Path(args.lsolo_b400).expanduser() if args.lsolo_b400 else None

    # Load series
    lsolo_b200_t0 = _to_series(_extract_losses(_find_console(d_lsolo_b200, 0), tier=0))
    ls_b400_t0 = _to_series(_extract_losses(_find_console(d_ls_b400, 0), tier=0))
    ls_b400_t1 = _to_series(_extract_losses(_find_console(d_ls_b400, 1), tier=1))
    lsolo_b400_t0: Optional[LossSeries] = None
    if d_lsolo_b400 is not None:
        lsolo_b400_t0 = _to_series(_extract_losses(_find_console(d_lsolo_b400, 0), tier=0))

    # ---- Tier0 plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=160)
    for label, series in [
        ("L-solo b=200 (tier0)", lsolo_b200_t0),
        ("L+S b=400 (tier0)", ls_b400_t0),
    ] + ([("L-solo b=400 (tier0)", lsolo_b400_t0)] if lsolo_b400_t0 is not None else []):
        y = _moving_average(series.loss, args.smooth_window)
        ax.plot(series.steps, y, linewidth=2.0, label=label)
    ax.set_title(f"Tier0 loss (smoothed window={args.smooth_window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    out_t0 = out_dir / f"{args.out_prefix}_tier0_loss.png"
    fig.tight_layout()
    fig.savefig(out_t0)
    plt.close(fig)
    print(f"Wrote: {out_t0}")

    # ---- Global plot (only meaningful for L+S and optional L-solo b=400)
    steps, a0, a1 = _align(ls_b400_t0, ls_b400_t1)
    global_ls = LossSeries(steps=steps, loss=(a0 + a1) / 2.0) if steps.size else LossSeries(steps=np.array([], dtype=np.int64), loss=np.array([], dtype=np.float64))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=160)
    if global_ls.steps.size:
        ax.plot(global_ls.steps, _moving_average(global_ls.loss, args.smooth_window), linewidth=2.0, label="L+S b=400 (global=(t0+t1)/2)")
    if lsolo_b400_t0 is not None and lsolo_b400_t0.steps.size:
        ax.plot(lsolo_b400_t0.steps, _moving_average(lsolo_b400_t0.loss, args.smooth_window), linewidth=2.0, label="L-solo b=400 (tier0)")
    ax.set_title(f"Global loss (smoothed window={args.smooth_window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    out_gl = out_dir / f"{args.out_prefix}_global_loss.png"
    fig.tight_layout()
    fig.savefig(out_gl)
    plt.close(fig)
    print(f"Wrote: {out_gl}")

    # ---- Cosine shadow plot (from L+S tier0 log)
    c_steps, mean_cos, drop_frac = _extract_cosine_shadow(_find_console(d_ls_b400, 0))
    if c_steps.size:
        mc_s = _moving_average(mean_cos, args.smooth_window)
        df_s = _moving_average(drop_frac, max(5, args.smooth_window // 2))
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=160)
        ax0, ax1 = axes
        ax0.plot(c_steps, mc_s, linewidth=2.0)
        ax0.set_title(f"Mean cosine (smoothed window={args.smooth_window})")
        ax0.set_xlabel("Step")
        ax0.set_ylabel("mean_cos")
        ax0.grid(True, alpha=0.25)
        ax1.plot(c_steps, df_s, linewidth=2.0)
        ax1.set_title("Would-drop fraction (smoothed)")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("would_drop_frac")
        ax1.set_ylim(0.0, min(1.0, max(0.2, float(np.nanmax(df_s) * 1.3))))
        ax1.grid(True, alpha=0.25)
        fig.suptitle("Cosine mixer shadow telemetry (L+S b=400)")
        fig.tight_layout()
        out_cs = out_dir / f"{args.out_prefix}_cosine_shadow.png"
        fig.savefig(out_cs)
        plt.close(fig)
        print(f"Wrote: {out_cs}")
    else:
        print("No cosine-shadow stats found (did you set PSYCHE_DISTRO_COSINE_MIXER_SHADOW=1?)")

    # ---- Quick numeric summary (useful when scanning logs)
    lo, hi = 200, 499
    print("---")
    print(f"Means over steps {lo}..{hi} (tier0 focus):")
    print(f"  L-solo b=200 tier0: {_mean_window(lsolo_b200_t0, lo, hi):.6f}")
    print(f"  L+S b=400 tier0:    {_mean_window(ls_b400_t0, lo, hi):.6f}")
    if lsolo_b400_t0 is not None:
        print(f"  L-solo b=400 tier0: {_mean_window(lsolo_b400_t0, lo, hi):.6f}")
    if global_ls.steps.size:
        print(f"  L+S b=400 global:   {_mean_window(global_ls, lo, hi):.6f}")


if __name__ == "__main__":
    main()

