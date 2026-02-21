#!/usr/bin/env python3
"""
Compare two aligned local-testnet runs (baseline vs experiment) and generate plots.

Intended for MatFormer + distillation experiments where:
- tier-0 loss is plain CE (from `client_loss ... loss=...`)
- tier-1 loss may be a *combined* objective (CE + KD), so we additionally parse:
  `psyche_modeling::causal_language_model: distillation stats step=... ce=... kd=...`

Outputs PNGs into `plots/rendered/`.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class Series:
    label: str
    steps: np.ndarray
    values: np.ndarray


def _read_lines(path: Path) -> Iterable[str]:
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore")
            yield ANSI_ESCAPE.sub("", line)


def _find_client_logs(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("client-*.txt"))


def _infer_tier(log_path: Path) -> Optional[int]:
    """
    Infer MatFormer tier for a per-client log.

    Prefer the explicit `loaded_model` marker when present, but some local-testnet
    runs may start capturing stdout/stderr after model load. In that case, fall
    back to the first tier observed in training/metrics logs.
    """
    candidate: Optional[int] = None
    for line in _read_lines(log_path):
        # Look for a stable marker written on model load.
        if "integration_test_log_marker=loaded_model" in line:
            m = re.search(r"\bmatformer_tier=(\d+)\b", line)
            if m:
                return int(m.group(1))

        # Fallback: capture tier from any "stable" per-client line.
        if candidate is None and (
            "Got training assignment" in line or "integration_test_log_marker=loss" in line
        ):
            m = re.search(r"\bmatformer_tier=(\d+)\b", line)
            if m:
                candidate = int(m.group(1))

    return candidate


def _pick_log_for_tier(run_dir: Path, tier: int) -> Path:
    logs = _find_client_logs(run_dir)
    if not logs:
        raise FileNotFoundError(f"no client logs found under {run_dir}")

    tier_to_path: Dict[int, Path] = {}
    for p in logs:
        t = _infer_tier(p)
        if t is None:
            continue
        tier_to_path[t] = p

    if tier not in tier_to_path:
        known = ", ".join(str(k) for k in sorted(tier_to_path.keys()))
        raise FileNotFoundError(f"no client log for tier={tier} under {run_dir} (found: {known})")

    return tier_to_path[tier]


def _extract_client_loss(log_path: Path, tier: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for line in _read_lines(log_path):
        if "client_loss" not in line or "integration_test_log_marker=loss" not in line:
            continue
        tier_m = re.search(r"\bmatformer_tier=(\d+)\b", line)
        if not tier_m or int(tier_m.group(1)) != tier:
            continue
        step_m = re.search(r"\bstep=(\d+)\b", line)
        loss_m = re.search(r"\bloss=([0-9.]+)\b", line)
        if not step_m or not loss_m:
            continue
        step = int(step_m.group(1))
        if step in out:
            continue
        out[step] = float(loss_m.group(1))
    return out


@dataclass(frozen=True)
class DistillStats:
    ce: float
    kd: float
    beta: float
    q_topk_mass: float


def _extract_distill_stats(log_path: Path) -> Dict[int, DistillStats]:
    out: Dict[int, DistillStats] = {}
    for line in _read_lines(log_path):
        if "distillation stats" not in line:
            continue
        step_m = re.search(r"\bstep=(\d+)\b", line)
        ce_m = re.search(r"\bce=([0-9.eE+-]+)\b", line)
        kd_m = re.search(r"\bkd=([0-9.eE+-]+)\b", line)
        beta_m = re.search(r"\bbeta=([0-9.eE+-]+)\b", line)
        q_m = re.search(r"\bq_topk_mass=([0-9.eE+-]+)\b", line)
        if not (step_m and ce_m and kd_m and beta_m and q_m):
            continue
        step = int(step_m.group(1))
        if step in out:
            continue
        out[step] = DistillStats(
            ce=float(ce_m.group(1)),
            kd=float(kd_m.group(1)),
            beta=float(beta_m.group(1)),
            q_topk_mass=float(q_m.group(1)),
        )
    return out


def _extract_cosine_mixer_mean(log_path: Path) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for line in _read_lines(log_path):
        if "DisTrO cosine mixer stats (MatFormer prefix params)" not in line:
            continue
        step_m = re.search(r"\bstep=(\d+)\b", line)
        mean_m = re.search(r"\bmean_cos=([0-9.eE+-]+)\b", line)
        if not (step_m and mean_m):
            continue
        step = int(step_m.group(1))
        if step in out:
            continue
        out[step] = float(mean_m.group(1))
    return out


def _to_series(label: str, by_step: Dict[int, float]) -> Series:
    steps = np.array(sorted(by_step.keys()), dtype=np.int64)
    values = np.array([by_step[int(s)] for s in steps], dtype=np.float64)
    return Series(label=label, steps=steps, values=values)


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


def _plot_two_panel(
    series: List[Series],
    out_path: Path,
    title: str,
    ylabel: str,
    smooth_window: int,
) -> None:
    fig, (ax_raw, ax_smooth) = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    for s in series:
        ax_raw.plot(s.steps, s.values, linewidth=1.3, label=s.label)
        ax_smooth.plot(
            s.steps,
            _moving_average(s.values, smooth_window),
            linewidth=2.0,
            label=s.label,
        )

    ax_raw.set_title("Per-Step (Raw)")
    ax_raw.set_xlabel("Step")
    ax_raw.set_ylabel(ylabel)
    ax_raw.grid(True, alpha=0.25)

    ax_smooth.set_title(f"Per-Step (Smoothed, window={smooth_window})")
    ax_smooth.set_xlabel("Step")
    ax_smooth.set_ylabel(ylabel)
    ax_smooth.grid(True, alpha=0.25)
    ax_smooth.legend(loc="best", frameon=True)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True, help="baseline run dir (logs/<ts>/)")
    ap.add_argument("--experiment", type=Path, required=True, help="experiment run dir (logs/<ts>/)")
    ap.add_argument("--out-dir", type=Path, default=Path("plots/rendered"))
    ap.add_argument("--tag", default="compare", help="tag used in output filenames")
    ap.add_argument("--smooth-window", type=int, default=20)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    base_dir = (repo / args.baseline).resolve()
    exp_dir = (repo / args.experiment).resolve()
    out_dir = (repo / args.out_dir).resolve()

    # Tier-0: CE is directly `client_loss`.
    base_t0 = _pick_log_for_tier(base_dir, tier=0)
    exp_t0 = _pick_log_for_tier(exp_dir, tier=0)
    t0_base = _to_series("baseline (tier0 CE)", _extract_client_loss(base_t0, tier=0))
    t0_exp = _to_series("experiment (tier0 CE)", _extract_client_loss(exp_t0, tier=0))
    _plot_two_panel(
        [t0_base, t0_exp],
        out_dir / f"{args.tag}_tier0_ce.png",
        title="Tier-0 CE (Aligned Runs)",
        ylabel="CE loss",
        smooth_window=args.smooth_window,
    )

    # Tier-1: baseline CE is `client_loss`. Experiment CE comes from distillation stats.
    base_t1 = _pick_log_for_tier(base_dir, tier=1)
    exp_t1 = _pick_log_for_tier(exp_dir, tier=1)
    t1_base = _to_series("baseline (tier1 CE)", _extract_client_loss(base_t1, tier=1))

    # For warm-start distillation, CE is plain `client_loss` before distillation activates,
    # and comes from distillation stats once active (since `client_loss` becomes combined loss).
    exp_t1_ce_fallback = _extract_client_loss(exp_t1, tier=1)
    stats = _extract_distill_stats(exp_t1)
    exp_t1_ce = dict(exp_t1_ce_fallback)
    exp_t1_ce.update({step: s.ce for step, s in stats.items()})
    t1_ce_exp = _to_series("experiment (tier1 CE)", exp_t1_ce)
    _plot_two_panel(
        [t1_base, t1_ce_exp],
        out_dir / f"{args.tag}_tier1_ce.png",
        title="Tier-1 CE (Aligned Runs; Experiment CE from distillation stats)",
        ylabel="CE loss",
        smooth_window=args.smooth_window,
    )

    # DisTrO cosine mixer mean cosine (tier-0 apply side). Useful for gradient-alignment tuning.
    t0_cos_base = _to_series(
        "baseline (tier0 mean_cos)",
        _extract_cosine_mixer_mean(base_t0),
    )
    t0_cos_exp = _to_series(
        "experiment (tier0 mean_cos)",
        _extract_cosine_mixer_mean(exp_t0),
    )
    if t0_cos_base.steps.size or t0_cos_exp.steps.size:
        _plot_two_panel(
            [t0_cos_base, t0_cos_exp],
            out_dir / f"{args.tag}_tier0_mean_cos.png",
            title="DisTrO Cosine Mixer Mean Cosine (Tier-0; MatFormer prefix params)",
            ylabel="mean cosine",
            smooth_window=args.smooth_window,
        )

    # Extra: KD magnitude + q_topk_mass (experiment only)
    if stats:
        steps = np.array(sorted(stats.keys()), dtype=np.int64)
        kd = np.array([stats[int(s)].kd for s in steps], dtype=np.float64)
        q = np.array([stats[int(s)].q_topk_mass for s in steps], dtype=np.float64)

        fig, (ax_kd, ax_q) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=160)
        ax_kd.plot(steps, kd, linewidth=1.6)
        ax_kd.set_title("KD (per-step)")
        ax_kd.set_xlabel("Step")
        ax_kd.set_ylabel("KD loss")
        ax_kd.grid(True, alpha=0.25)

        ax_q.plot(steps, q, linewidth=1.6)
        ax_q.set_title("Teacher top-k mass (q_topk_mass)")
        ax_q.set_xlabel("Step")
        ax_q.set_ylabel("Probability mass")
        ax_q.grid(True, alpha=0.25)

        fig.suptitle("Distillation Diagnostics (Experiment)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.tag}_distill_diagnostics.png")
        plt.close(fig)

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
