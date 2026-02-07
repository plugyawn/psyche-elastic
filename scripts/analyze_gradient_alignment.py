#!/usr/bin/env python3
"""
Analyze gradient alignment logs from heterogeneous training experiments.

Parses log files to extract sign_agreement and cosine_similarity metrics,
grouped by param_type (mlp vs other).

Usage:
    python scripts/analyze_gradient_alignment.py logs/gradient-alignment-*/exp1-*.log
    python scripts/analyze_gradient_alignment.py logs/gradient-alignment-*/*.log
"""

import re
import sys
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class AlignmentMetrics:
    sign_agreement: float
    cosine_similarity: float
    param_type: str
    param_name: str
    param_index: int
    num_peers: int


def parse_log_line(line: str) -> AlignmentMetrics | None:
    """Parse a gradient_alignment log line into metrics."""
    if "gradient_alignment" not in line:
        return None

    # Match tracing format: field=value pairs
    # Example: sign_agreement=0.55 cosine_similarity=0.10 param_type="mlp" param_name="layers.0.mlp.c_fc.weight" param_index=5 num_peers=4
    patterns = {
        "sign_agreement": r"sign_agreement=([\d.]+)",
        "cosine_similarity": r"cosine_similarity=([\d.-]+)",
        "param_type": r'param_type="?(\w+)"?',
        "param_name": r'param_name="?([^"\s]+)"?',
        "param_index": r"param_index=(\d+)",
        "num_peers": r"num_peers=(\d+)",
    }

    values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            values[key] = match.group(1)

    if "sign_agreement" not in values:
        return None

    return AlignmentMetrics(
        sign_agreement=float(values.get("sign_agreement", 0)),
        cosine_similarity=float(values.get("cosine_similarity", 0)),
        param_type=values.get("param_type", "unknown"),
        param_name=values.get("param_name", "unknown"),
        param_index=int(values.get("param_index", 0)),
        num_peers=int(values.get("num_peers", 0)),
    )


def analyze_log_file(path: Path) -> dict:
    """Analyze a single log file and return summary statistics."""
    metrics_by_type = defaultdict(list)
    all_metrics = []

    with open(path, "r") as f:
        for line in f:
            m = parse_log_line(line)
            if m:
                metrics_by_type[m.param_type].append(m)
                all_metrics.append(m)

    if not all_metrics:
        return {"error": "No gradient_alignment entries found"}

    def summarize(metrics: list[AlignmentMetrics]) -> dict:
        if not metrics:
            return {}
        n = len(metrics)
        sign_avg = sum(m.sign_agreement for m in metrics) / n
        cos_avg = sum(m.cosine_similarity for m in metrics) / n
        sign_min = min(m.sign_agreement for m in metrics)
        sign_max = max(m.sign_agreement for m in metrics)
        cos_min = min(m.cosine_similarity for m in metrics)
        cos_max = max(m.cosine_similarity for m in metrics)
        return {
            "count": n,
            "sign_agreement": {"mean": sign_avg, "min": sign_min, "max": sign_max},
            "cosine_similarity": {"mean": cos_avg, "min": cos_min, "max": cos_max},
        }

    result = {
        "file": str(path),
        "total_entries": len(all_metrics),
        "overall": summarize(all_metrics),
        "by_type": {ptype: summarize(metrics) for ptype, metrics in metrics_by_type.items()},
    }

    # Extract unique param names for each type
    for ptype in metrics_by_type:
        unique_params = set(m.param_name for m in metrics_by_type[ptype])
        result["by_type"][ptype]["unique_params"] = len(unique_params)

    return result


def print_summary(results: list[dict]):
    """Print a formatted summary of all results."""
    print("\n" + "=" * 70)
    print("GRADIENT ALIGNMENT ANALYSIS SUMMARY")
    print("=" * 70)

    for r in results:
        if "error" in r:
            print(f"\n{r['file']}: {r['error']}")
            continue

        print(f"\n{'=' * 70}")
        print(f"File: {r['file']}")
        print(f"Total entries: {r['total_entries']}")

        if r.get("overall"):
            o = r["overall"]
            print(f"\nOverall:")
            print(f"  Sign agreement:     {o['sign_agreement']['mean']:.4f} (range: {o['sign_agreement']['min']:.4f} - {o['sign_agreement']['max']:.4f})")
            print(f"  Cosine similarity:  {o['cosine_similarity']['mean']:.4f} (range: {o['cosine_similarity']['min']:.4f} - {o['cosine_similarity']['max']:.4f})")

        if r.get("by_type"):
            print("\nBy parameter type:")
            for ptype, stats in r["by_type"].items():
                if stats:
                    print(f"\n  {ptype}:")
                    print(f"    Count: {stats['count']} ({stats.get('unique_params', '?')} unique params)")
                    print(f"    Sign agreement:    {stats['sign_agreement']['mean']:.4f}")
                    print(f"    Cosine similarity: {stats['cosine_similarity']['mean']:.4f}")


def compare_experiments(results: list[dict]):
    """Compare results across experiments."""
    if len(results) < 2:
        return

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)

    # Create comparison table
    headers = ["Experiment", "Sign Agr (All)", "Sign Agr (MLP)", "Cosine (All)", "Cosine (MLP)"]
    rows = []

    for r in results:
        if "error" in r:
            continue
        name = Path(r["file"]).stem
        overall = r.get("overall", {})
        mlp = r.get("by_type", {}).get("mlp", {})

        row = [
            name,
            f"{overall.get('sign_agreement', {}).get('mean', 0):.4f}",
            f"{mlp.get('sign_agreement', {}).get('mean', 0):.4f}" if mlp else "N/A",
            f"{overall.get('cosine_similarity', {}).get('mean', 0):.4f}",
            f"{mlp.get('cosine_similarity', {}).get('mean', 0):.4f}" if mlp else "N/A",
        ]
        rows.append(row)

    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"\n{header_line}")
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Expand glob patterns
    log_paths = []
    for pattern in sys.argv[1:]:
        if "*" in pattern:
            log_paths.extend(Path(".").glob(pattern))
        else:
            log_paths.append(Path(pattern))

    if not log_paths:
        print(f"No log files found matching: {sys.argv[1:]}")
        sys.exit(1)

    results = []
    for path in sorted(log_paths):
        if path.exists() and path.is_file():
            results.append(analyze_log_file(path))

    print_summary(results)
    compare_experiments(results)

    # Also output JSON for further processing
    json_output = Path("logs/gradient_alignment_analysis.json")
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_output}")


if __name__ == "__main__":
    main()
