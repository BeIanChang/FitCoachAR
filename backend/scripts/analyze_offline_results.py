"""
Simple analysis helper for offline_eval_results.csv.

Usage (from repo root):

  cd FitCoachAR
  python -m backend.scripts.analyze_offline_results \
    --csv offline_eval_results.csv

This script:
  - Groups rows by (exercise, pitch, yaw, backend)
  - Computes basic aggregates:
      * mean reps (curl/squat)
      * mean latency / fps
      * mean/std for angle and distance stability metrics
  - Prints a compact textual summary so you can quickly see
    how different backends behave under different views.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Aggregate:
    count: int = 0
    curl_sum: float = 0.0
    squat_sum: float = 0.0
    latency_sum: float = 0.0
    fps_sum: float = 0.0
    elbow_std_sum: float = 0.0
    knee_std_sum: float = 0.0
    rshl_rpalm_std_sum: float = 0.0
    rknee_rhip_std_sum: float = 0.0

    def add(self, row: Dict[str, str]) -> None:
        self.count += 1

        def f(name: str) -> Optional[float]:
            v = row.get(name)
            if v in (None, "", "None"):
                return None
            try:
                return float(v)
            except ValueError:
                return None

        for attr, col in (
            ("curl_sum", "curl_counter"),
            ("squat_sum", "squat_counter"),
            ("latency_sum", "avg_latency_ms"),
            ("fps_sum", "fps"),
            ("elbow_std_sum", "elbow_std"),
            ("knee_std_sum", "knee_std"),
            ("rshl_rpalm_std_sum", "rshl_rpalm_std"),
            ("rknee_rhip_std_sum", "rknee_rhip_std"),
        ):
            val = f(col)
            if val is not None:
                setattr(self, attr, getattr(self, attr) + val)

    def mean(self, attr: str) -> Optional[float]:
        if self.count == 0:
            return None
        return getattr(self, attr) / self.count


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze(csv_path: Path) -> None:
    rows = load_rows(csv_path)
    if not rows:
        print("No rows found in CSV.")
        return

    groups: Dict[tuple, Aggregate] = defaultdict(Aggregate)

    for row in rows:
        key = (
            row.get("exercise", ""),
            row.get("pitch", ""),
            row.get("yaw", ""),
            row.get("backend", ""),
        )
        groups[key].add(row)

    # Pretty print summary
    print("=== Offline evaluation summary (aggregated by exercise/pitch/yaw/backend) ===")
    header = (
        "exercise",
        "pitch",
        "yaw",
        "backend",
        "n",
        "curl_mean",
        "squat_mean",
        "lat_ms_mean",
        "fps_mean",
        "elbow_std_mean",
        "knee_std_mean",
        "rshl_rpalm_std_mean",
        "rknee_rhip_std_mean",
    )
    print("\t".join(header))

    for (exercise, pitch, yaw, backend), agg in sorted(groups.items()):
        row_out = [
            exercise or "-",
            pitch or "-",
            yaw or "-",
            backend or "-",
            str(agg.count),
            f"{agg.mean('curl_sum'):.2f}" if agg.count else "NA",
            f"{agg.mean('squat_sum'):.2f}" if agg.count else "NA",
            f"{agg.mean('latency_sum'):.1f}" if agg.count else "NA",
            f"{agg.mean('fps_sum'):.1f}" if agg.count else "NA",
            f"{agg.mean('elbow_std_sum'):.2f}" if agg.count else "NA",
            f"{agg.mean('knee_std_sum'):.2f}" if agg.count else "NA",
            f"{agg.mean('rshl_rpalm_std_sum'):.4f}" if agg.count else "NA",
            f"{agg.mean('rknee_rhip_std_sum'):.4f}" if agg.count else "NA",
        ]
        print("\t".join(row_out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze offline_eval_results.csv.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("offline_eval_results.csv"),
        help="Path to offline_eval_results.csv",
    )
    args = parser.parse_args()
    analyze(args.csv)


if __name__ == "__main__":
    main()

