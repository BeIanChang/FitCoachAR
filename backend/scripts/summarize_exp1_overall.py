"""
Aggregate the per-exercise summary into a single row per backend.

Input:
  fit3d_results/exp1_model/summary_by_backend.csv
    backend,exercise,n_views,upper_mean,upper_std,upper_better_mean,upper_better_std,
    lower_mean,lower_std,lower_better_mean,lower_better_std,dist_mean,dist_std,
    fps_mean,fps_std

Output:
  fit3d_results/exp1_model/summary_overall.csv
    backend,n_views,upper_mean,upper_std,upper_better_mean,upper_better_std,
    lower_mean,lower_std,lower_better_mean,lower_better_std,dist_mean,dist_std,
    fps_mean,fps_std
  Values are averaged over exercises (simple mean across available rows);
  n_views is summed over exercises.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def to_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    import math

    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(var)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate summary_by_backend.csv into one row per backend."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fit3d_results/exp1_model/summary_by_backend.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fit3d_results/exp1_model/summary_overall.csv"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[WARN] input not found: {args.input}")
        return

    with args.input.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_backend: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        backend = (r.get("backend") or "").strip()
        by_backend.setdefault(backend, []).append(r)

    out_rows: List[Dict[str, object]] = []
    num_fields = [
        "upper_mean",
        "upper_std",
        "upper_better_mean",
        "upper_better_std",
        "lower_mean",
        "lower_std",
        "lower_better_mean",
        "lower_better_std",
        "dist_mean",
        "dist_std",
    ]

    for backend, rs in sorted(by_backend.items()):
        agg: Dict[str, List[float]] = {k: [] for k in num_fields}
        fps_vals: List[float] = []
        n_views_total = 0

        for r in rs:
            n_views_total += int(float(r.get("n_views") or 0))
            for k in num_fields:
                v = to_float(r.get(k))
                if v is not None:
                    agg[k].append(v)
            fp = to_float(r.get("fps_mean"))
            if fp is not None:
                fps_vals.append(fp)

        out = {"backend": backend, "n_views": n_views_total}
        for k, vals in agg.items():
            out[k] = round(mean(vals), 2)

        fps_mean, fps_std = mean_std(fps_vals)
        out["fps_mean"] = round(fps_mean, 2) if fps_vals else ""
        out["fps_std"] = round(fps_std, 2) if fps_vals else ""

        out_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "n_views",
        "upper_mean",
        "upper_std",
        "upper_better_mean",
        "upper_better_std",
        "lower_mean",
        "lower_std",
        "lower_better_mean",
        "lower_better_std",
        "dist_mean",
        "dist_std",
        "fps_mean",
        "fps_std",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[INFO] wrote overall summary to {args.output}")


if __name__ == "__main__":
    main()
