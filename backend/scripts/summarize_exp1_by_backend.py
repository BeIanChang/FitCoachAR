"""
Second-level summary: aggregate Exp#1 metrics over views.

输入：fit3d_results/exp1_model/summary_simple.csv
  每行: exercise, subject, view, backend,
        left/right upper/lower mean/std, better mean/std, dist_mean/std, fps

输出：fit3d_results/exp1_model/summary_by_backend.csv
  每行: backend, exercise,
        left_upper_mean/std, right_upper_mean/std, upper_better_mean/std,
        left_lower_mean/std, right_lower_mean/std, lower_better_mean/std,
        dist_mean, dist_std, fps_mean, fps_std, n_views
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
        description="Aggregate summary_simple.csv over views."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fit3d_results/exp1_model/summary_simple.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fit3d_results/exp1_model/summary_by_backend.csv"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[WARN] input not found: {args.input}")
        return

    with args.input.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in rows:
        backend = (r.get("backend") or "").strip()
        exercise = (r.get("exercise") or "").strip()
        key = (backend, exercise)
        groups.setdefault(key, []).append(r)

    out_rows: List[Dict[str, object]] = []
    for (backend, exercise), rs in sorted(groups.items()):
        # per-view metrics
        upper_means: List[float] = []
        upper_stds: List[float] = []
        upper_better_means: List[float] = []
        upper_better_stds: List[float] = []
        lower_means: List[float] = []
        lower_stds: List[float] = []
        dist_means: List[float] = []
        dist_stds_list: List[float] = []
        fps_vals: List[float] = []
        lower_better_means: List[float] = []
        lower_better_stds: List[float] = []

        for r in rs:
            um = to_float(r.get("upper_mean"))
            us = to_float(r.get("upper_std"))
            ub_m = to_float(r.get("upper_better_mean"))
            ub_s = to_float(r.get("upper_better_std"))
            lm = to_float(r.get("lower_mean"))
            ls = to_float(r.get("lower_std"))
            lb_m = to_float(r.get("lower_better_mean"))
            lb_s = to_float(r.get("lower_better_std"))
            dm = to_float(r.get("dist_mean"))
            ds = to_float(r.get("dist_std"))
            fp = to_float(r.get("fps"))

            if um is not None:
                upper_means.append(um)
            if us is not None:
                upper_stds.append(us)

            if ub_m is not None:
                upper_better_means.append(ub_m)
            if ub_s is not None:
                upper_better_stds.append(ub_s)
            if lm is not None:
                lower_means.append(lm)
            if ls is not None:
                lower_stds.append(ls)
            if lb_m is not None:
                lower_better_means.append(lb_m)
            if lb_s is not None:
                lower_better_stds.append(lb_s)

            if dm is not None:
                dist_means.append(dm)
            if ds is not None:
                dist_stds_list.append(ds)
            if fp is not None:
                fps_vals.append(fp)

        def avg(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else float("nan")

        upper_mean = avg(upper_means)
        upper_std = avg(upper_stds)
        lower_mean = avg(lower_means)
        lower_std = avg(lower_stds)
        upper_better_mean = avg(upper_better_means)
        upper_better_std = avg(upper_better_stds)
        lower_better_mean = avg(lower_better_means)
        lower_better_std = avg(lower_better_stds)
        dist_mean = avg(dist_means)
        dist_std = avg(dist_stds_list)

        fps_mean, fps_std = mean_std(fps_vals)

        out_rows.append(
            {
                "backend": backend,
                "exercise": exercise,
                "n_views": len(rs),
                "upper_mean": round(upper_mean, 2),
                "upper_std": round(upper_std, 2),
                "upper_better_mean": round(upper_better_mean, 2),
                "upper_better_std": round(upper_better_std, 2),
                "lower_mean": round(lower_mean, 2),
                "lower_std": round(lower_std, 2),
                "lower_better_mean": round(lower_better_mean, 2),
                "lower_better_std": round(lower_better_std, 2),
                "dist_mean": round(dist_mean, 4),
                "dist_std": round(dist_std, 4),
                "fps_mean": round(fps_mean, 2) if fps_vals else "",
                "fps_std": round(fps_std, 2) if fps_vals else "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "exercise",
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

    print(f"[INFO] wrote backend-level summary to {args.output}")


if __name__ == "__main__":
    main()
