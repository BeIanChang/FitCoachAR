"""
Quick analysis helper for `offline_eval_fit3d_gt.csv`.

Usage (from repo root):

  python -m backend.scripts.analyze_fit3d_gt \
    --csv offline_eval_fit3d_gt.csv

You can filter by exercise, for example:

  python -m backend.scripts.analyze_fit3d_gt \
    --csv offline_eval_fit3d_gt.csv \
    --exercise bicep_curl

The script prints:
  1) Per (exercise, backend) summary: n, mean gt/pred reps, mean abs error,
     mean latency and FPS.
  2) Per (exercise, backend, view) summary, to see camera/view sensitivity.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _to_float(x: str) -> Optional[float]:
    x = x.strip()
    if not x:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: str) -> Optional[int]:
    x = x.strip()
    if not x:
        return None
    try:
        return int(x)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze offline_eval_fit3d_gt.csv summaries.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to offline_eval_fit3d_gt.csv")
    parser.add_argument(
        "--exercise",
        type=str,
        default=None,
        help="Optional exercise filter, e.g. bicep_curl / squat / push_up.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    with args.csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if args.exercise and r.get("exercise") != args.exercise:
                continue
            rows.append(r)

    if not rows:
        print("No rows matched the filter.")
        return

    # ---- 1) Per (exercise, backend) summary ----
    group1: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (r.get("exercise", ""), r.get("backend", ""))
        group1[key].append(r)

    print("=== Per exercise + backend summary ===")
    print("exercise\tbackend\tn\tmean_gt\tmean_pred\tmean_abs_err\tlat_ms\tfps")
    for (exercise, backend), rs in sorted(group1.items()):
        gt_vals: List[int] = []
        pred_vals: List[int] = []
        abs_err_vals: List[float] = []
        lat_vals: List[float] = []
        fps_vals: List[float] = []
        for r in rs:
            gt = _to_int(r.get("gt_reps", "") or "")
            pred = _to_int(r.get("pred_reps", "") or "")
            abs_err = _to_float(r.get("abs_rep_error", "") or "")
            lat = _to_float(r.get("avg_latency_ms", "") or "")
            fps = _to_float(r.get("fps", "") or "")
            if gt is not None:
                gt_vals.append(gt)
            if pred is not None:
                pred_vals.append(pred)
            if abs_err is not None:
                abs_err_vals.append(abs_err)
            if lat is not None:
                lat_vals.append(lat)
            if fps is not None:
                fps_vals.append(fps)

        def avg(vs: List[float]) -> float:
            return sum(vs) / len(vs) if vs else float("nan")

        print(
            f"{exercise}\t{backend}\t{len(rs)}\t"
            f"{avg([float(g) for g in gt_vals]):.2f}\t"
            f"{avg([float(p) for p in pred_vals]):.2f}\t"
            f"{avg(abs_err_vals):.2f}\t"
            f"{avg(lat_vals):.1f}\t"
            f"{avg(fps_vals):.1f}"
        )

    # ---- 2) Per (exercise, backend, view) summary ----
    group2: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (r.get("exercise", ""), r.get("backend", ""), r.get("view", ""))
        group2[key].append(r)

    print("\n=== Per exercise + backend + view summary ===")
    print("exercise\tbackend\tview\tn\tmean_gt\tmean_pred\tmean_abs_err\telbow_std\tknee_std")
    for (exercise, backend, view), rs in sorted(group2.items()):
        gt_vals: List[int] = []
        pred_vals: List[int] = []
        abs_err_vals: List[float] = []
        elbow_std_vals: List[float] = []
        knee_std_vals: List[float] = []
        for r in rs:
            gt = _to_int(r.get("gt_reps", "") or "")
            pred = _to_int(r.get("pred_reps", "") or "")
            abs_err = _to_float(r.get("abs_rep_error", "") or "")
            e_std = _to_float(r.get("elbow_std", "") or "")
            k_std = _to_float(r.get("knee_std", "") or "")
            if gt is not None:
                gt_vals.append(gt)
            if pred is not None:
                pred_vals.append(pred)
            if abs_err is not None:
                abs_err_vals.append(abs_err)
            if e_std is not None:
                elbow_std_vals.append(e_std)
            if k_std is not None:
                knee_std_vals.append(k_std)

        def avg(vs: List[float]) -> float:
            return sum(vs) / len(vs) if vs else float("nan")

        print(
            f"{exercise}\t{backend}\t{view}\t{len(rs)}\t"
            f"{avg([float(g) for g in gt_vals]):.2f}\t"
            f"{avg([float(p) for p in pred_vals]):.2f}\t"
            f"{avg(abs_err_vals):.2f}\t"
            f"{avg(elbow_std_vals):.1f}\t"
            f"{avg(knee_std_vals):.1f}"
        )


if __name__ == "__main__":
    main()

