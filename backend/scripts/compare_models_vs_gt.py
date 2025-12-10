"""
Compare model-predicted skeleton metrics vs Fit3D ground truth.

This script joins:
  - offline_eval_fit3d_gt.csv  (per-video, per-backend model metrics)
  - fit3d_subset/fit3d_gt_metrics.csv (per subject/view/exercise GT metrics)

and computes, for each row:
  - differences between model mean/std and GT mean/std for:
      * elbow angle
      * knee angle
      * left_elbow angle
      * left_knee angle
      * shl_dist
      * rshl_rpalm
      * rshl_rhip
      * rpalm_rhip
      * rknee_rhip
      * rknee_rfeet
      * rhip_rfeet

Then it aggregates by (exercise, backend) and by (exercise, backend, view)
to give you a compact summary of average absolute errors.

Usage example (from repo root):

  python -m backend.scripts.compare_models_vs_gt \\
    --model-csv offline_eval_fit3d_gt.csv \\
    --gt-csv fit3d_subset/fit3d_gt_metrics.csv

You can filter by exercise:

  python -m backend.scripts.compare_models_vs_gt \\
    --model-csv offline_eval_fit3d_gt.csv \\
    --gt-csv fit3d_subset/fit3d_gt_metrics.csv \\
    --exercise bicep_curl
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METRIC_BASES = [
    "elbow",
    "left_elbow",
    "knee",
    "left_knee",
    "shl",
    "rshl_rpalm",
    "rshl_rhip",
    "rpalm_rhip",
    "rknee_rhip",
    "rknee_rfeet",
    "rhip_rfeet",
]


def _to_float(x: str) -> Optional[float]:
    x = x.strip()
    if not x:
        return None
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model metrics vs Fit3D GT.")
    parser.add_argument(
        "--model-csv",
        type=Path,
        required=True,
        help="Path to offline_eval_fit3d_gt.csv (model outputs).",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        required=True,
        help="Path to fit3d_gt_metrics.csv (GT metrics).",
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default=None,
        help="Optional exercise filter, e.g. bicep_curl / squat / push_up.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write per-(exercise,backend) summary as CSV.",
    )
    parser.add_argument(
        "--output-view-csv",
        type=Path,
        default=None,
        help="Optional path to write per-(exercise,backend,view) summary as CSV.",
    )
    args = parser.parse_args()

    # Load GT metrics keyed by (exercise, subject, view)
    gt_map: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    with args.gt_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exercise = (r.get("exercise") or "").strip()
            subject = (r.get("subject") or "").strip()
            view = (r.get("view") or "").strip()
            if not exercise or not subject or not view:
                continue
            key = (exercise, subject, view)
            gt_map[key] = r  # keep raw row; we'll parse fields later

    # Load model rows and compute per-row errors
    joined_rows: List[Dict[str, object]] = []
    with args.model_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exercise = (r.get("exercise") or "").strip()
            if args.exercise and exercise != args.exercise:
                continue
            subject = (r.get("subject") or "").strip()
            view = (r.get("view") or "").strip()
            backend = (r.get("backend") or "").strip()
            key = (exercise, subject, view)
            gt = gt_map.get(key)
            if gt is None:
                # No GT for this combo; skip.
                continue

            out: Dict[str, object] = dict(r)
            out["exercise"] = exercise
            out["subject"] = subject
            out["view"] = view
            out["backend"] = backend

            # For each base metric, compute |mean_model - mean_gt| and |std_model - std_gt|
            for base in METRIC_BASES:
                model_mean = _to_float(r.get(f"{base}_mean", "") or "")
                model_std = _to_float(r.get(f"{base}_std", "") or "")
                gt_mean = _to_float(gt.get(f"{base}_gt_mean", "") or "")
                gt_std = _to_float(gt.get(f"{base}_gt_std", "") or "")

                if model_mean is not None and gt_mean is not None:
                    out[f"{base}_mean_abs_err"] = abs(model_mean - gt_mean)
                else:
                    out[f"{base}_mean_abs_err"] = None

                if model_std is not None and gt_std is not None:
                    out[f"{base}_std_abs_err"] = abs(model_std - gt_std)
                else:
                    out[f"{base}_std_abs_err"] = None

            joined_rows.append(out)

    if not joined_rows:
        print("No joined rows (check exercise filter and GT mapping).")
        return

    # ---- Aggregate by (exercise, backend) ----
    group1: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in joined_rows:
        exercise = str(r["exercise"])
        backend = str(r["backend"])
        group1[(exercise, backend)].append(r)

    print("=== Mean absolute errors vs GT (per exercise + backend) ===")
    headers = ["exercise", "backend", "n"]
    for base in METRIC_BASES:
        headers.append(f"{base}_mean_err")
    for base in METRIC_BASES:
        headers.append(f"{base}_std_err")
    print("\t".join(headers))

    summary_rows: List[Dict[str, object]] = []

    for (exercise, backend), rs in sorted(group1.items()):
        def avg_field(field: str) -> float:
            vals = [
                float(v)
                for v in (r.get(field) for r in rs)
                if v is not None and str(v).strip() != ""
            ]
            return sum(vals) / len(vals) if vals else float("nan")

        row_vals: List[str] = [exercise, backend, str(len(rs))]
        row_dict: Dict[str, object] = {
            "exercise": exercise,
            "backend": backend,
            "n": len(rs),
        }
        for base in METRIC_BASES:
            val = avg_field(f"{base}_mean_abs_err")
            row_vals.append(f"{val:.4f}")
            row_dict[f"{base}_mean_err"] = val
        for base in METRIC_BASES:
            val = avg_field(f"{base}_std_abs_err")
            row_vals.append(f"{val:.4f}")
            row_dict[f"{base}_std_err"] = val
        print("\t".join(row_vals))
        summary_rows.append(row_dict)

    # 如果指定了输出 CSV，就把 per-(exercise,backend) 汇总写进去
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\n[INFO] per-(exercise,backend) summary written to {args.output_csv}")

    # ---- Aggregate by (exercise, backend, view) ----
    group2: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in joined_rows:
        exercise = str(r["exercise"])
        backend = str(r["backend"])
        view = str(r["view"])
        group2[(exercise, backend, view)].append(r)

    print("\n=== Mean absolute errors vs GT (per exercise + backend + view) ===")
    headers_view = ["exercise", "backend", "view", "n"]
    for base in METRIC_BASES:
        headers_view.append(f"{base}_mean_err")
    for base in METRIC_BASES:
        headers_view.append(f"{base}_std_err")
    print("\t".join(headers_view))

    view_rows: List[Dict[str, object]] = []

    for (exercise, backend, view), rs in sorted(group2.items()):
        def avg_field(field: str) -> float:
            vals = [
                float(v)
                for v in (r.get(field) for r in rs)
                if v is not None and str(v).strip() != ""
            ]
            return sum(vals) / len(vals) if vals else float("nan")

        row_vals: List[str] = [exercise, backend, view, str(len(rs))]
        row_dict: Dict[str, object] = {
            "exercise": exercise,
            "backend": backend,
            "view": view,
            "n": len(rs),
        }
        for base in METRIC_BASES:
            val = avg_field(f"{base}_mean_abs_err")
            row_vals.append(f"{val:.4f}")
            row_dict[f"{base}_mean_err"] = val
        for base in METRIC_BASES:
            val = avg_field(f"{base}_std_abs_err")
            row_vals.append(f"{val:.4f}")
            row_dict[f"{base}_std_err"] = val
        print("\t".join(row_vals))
        view_rows.append(row_dict)

    if args.output_view_csv is not None:
        args.output_view_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_view_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers_view)
            writer.writeheader()
            for row in view_rows:
                writer.writerow(row)
        print(f"\n[INFO] per-(exercise,backend,view) summary written to {args.output_view_csv}")


if __name__ == "__main__":
    main()
