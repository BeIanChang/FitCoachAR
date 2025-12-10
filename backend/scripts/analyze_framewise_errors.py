"""
Aggregate framewise GT vs. model errors into a compact CSV.

假设已经有以下 JSON 结构：
  - GT: framewise_results/gt/<exercise>__<subject>__<view>.json
  - 模型: framewise_results/<backend>/<exercise>__<subject>__<view>.json
    其中 backend ∈ {mediapipe_2d, mediapipe_3d, movenet_3d, motionbert3d, ...}

每个 JSON 的 frames 列表中包含:
  {
    "frame_index": int,
    "elbow_R": float | null,
    "elbow_L": float | null,
    "knee_R":  float | null,
    "knee_L":  float | null,
    "shl": float | null,
    "rshl_rpalm": float | null,
    "rknee_rhip": float | null,
    "rhip_rfeet": float | null,
  }

本脚本对每个 (exercise, subject, view, backend) 计算：
  - 每个 metric 的 mean |Δ|、std |Δ|、样本数 n
并写入 CSV 方便后续在报告中引用。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_series(path: Path) -> Dict[int, Dict[str, float]]:
    with path.open() as f:
        data = json.load(f)
    frames = data.get("frames", [])
    by_idx: Dict[int, Dict[str, float]] = {}
    for fr in frames:
        idx = int(fr.get("frame_index"))
        by_idx[idx] = fr
    return by_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize framewise GT vs model errors.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("framewise_results"),
        help="Root directory containing gt/ and backend subdirs.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="s03",
        help="Subject ID to analyze (e.g. s03).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["mediapipe_2d", "mediapipe_3d", "movenet_3d", "motionbert3d"],
        help="Backend names to include (must match directory names under root).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "elbow_R",
            "elbow_L",
            "knee_R",
            "knee_L",
            "shl",
            "rshl_rpalm",
            "rknee_rhip",
            "rhip_rfeet",
        ],
        help="Metric keys to compare.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("framewise_results/framewise_error_summary.csv"),
        help="Path to write summary CSV.",
    )
    args = parser.parse_args()

    gt_dir = args.root / "gt"
    if not gt_dir.exists():
        raise SystemExit(f"GT directory not found: {gt_dir}")

    rows: List[Dict[str, object]] = []

    # 遍历所有 GT 文件：<exercise>__<subject>__<view>.json
    for gt_path in sorted(gt_dir.glob(f"*__{args.subject}__*.json")):
        stem = gt_path.stem  # e.g. barbell_row__s03__50591643
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        exercise, subject, view = parts
        if subject != args.subject:
            continue

        gt_series = load_series(gt_path)
        frame_indices = sorted(gt_series.keys())
        if not frame_indices:
            continue

        for backend in args.backends:
            backend_dir = args.root / backend
            b_path = backend_dir / f"{exercise}__{subject}__{view}.json"
            if not b_path.exists():
                continue
            model_series = load_series(b_path)

            metric_errs: Dict[str, List[float]] = {m: [] for m in args.metrics}

            for idx in frame_indices:
                gt_row = gt_series.get(idx)
                model_row = model_series.get(idx)
                if not gt_row or not model_row:
                    continue
                for m in args.metrics:
                    gv = gt_row.get(m)
                    mv = model_row.get(m)
                    if gv is None or mv is None:
                        continue
                    if isinstance(gv, (float, int)) and isinstance(mv, (float, int)):
                        gv_f = float(gv)
                        mv_f = float(mv)
                        if np.isnan(gv_f) or np.isnan(mv_f):
                            continue
                        metric_errs[m].append(abs(mv_f - gv_f))

            # 如果所有 metric 都没有有效样本，跳过该 backend
            if not any(metric_errs[m] for m in args.metrics):
                continue

            row: Dict[str, object] = {
                "exercise": exercise,
                "subject": subject,
                "view": view,
                "backend": backend,
            }

            for m in args.metrics:
                arr = np.array(metric_errs[m], dtype=float)
                if arr.size == 0:
                    row[f"{m}_mean_abs_err"] = None
                    row[f"{m}_std_abs_err"] = None
                    row[f"{m}_n"] = 0
                else:
                    row[f"{m}_mean_abs_err"] = float(arr.mean())
                    row[f"{m}_std_abs_err"] = float(arr.std())
                    row[f"{m}_n"] = int(arr.size)

            rows.append(row)

    if not rows:
        raise SystemExit("No errors computed; check inputs and arguments.")

    # 写 CSV
    fieldnames = ["exercise", "subject", "view", "backend"]
    for m in args.metrics:
        fieldnames.extend(
            [f"{m}_mean_abs_err", f"{m}_std_abs_err", f"{m}_n"]
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Wrote summary to {args.output_csv}")


if __name__ == "__main__":
    main()

