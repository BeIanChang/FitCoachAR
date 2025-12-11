"""
Summarize framewise HRNet2D metrics into per-video means.

输入：
  framewise_results/hrnet2d/<exercise>__<subject>__<view>.json

JSON 结构（由 framewise_hrnet2d_from_kpts.py 生成）：
  {
    "subject": "s03",
    "exercise": "squat",
    "view": "50591643",
    "backend": "hrnet2d",
    "gt_reps": 5,
    "start_frame": 405,
    "end_frame": 1090,
    "frames": [
      {
        "frame_index": 405,
        "elbow_R": ...,
        "elbow_L": ...,
        "knee_R": ...,
        ...
      },
      ...
    ]
  }

本脚本对每个 JSON：
  - 从 frames 列表中逐帧读取各个 metric；
  - 丢弃 None / NaN；
  - 对剩余值取平均；
  - 输出一行 CSV：

    exercise,subject,view,backend,frames,
    elbow_R_mean,elbow_L_mean,knee_R_mean,knee_L_mean,
    hip_R_mean,hip_L_mean,shoulder_R_mean,shoulder_L_mean,
    shl_mean,rshl_rpalm_mean,rshl_rhip_mean,rpalm_rhip_mean,
    rknee_rhip_mean,rknee_rfeet_mean,rhip_rfeet_mean

方便和 GT 或其它模型的 clip-level 统计对比。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


METRICS = [
    "elbow_R",
    "elbow_L",
    "knee_R",
    "knee_L",
    "hip_R",
    "hip_L",
    "shoulder_R",
    "shoulder_L",
    "shl",
    "rshl_rpalm",
    "rshl_rhip",
    "rpalm_rhip",
    "rknee_rhip",
    "rknee_rfeet",
    "rhip_rfeet",
]


def mean_ignore_nan(vals: List[Any]) -> float:
    arr = np.asarray(
        [v for v in vals if v is not None and np.isfinite(float(v))],
        dtype=float,
    )
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize framewise HRNet2D metrics into per-video means."
    )
    parser.add_argument(
        "--framewise-dir",
        type=Path,
        default=Path("framewise_results/hrnet2d"),
        help="Directory containing hrnet2d framewise JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fit3d_results/framewise_hrnet2d_summary.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []

    for path in sorted(args.framewise_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)

        exercise = (data.get("exercise") or "").strip()
        subject = (data.get("subject") or "").strip()
        view = (data.get("view") or "").strip()
        backend = (data.get("backend") or "hrnet2d").strip()

        frames = data.get("frames", [])
        n_frames = len(frames)

        row: Dict[str, Any] = {
            "exercise": exercise,
            "subject": subject,
            "view": view,
            "backend": backend,
            "frames": n_frames,
        }

        for m in METRICS:
            vals = [fr.get(m) for fr in frames]
            row[f"{m}_mean"] = mean_ignore_nan(vals)

        rows.append(row)

    if not rows:
        print(f"[WARN] no framewise hrnet2d JSON found in {args.framewise_dir}")
        return

    # Collect union of fieldnames
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

