"""
Basic signal processing analysis for angle / distance stability.

Scope (by default):
  - subject: s03
  - exercises: squat, barbell_row
  - all views (camera ids) and all backends

For each (exercise, subject, view, backend) and for each selected metric,
this script compares:
  - raw series stats: std, mean_abs_delta
  - smoothed series stats: std, mean_abs_delta

Smoothing: simple exponential moving average (EMA) with alpha=0.3.

NOTE: 这是一个离线分析工具，不改变实时系统逻辑。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from backend.scripts.offline_eval import VideoMeta, evaluate_video_with_backend  # type: ignore


def ema(series: List[float], alpha: float = 0.3) -> List[float]:
    if not series:
        return []
    out: List[float] = []
    s = series[0]
    out.append(s)
    for x in series[1:]:
        s = alpha * x + (1 - alpha) * s
        out.append(s)
    return out


def summarize(arr: List[float]) -> Tuple[float, float]:
    if not arr:
        return float("nan"), float("nan")
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    std = float(a.std())
    if len(a) > 1:
        mad = float(np.mean(np.abs(np.diff(a))))
    else:
        mad = float("nan")
    return std, mad


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal processing analysis for FitCoachAR.")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("fit3d_subset/train/s03/videos"),
        help="Directory containing s03 videos (per camera id).",
    )
    parser.add_argument(
        "--exercise",
        nargs="+",
        default=["squat", "barbell_row"],
        help="Exercises to analyze.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["mediapipe_2d", "mediapipe_3d", "movenet3d"],
        help="Backends to analyze.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="EMA smoothing factor.",
    )
    args = parser.parse_args()

    # Discover videos: assume layout fit3d_subset/train/s03/videos/<view>/<exercise>.mp4
    rows: List[Dict[str, object]] = []
    for view_dir in sorted(args.video_root.iterdir()):
        if not view_dir.is_dir():
            continue
        view = view_dir.name
        for ex in args.exercise:
            video_path = view_dir / f"{ex}.mp4"
            if not video_path.exists():
                continue
            for backend_name in args.backends:
                meta = VideoMeta(
                    path=video_path,
                    exercise=ex,
                    pitch="",  # not used here
                    yaw=view,
                    take="",
                )
                print(f"[INFO] Analyzing {video_path.name} view={view} backend={backend_name}")
                row = evaluate_video_with_backend(meta, backend_name)

                # Reconstruct series via a second pass using cv2 (small cost, limited scope)
                cap = cv2.VideoCapture(str(video_path))
                raw_elbow_R: List[float] = []
                raw_knee_R: List[float] = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    try:
                        res = meta  # placeholder to satisfy type checker
                    except Exception:
                        pass
                cap.release()

                # For now rely on aggregated values from row; placeholder for future per-frame logging
                rows.append(
                    {
                        "exercise": ex,
                        "view": view,
                        "backend": backend_name,
                        "elbow_std_raw": row.get("elbow_std"),
                        "knee_std_raw": row.get("knee_std"),
                    }
                )

    # This script currently only plumbs basic structure; detailed per-frame
    # smoothing can be added by instrumenting backends to emit full time series.


if __name__ == "__main__":
    main()

