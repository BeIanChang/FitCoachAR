"""
Offline evaluation script for comparing pose backends on recorded videos.

Usage example (from repo root):

  python -m backend.scripts.offline_eval \\
    --video-dir data/videos \\
    --backends mediapipe_2d mediapipe_3d movenet3d

File naming convention (recommended):

  <exercise>__pitch-<pitch>__yaw-<yaw>__take-<id>.mp4

Where:
  exercise:  bicep_curls | squats | pushups | lateral_raises | barbell_rows
  pitch:     eye (平视) | down45 (俯视45°) | up45 (仰视45°)
  yaw:       front | front45 | side | back45 | back
  id:        any short identifier, e.g. 01, 02, good, noisy

Examples:
  bicep_curls__pitch-eye__yaw-front__take-01.mp4
  squats__pitch-down45__yaw-side__take-02.mp4
  pushups__pitch-eye__yaw-side__take-01.mp4
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


# Allow running via `python -m backend.scripts.offline_eval` from repo root.
ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / 'backend'
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pose_backends  # type: ignore
from pose_backends import build_pose_backend  # type: ignore


@dataclass
class VideoMeta:
    path: Path
    exercise: str
    pitch: str
    yaw: str
    take: str


def parse_video_meta(path: Path, root: Path) -> Optional[VideoMeta]:
    """Parse metadata from either filename or directory layout.

    Supported patterns:
      1) <exercise>__pitch-<pitch>__yaw-<yaw>__take-<id>.<ext>
      2) <root>/<exercise>/<pitch>/<yaw>.<ext>
    """
    stem = path.stem
    parts = stem.split("__")
    # Pattern 1: encoded in filename
    if len(parts) == 4:
        exercise = parts[0]
        pitch_part = parts[1]
        yaw_part = parts[2]
        take_part = parts[3]

        if not pitch_part.startswith("pitch-") or not yaw_part.startswith("yaw-") or not take_part.startswith("take-"):
            return None

        pitch = pitch_part.replace("pitch-", "", 1)
        yaw = yaw_part.replace("yaw-", "", 1)
        take = take_part.replace("take-", "", 1)
        return VideoMeta(path=path, exercise=exercise, pitch=pitch, yaw=yaw, take=take)

    # Pattern 2: dataset/<exercise>/<pitch>/<yaw>.<ext>
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    rel_parts = rel.parts
    if len(rel_parts) < 3:
        return None
    exercise = rel_parts[-3]
    pitch = rel_parts[-2]
    yaw = stem
    take = ""
    return VideoMeta(path=path, exercise=exercise, pitch=pitch, yaw=yaw, take=take)


def find_videos(video_dir: Path) -> List[VideoMeta]:
    videos: List[VideoMeta] = []
    exts = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.mkv", "*.avi")
    for ext in exts:
        for path in video_dir.rglob(ext):
            meta = parse_video_meta(path, video_dir)
            if meta:
                videos.append(meta)
    return sorted(videos, key=lambda m: str(m.path))


def extract_rep_counts(result: Dict[str, object]) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract curl and squat counters (if present) from a backend result payload.
    Returns (curl_counter, squat_counter), where each may be None.
    """
    curl = None
    squat = None
    if "curl_counter" in result:
        try:
            curl = int(result["curl_counter"])  # type: ignore[arg-type]
        except Exception:
            curl = None
    if "squat_counter" in result:
        try:
            squat = int(result["squat_counter"])  # type: ignore[arg-type]
        except Exception:
            squat = None
    return curl, squat


def evaluate_video_with_backend(
    meta: VideoMeta,
    backend_name: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Dict[str, object]:
    """
    Run a single backend on a single video and compute basic metrics.
    """
    backend = build_pose_backend(backend_name)

    # Tell the backend which exercise we are evaluating.
    try:
        backend.handle_command({"command": "select_exercise", "exercise": meta.exercise})
    except Exception:
        # Some backends may not need this; ignore failures.
        pass

    cap = cv2.VideoCapture(str(meta.path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {meta.path}")

    frame_count = 0
    frame_index = 0  # 1-based index of frames read from the video
    total_proc_time = 0.0
    last_result: Dict[str, object] = {}
    elbow_R_series: List[float] = []
    elbow_L_series: List[float] = []
    knee_R_series: List[float] = []
    knee_L_series: List[float] = []
    shl_series: List[float] = []
    rshl_rpalm_series: List[float] = []
    rshl_rhip_series: List[float] = []
    rpalm_rhip_series: List[float] = []
    rknee_rhip_series: List[float] = []
    rknee_rfeet_series: List[float] = []
    rhip_rfeet_series: List[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1

        # Optional cropping by frame index: only process [start_frame, end_frame].
        if start_frame is not None and frame_index < start_frame:
            continue
        if end_frame is not None and frame_index > end_frame:
            break

        frame_count += 1

        t0 = time.perf_counter()
        try:
            result = backend.process_frame(frame)  # type: ignore[attr-defined]
        except Exception:
            result = None
        total_proc_time += time.perf_counter() - t0

        if result:
            last_result = result
            # Collect angle trajectories when available (right & left)
            try:
                if "right_elbow_angle" in result and result["right_elbow_angle"] is not None:  # type: ignore[index]
                    elbow_R_series.append(float(result["right_elbow_angle"]))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if "left_elbow_angle" in result and result["left_elbow_angle"] is not None:  # type: ignore[index]
                    elbow_L_series.append(float(result["left_elbow_angle"]))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if "right_knee_angle" in result and result["right_knee_angle"] is not None:  # type: ignore[index]
                    knee_R_series.append(float(result["right_knee_angle"]))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if "left_knee_angle" in result and result["left_knee_angle"] is not None:  # type: ignore[index]
                    knee_L_series.append(float(result["left_knee_angle"]))  # type: ignore[arg-type]
            except Exception:
                pass
            # Collect distance metrics when available
            try:
                if "metric_shl_dist" in result and result["metric_shl_dist"] is not None:  # type: ignore[index]
                    shl_series.append(float(result["metric_shl_dist"]))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if "metric_rshl_rpalm" in result and result["metric_rshl_rpalm"] is not None:  # type: ignore[index]
                    rshl_rpalm_series.append(float(result["metric_rshl_rpalm"]))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if "metric_rshl_rhip" in result and result["metric_rshl_rhip"] is not None:  # type: ignore[index]
                    rshl_rhip_series.append(float(result["metric_rshl_rhip"]))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if "metric_rpalm_rhip" in result and result["metric_rpalm_rhip"] is not None:  # type: ignore[index]
                    rpalm_rhip_series.append(float(result["metric_rpalm_rhip"]))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if "metric_rknee_rhip" in result and result["metric_rknee_rhip"] is not None:  # type: ignore[index]
                    rknee_rhip_series.append(float(result["metric_rknee_rhip"]))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if "metric_rknee_rfeet" in result and result["metric_rknee_rfeet"] is not None:  # type: ignore[index]
                    rknee_rfeet_series.append(float(result["metric_rknee_rfeet"]))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                if "metric_rhip_rfeet" in result and result["metric_rhip_rfeet"] is not None:  # type: ignore[index]
                    rhip_rfeet_series.append(float(result["metric_rhip_rfeet"]))  # type: ignore[arg-type]
            except Exception:
                pass

    cap.release()

    # Close resources if backend exposes a close() method.
    try:
        backend.close()  # type: ignore[attr-defined]
    except Exception:
        pass

    avg_latency_ms = (total_proc_time / frame_count * 1000.0) if frame_count > 0 else None
    fps = (frame_count / total_proc_time) if total_proc_time > 0 else None

    curl_counter, squat_counter = extract_rep_counts(last_result)

    def summarize(series: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if len(series) == 0:
            return None, None, None
        n = float(len(series))
        mean = sum(series) / n
        var = sum((x - mean) ** 2 for x in series) / n
        std = var ** 0.5
        if len(series) > 1:
            mean_abs_delta = sum(abs(series[i] - series[i - 1]) for i in range(1, len(series))) / (len(series) - 1)
        else:
            mean_abs_delta = None
        return mean, std, mean_abs_delta

    elbow_mean, elbow_std, elbow_mean_abs_delta = summarize(elbow_R_series)
    left_elbow_mean, left_elbow_std, left_elbow_mean_abs_delta = summarize(elbow_L_series)
    knee_mean, knee_std, knee_mean_abs_delta = summarize(knee_R_series)
    left_knee_mean, left_knee_std, left_knee_mean_abs_delta = summarize(knee_L_series)
    shl_mean, shl_std, shl_mean_abs_delta = summarize(shl_series)
    rshl_rpalm_mean, rshl_rpalm_std, rshl_rpalm_mean_abs_delta = summarize(rshl_rpalm_series)
    rshl_rhip_mean, rshl_rhip_std, rshl_rhip_mean_abs_delta = summarize(rshl_rhip_series)
    rpalm_rhip_mean, rpalm_rhip_std, rpalm_rhip_mean_abs_delta = summarize(rpalm_rhip_series)
    rknee_rhip_mean, rknee_rhip_std, rknee_rhip_mean_abs_delta = summarize(rknee_rhip_series)
    rknee_rfeet_mean, rknee_rfeet_std, rknee_rfeet_mean_abs_delta = summarize(rknee_rfeet_series)
    rhip_rfeet_mean, rhip_rfeet_std, rhip_rfeet_mean_abs_delta = summarize(rhip_rfeet_series)

    return {
        "video": str(meta.path),
        "exercise": meta.exercise,
        "pitch": meta.pitch,
        "yaw": meta.yaw,
        "take": meta.take,
        "backend": backend_name,
        "frames": frame_count,
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
        "curl_counter": curl_counter,
        "squat_counter": squat_counter,
        "elbow_mean": elbow_mean,
        "elbow_std": elbow_std,
        "elbow_mean_abs_delta": elbow_mean_abs_delta,
        "left_elbow_mean": left_elbow_mean,
        "left_elbow_std": left_elbow_std,
        "left_elbow_mean_abs_delta": left_elbow_mean_abs_delta,
        "knee_mean": knee_mean,
        "knee_std": knee_std,
        "knee_mean_abs_delta": knee_mean_abs_delta,
        "left_knee_mean": left_knee_mean,
        "left_knee_std": left_knee_std,
        "left_knee_mean_abs_delta": left_knee_mean_abs_delta,
        "shl_mean": shl_mean,
        "shl_std": shl_std,
        "shl_mean_abs_delta": shl_mean_abs_delta,
        "rshl_rpalm_mean": rshl_rpalm_mean,
        "rshl_rpalm_std": rshl_rpalm_std,
        "rshl_rpalm_mean_abs_delta": rshl_rpalm_mean_abs_delta,
        "rshl_rhip_mean": rshl_rhip_mean,
        "rshl_rhip_std": rshl_rhip_std,
        "rshl_rhip_mean_abs_delta": rshl_rhip_mean_abs_delta,
        "rpalm_rhip_mean": rpalm_rhip_mean,
        "rpalm_rhip_std": rpalm_rhip_std,
        "rpalm_rhip_mean_abs_delta": rpalm_rhip_mean_abs_delta,
        "rknee_rhip_mean": rknee_rhip_mean,
        "rknee_rhip_std": rknee_rhip_std,
        "rknee_rhip_mean_abs_delta": rknee_rhip_mean_abs_delta,
        "rknee_rfeet_mean": rknee_rfeet_mean,
        "rknee_rfeet_std": rknee_rfeet_std,
        "rknee_rfeet_mean_abs_delta": rknee_rfeet_mean_abs_delta,
        "rhip_rfeet_mean": rhip_rfeet_mean,
        "rhip_rfeet_std": rhip_rfeet_std,
        "rhip_rfeet_mean_abs_delta": rhip_rfeet_mean_abs_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for FitCoachAR pose backends.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory containing recorded videos.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["mediapipe_2d", "mediapipe_3d", "movenet3d"],
        help="List of backend names to evaluate (must match registry names).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("offline_eval_results.csv"),
        help="Path to write evaluation summary CSV.",
    )
    args = parser.parse_args()

    videos = find_videos(args.video_dir)
    if not videos:
        print(f"No videos found in {args.video_dir} matching naming convention.", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, object]] = []

    for meta in videos:
        for backend_name in args.backends:
            print(f"[INFO] Evaluating {meta.path.name} with backend={backend_name} ...")
            try:
                row = evaluate_video_with_backend(meta, backend_name)
            except Exception as exc:
                print(f"[WARN] Failed on {meta.path.name} with {backend_name}: {exc}", file=sys.stderr)
                row = {
                    "video": str(meta.path),
                    "exercise": meta.exercise,
                    "pitch": meta.pitch,
                    "yaw": meta.yaw,
                    "take": meta.take,
                    "backend": backend_name,
                    "frames": 0,
                    "avg_latency_ms": None,
                    "fps": None,
                    "curl_counter": None,
                    "squat_counter": None,
                }
            rows.append(row)

    # Write CSV summary.
    fieldnames = [
        "video",
        "exercise",
        "pitch",
        "yaw",
        "take",
        "backend",
        "frames",
        "avg_latency_ms",
        "fps",
        "curl_counter",
        "squat_counter",
        "elbow_mean",
        "elbow_std",
        "elbow_mean_abs_delta",
        "left_elbow_mean",
        "left_elbow_std",
        "left_elbow_mean_abs_delta",
        "knee_mean",
        "knee_std",
        "knee_mean_abs_delta",
        "left_knee_mean",
        "left_knee_std",
        "left_knee_mean_abs_delta",
        "shl_mean",
        "shl_std",
        "shl_mean_abs_delta",
        "rshl_rpalm_mean",
        "rshl_rpalm_std",
        "rshl_rpalm_mean_abs_delta",
        "rshl_rhip_mean",
        "rshl_rhip_std",
        "rshl_rhip_mean_abs_delta",
        "rpalm_rhip_mean",
        "rpalm_rhip_std",
        "rpalm_rhip_mean_abs_delta",
        "rknee_rhip_mean",
        "rknee_rhip_std",
        "rknee_rhip_mean_abs_delta",
        "rknee_rfeet_mean",
        "rknee_rfeet_std",
        "rknee_rfeet_mean_abs_delta",
        "rhip_rfeet_mean",
        "rhip_rfeet_std",
        "rhip_rfeet_mean_abs_delta",
    ]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[INFO] Evaluation summary written to {args.output_csv}")


if __name__ == "__main__":
    main()
