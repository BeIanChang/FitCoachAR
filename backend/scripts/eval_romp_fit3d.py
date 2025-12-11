"""
Run the ROMP 3D model on a small subset of Fit3D videos and save raw joints.

Usage (from FitCoachAR root, with ROMP service already running, e.g.):

    # in another terminal, start ROMP service (romp env)
    # uvicorn backend.services.romp_service:app --host 127.0.0.1 --port 8605

    # then in FitCoachAR env:
    python -m backend.scripts.eval_romp_fit3d \
      --fit3d-root fit3d_subset/train \
      --meta-csv   fit3d_subset/gt_meta_s03.csv \
      --subject    s03 \
      --exercises  bicep_curl squat push_up lateral_raise barbell_row \
      --romp-url   http://127.0.0.1:8605/infer_image \
      --out-root   fit3d_romp_preds

For每个 (exercise, subject, view) 只取一条视频（第一条匹配 gt_meta_s03.csv 的记录），
把裁剪后的帧逐帧发给 ROMP 服务，并保存：

  fit3d_romp_preds/<exercise>__<subject>__<view>/romp_joints3d.json
      {
        "exercise": "...",
        "subject": "s03",
        "view": "50591643",
        "video": "fit3d_subset/train/...",
        "frames": T,
        "joints3d": [ [ [x,y,z], ... ], ... ],   # T x J x 3
        "joints2d": [ [ [u,v],   ... ], ... ] or null,
        "latency_ms_total": ...,
        "latency_ms_per_frame": ...
      }

  fit3d_romp_preds/<exercise>__<subject>__<view>/metrics.json
      { "frames": ..., "latency_ms_total": ..., "latency_ms_per_frame": ... }

后续如果要和 GT 做角度 / 距离对比，可以另写脚本读取 joints3d。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


@dataclass
class MetaRow:
    video_rel: str
    exercise: str
    subject: str
    view: str
    start_frame: Optional[int]
    end_frame: Optional[int]


def load_meta(
    meta_csv: Path, subject: str, exercises: List[str]
) -> List[MetaRow]:
    """Load gt_meta.csv and return rows for one subject and a subset of exercises."""
    rows: List[MetaRow] = []
    with meta_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ex = (r.get("exercise") or "").strip()
            if ex not in exercises:
                continue
            subj = (r.get("subject") or "").strip()
            if subj != subject:
                continue
            view = (r.get("view") or "").strip()
            video_rel = (r.get("video") or "").strip()
            if not view or not video_rel:
                continue
            try:
                sf = int((r.get("start_frame") or "").strip())
            except Exception:
                sf = None
            try:
                ef = int((r.get("end_frame") or "").strip())
            except Exception:
                ef = None
            rows.append(
                MetaRow(
                    video_rel=video_rel,
                    exercise=ex,
                    subject=subj,
                    view=view,
                    start_frame=sf,
                    end_frame=ef,
                )
            )
    return rows


def pick_one_view_per_exercise(rows: List[MetaRow]) -> List[MetaRow]:
    """从 meta 里为每个 exercise 选第一条记录."""
    picked: Dict[str, MetaRow] = {}
    for r in rows:
        if r.exercise in picked:
            continue
        picked[r.exercise] = r
    return list(picked.values())


def run_romp_on_clip(
    video_path: Path,
    start_frame: Optional[int],
    end_frame: Optional[int],
    romp_url: str,
    timeout: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], float, int]:
    """
    对一个视频片段逐帧调用 ROMP 服务.

    Returns:
        joints3d: [T, J, 3]
        joints2d: [T, J, 2] or None
        total_latency_ms: float
        frames: int
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    j3_list: List[np.ndarray] = []
    j2_list: List[np.ndarray] = []
    latencies: List[float] = []

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if start_frame is not None and frame_idx < start_frame:
            continue
        if end_frame is not None and frame_idx > end_frame:
            break

        ok_enc, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok_enc:
            continue

        files = {
            "file": ("frame.jpg", buf.tobytes(), "image/jpeg"),
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(romp_url, files=files, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"[WARN] ROMP request failed on {video_path.name} frame {frame_idx}: {e}")
            continue
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        joints3d = np.asarray(payload.get("joints_3d"), dtype=float)
        if joints3d.ndim != 2 or joints3d.shape[1] != 3:
            print(
                f"[WARN] Unexpected joints_3d shape on {video_path.name} frame {frame_idx}: "
                f"{joints3d.shape}"
            )
            continue
        j3_list.append(joints3d)

        joints2d_raw = payload.get("joints_2d")
        if joints2d_raw is not None:
            j2 = np.asarray(joints2d_raw, dtype=float)
            if j2.ndim == 2 and j2.shape[1] >= 2:
                j2_list.append(j2[:, :2])

    cap.release()

    if not j3_list:
        raise RuntimeError(f"No ROMP predictions produced for {video_path}")

    joints3d_seq = np.stack(j3_list, axis=0)  # [T,J,3]
    joints2d_seq: Optional[np.ndarray]
    if j2_list and len(j2_list) == len(j3_list):
        joints2d_seq = np.stack(j2_list, axis=0)  # [T,J,2]
    else:
        joints2d_seq = None

    total_latency_ms = float(sum(latencies))
    return joints3d_seq, joints2d_seq, total_latency_ms, joints3d_seq.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROMP on Fit3D videos and save raw joints.")
    parser.add_argument(
        "--fit3d-root",
        type=Path,
        default=Path("fit3d_subset/train"),
        help="Root of Fit3D subset videos (train/sXX/...).",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=Path("fit3d_subset/gt_meta_s03.csv"),
        help="Meta CSV describing videos, subject, view, and start/end frames.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="s03",
        help="Subject id to filter (e.g., s03).",
    )
    parser.add_argument(
        "--exercises",
        nargs="+",
        default=["bicep_curl", "squat", "push_up", "lateral_raise", "barbell_row"],
        help="Exercises to include.",
    )
    parser.add_argument(
        "--romp-url",
        type=str,
        default="http://127.0.0.1:8605/infer_image",
        help="ROMP service URL (POST /infer_image).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per frame in seconds.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("fit3d_romp_preds"),
        help="Output root for ROMP predictions.",
    )
    args = parser.parse_args()

    meta_rows = load_meta(args.meta_csv, args.subject, args.exercises)
    if not meta_rows:
        print(f"[WARN] No meta rows found for subject={args.subject}")
        return

    clips = pick_one_view_per_exercise(meta_rows)
    print(f"[INFO] Selected {len(clips)} clips:")
    for r in clips:
        print(f"  - {r.exercise}: view={r.view}, video={r.video_rel}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    for r in clips:
        video_path = args.fit3d_root / r.video_rel
        out_dir = args.out_root / f"{r.exercise}__{r.subject}__{r.view}"
        out_dir.mkdir(parents=True, exist_ok=True)
        joints_path = out_dir / "romp_joints3d.json"
        metrics_path = out_dir / "metrics.json"

        print(f"[INFO] ROMP on {video_path} -> {out_dir}")
        try:
            j3, j2, total_ms, frames = run_romp_on_clip(
                video_path=video_path,
                start_frame=r.start_frame,
                end_frame=r.end_frame,
                romp_url=args.romp_url,
                timeout=args.timeout,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {video_path}: {e}")
            continue

        mean_ms = float(total_ms / max(frames, 1))

        data = {
            "exercise": r.exercise,
            "subject": r.subject,
            "view": r.view,
            "video": str(video_path),
            "frames": int(frames),
            "joints3d": j3.tolist(),
            "joints2d": j2.tolist() if j2 is not None else None,
            "latency_ms_total": total_ms,
            "latency_ms_per_frame": mean_ms,
        }
        with joints_path.open("w") as f:
            json.dump(data, f)

        with metrics_path.open("w") as f:
            json.dump(
                {
                    "frames": int(frames),
                    "latency_ms_total": total_ms,
                    "latency_ms_per_frame": mean_ms,
                },
                f,
            )

        print(
            f"[INFO] Saved ROMP joints for {r.exercise} view={r.view}: "
            f"{frames} frames, total {total_ms:.1f} ms, {mean_ms:.2f} ms/frame"
        )


if __name__ == "__main__":
    main()

