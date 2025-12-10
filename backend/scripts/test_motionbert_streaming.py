"""
Streaming-style latency test for MotionBERT (using MoveNet service on real video).

思路：
  - 从一段真实视频逐帧读取图像；
  - 每帧通过已有的 MoveNet HTTP 服务获取 2D 关键点；
  - 将最近 window 帧 (默认 81) 的 2D 关键点作为一个序列送入 MotionBERT；
  - 记录每次前向的耗时，统计 mean/median/95th 等。

注意：
  - 需要提前在 tf-macos 环境里启动 MoveNet 服务：

        python backend/services/movenet_service.py \\
          --model models/movenet_3d.tflite \\
          --host 127.0.0.1 --port 8502

  - 本脚本只关注计算延迟，不计算误差。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import requests
import torch
import sys

# 保证可以找到 MotionBERT 仓库（与 eval_motionbert_fit3d 的做法一致）
THIS_DIR = Path(__file__).resolve().parents[2]
MB_ROOT = THIS_DIR.parent / "MotionBERT"
if str(MB_ROOT) not in sys.path:
    sys.path.insert(0, str(MB_ROOT))

from lib.utils.tools import get_config  # type: ignore  # noqa: E402
from lib.utils.learning import load_backbone  # type: ignore  # noqa: E402
from backend.scripts.compute_fit3d_gt_metrics import EXERCISE_TO_FIT3D  # type: ignore  # noqa: E402


H36M_IDX: Dict[str, int] = {
    "hip": 0,
    "right_hip": 1,
    "right_knee": 2,
    "right_ankle": 3,
    "left_hip": 4,
    "left_knee": 5,
    "left_ankle": 6,
    "spine": 7,
    "thorax": 8,
    "neck": 9,
    "head": 10,
    "left_shoulder": 11,
    "left_elbow": 12,
    "left_wrist": 13,
    "right_shoulder": 14,
    "right_elbow": 15,
    "right_wrist": 16,
}


def movenet_frame_to_h36m(
    frame_bgr: np.ndarray,
    service_url: str,
    timeout: float,
) -> np.ndarray | None:
    """调用 MoveNet 服务，返回一帧的 H36M 2D 关键点 [17,3]（归一化到 [-1,1]）。"""
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return None
    frame_b64 = base64.b64encode(buf).decode("utf-8")
    try:
        resp = requests.post(service_url, json={"frame": frame_b64}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        keypoints = payload.get("keypoints")
    except Exception:
        return None
    if keypoints is None:
        return None

    kp = np.asarray(keypoints, dtype=float)  # [17,3] (y,x,score) in [0,1]
    y_norm = kp[:, 0]
    x_norm = kp[:, 1]
    conf = kp[:, 2]

    center = np.array([0.5, 0.5], dtype=float)
    scale = 0.5

    pts: Dict[str, np.ndarray] = {}
    pts["left_shoulder"] = np.array([x_norm[5], y_norm[5], conf[5]])
    pts["right_shoulder"] = np.array([x_norm[6], y_norm[6], conf[6]])
    pts["left_elbow"] = np.array([x_norm[7], y_norm[7], conf[7]])
    pts["right_elbow"] = np.array([x_norm[8], y_norm[8], conf[8]])
    pts["left_wrist"] = np.array([x_norm[9], y_norm[9], conf[9]])
    pts["right_wrist"] = np.array([x_norm[10], y_norm[10], conf[10]])
    pts["left_hip"] = np.array([x_norm[11], y_norm[11], conf[11]])
    pts["right_hip"] = np.array([x_norm[12], y_norm[12], conf[12]])
    pts["left_knee"] = np.array([x_norm[13], y_norm[13], conf[13]])
    pts["right_knee"] = np.array([x_norm[14], y_norm[14], conf[14]])
    pts["left_ankle"] = np.array([x_norm[15], y_norm[15], conf[15]])
    pts["right_ankle"] = np.array([x_norm[16], y_norm[16], conf[16]])
    pts["nose"] = np.array([x_norm[0], y_norm[0], conf[0]])

    hip = (pts["left_hip"] + pts["right_hip"]) / 2.0
    thorax = (pts["left_shoulder"] + pts["right_shoulder"]) / 2.0
    spine = (hip + thorax) / 2.0

    kpts = np.zeros((17, 3), dtype=float)
    kpts[H36M_IDX["hip"]] = hip
    kpts[H36M_IDX["right_hip"]] = pts["right_hip"]
    kpts[H36M_IDX["right_knee"]] = pts["right_knee"]
    kpts[H36M_IDX["right_ankle"]] = pts["right_ankle"]
    kpts[H36M_IDX["left_hip"]] = pts["left_hip"]
    kpts[H36M_IDX["left_knee"]] = pts["left_knee"]
    kpts[H36M_IDX["left_ankle"]] = pts["left_ankle"]
    kpts[H36M_IDX["spine"]] = spine
    kpts[H36M_IDX["thorax"]] = thorax
    kpts[H36M_IDX["neck"]] = thorax
    kpts[H36M_IDX["head"]] = pts["nose"]
    kpts[H36M_IDX["left_shoulder"]] = pts["left_shoulder"]
    kpts[H36M_IDX["left_elbow"]] = pts["left_elbow"]
    kpts[H36M_IDX["left_wrist"]] = pts["left_wrist"]
    kpts[H36M_IDX["right_shoulder"]] = pts["right_shoulder"]
    kpts[H36M_IDX["right_elbow"]] = pts["right_elbow"]
    kpts[H36M_IDX["right_wrist"]] = pts["right_wrist"]

    coords = kpts[:, :2]
    coords = (coords - center) / scale  # [-1,1]
    kpts[:, :2] = coords
    return kpts.astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure streaming MotionBERT latency on a real video.")
    parser.add_argument(
        "--video",
        type=Path,
        help="Path to an RGB video. 如果省略，则用 Fit3D: --exercise + --view + --subject.",
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default="barbell_row",
        help="Fit3D internal exercise name (when --video 未指定)。",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="s03",
        help="Fit3D subject (for default video path).",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="50591643",
        help="Fit3D camera id (for default video path).",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("fit3d_subset/train"),
        help="Root containing Fit3D videos (sXX/videos/<view>/<name>.mp4).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../MotionBERT/configs/pose3d/MB_ft_h36m.yaml"),
        help="MotionBERT config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../MotionBERT/checkpoint/pose3d/MB_ft_h36m/best_epoch.bin"),
        help="MotionBERT checkpoint.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=81,
        help="Temporal window length for streaming lifting.",
    )
    parser.add_argument(
        "--movenet-service-url",
        type=str,
        default=os.getenv("MOVENET_SERVICE_URL", "http://127.0.0.1:8502/infer"),
        help="MoveNet HTTP service URL.",
    )
    parser.add_argument(
        "--movenet-timeout",
        type=float,
        default=float(os.getenv("MOVENET_SERVICE_TIMEOUT", "2.0")),
        help="Timeout (seconds) for MoveNet HTTP calls.",
    )
    args = parser.parse_args()

    # Resolve video path
    if args.video is not None:
        video_path = args.video
    else:
        fit3d_name = EXERCISE_TO_FIT3D[args.exercise]
        video_path = (
            args.video_root
            / args.subject
            / "videos"
            / args.view
            / f"{fit3d_name}.mp4"
        )

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    # Load MotionBERT
    cfg = get_config(str(args.config))
    model = load_backbone(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_pos"], strict=False)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    from time import perf_counter

    buffer: List[np.ndarray] = []
    forward_times_ms: List[float] = []
    total_frames = 0
    used_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        kpts = movenet_frame_to_h36m(
            frame,
            service_url=args.movenet_service_url,
            timeout=args.movenet_timeout,
        )
        if kpts is None:
            continue
        buffer.append(kpts)

        if len(buffer) < args.window:
            continue
        if len(buffer) > args.window:
            buffer.pop(0)

        # Window ready: run MotionBERT once
        x = torch.from_numpy(np.stack(buffer, axis=0))[None, ...]  # [1,T,17,3]
        t0 = perf_counter()
        with torch.no_grad():
            _ = model(x)
        dt_ms = (perf_counter() - t0) * 1000.0
        forward_times_ms.append(dt_ms)
        used_frames += 1

    cap.release()

    if not forward_times_ms:
        print("No valid frames / windows; check MoveNet service and video.")
        return

    arr = np.array(forward_times_ms, dtype=float)
    mean_ms = float(arr.mean())
    med_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))
    fps_equiv = 1000.0 / mean_ms

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Total frames read: {total_frames}")
    print(f"[INFO] MotionBERT calls (windows): {used_frames}")
    print(
        f"[INFO] Per-window forward ms (T={args.window}): "
        f"mean={mean_ms:.1f}, median={med_ms:.1f}, p95={p95_ms:.1f}, "
        f"equiv_fps={fps_equiv:.1f}"
    )
    # 理论算法性延迟（窗口中心）：
    print(
        f"[INFO] Algorithmic latency (window/2 at 30fps): "
        f"{args.window/2/30:.2f} seconds (不含算力开销)"
    )


if __name__ == "__main__":
    main()
