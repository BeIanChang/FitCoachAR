"""
Framewise MotionBERT 3D metrics on Fit3D videos using MoveNet 2D as input.

和上层目录 backend/scripts/framewise_motionbert.py 相同逻辑，只是放在
FitCoachAR 包内部，方便通过

    python -m backend.scripts.framewise_motionbert

在 FitCoachAR 根目录下直接调用。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import base64
import cv2
import numpy as np
import torch
import requests
import os

# ---- repo paths --------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parents[2]
MB_ROOT = THIS_DIR.parent / "MotionBERT"
if str(MB_ROOT) not in sys.path:
    sys.path.insert(0, str(MB_ROOT))

from lib.utils.tools import get_config  # type: ignore  # noqa: E402
from lib.utils.learning import load_backbone  # type: ignore  # noqa: E402
from backend.scripts.compute_fit3d_gt_metrics import EXERCISE_TO_FIT3D  # type: ignore  # noqa: E402


@dataclass
class MetaRow:
    video_rel: str
    exercise: str
    gt_reps: int
    start_frame: Optional[int]
    end_frame: Optional[int]
    subject: str
    view: str


def load_meta(meta_csv: Path, subject: str, exercises: List[str]) -> List[MetaRow]:
    rows: List[MetaRow] = []
    with meta_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ex = (r.get("exercise") or "").strip()
            if ex not in exercises or ex not in EXERCISE_TO_FIT3D:
                continue
            subj = (r.get("subject") or "").strip()
            if subj != subject:
                continue
            view = (r.get("view") or "").strip()
            video_rel = (r.get("video") or "").strip()
            if not view or not video_rel:
                continue
            try:
                gt_reps = int((r.get("gt_reps") or "0").strip())
            except Exception:
                gt_reps = 0
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
                    gt_reps=gt_reps,
                    start_frame=sf,
                    end_frame=ef,
                    subject=subj,
                    view=view,
                )
            )
    return rows


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


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0.0 or v2_norm == 0.0:
        return float("nan")
    cos_theta = float(np.dot(v1, v2) / (v1_norm * v2_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def movenet_to_h36m_sequence(
    video_path: Path,
    start_frame: Optional[int],
    end_frame: Optional[int],
    service_url: str,
    timeout: float,
) -> Tuple[np.ndarray, List[int]]:
    """用外部 MoveNet2D 服务对视频逐帧做推理，并映射到 H36M 关节。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # 服务返回的坐标是 [0,1] 归一化，这里直接在归一化空间做中心化和缩放：
    # center=(0.5,0.5), scale=0.5  ==> [-1,1]。
    center = np.array([0.5, 0.5], dtype=float)
    scale = 0.5

    frames: List[np.ndarray] = []
    indices: List[int] = []
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
        # 编码图片并调用 MoveNet 服务
        ok_enc, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok_enc:
            continue
        frame_b64 = base64.b64encode(buf).decode("utf-8")
        try:
            resp = requests.post(
                service_url,
                json={"frame": frame_b64},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            keypoints = payload.get("keypoints")
        except Exception:
            continue
        if keypoints is None:
            continue

        kp = np.asarray(keypoints, dtype=float)  # [17,3] (y,x,score)
        y_norm = kp[:, 0]
        x_norm = kp[:, 1]
        conf = kp[:, 2]

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
        coords = coords - center
        coords = coords / scale
        kpts[:, :2] = coords

        frames.append(kpts)
        indices.append(frame_idx)

    cap.release()

    if not frames:
        raise RuntimeError(f"No valid MoveNet pose frames extracted from {video_path}")

    motion = np.stack(frames, axis=0).astype(np.float32)
    return motion, indices


def metrics_from_3d_per_frame(pred_3d: np.ndarray) -> List[Dict[str, float]]:
    """从 MotionBERT 3D 逐帧计算角度 / 距离特征。"""
    frames = pred_3d

    ls = H36M_IDX["left_shoulder"]
    rs = H36M_IDX["right_shoulder"]
    le = H36M_IDX["left_elbow"]
    re = H36M_IDX["right_elbow"]
    lw = H36M_IDX["left_wrist"]
    rw = H36M_IDX["right_wrist"]
    lh = H36M_IDX["left_hip"]
    rh = H36M_IDX["right_hip"]
    lk = H36M_IDX["left_knee"]
    rk = H36M_IDX["right_knee"]
    la = H36M_IDX["left_ankle"]
    ra = H36M_IDX["right_ankle"]

    out: List[Dict[str, float]] = []
    for f in frames:
        p_ls = f[ls]
        p_rs = f[rs]
        p_le = f[le]
        p_re = f[re]
        p_lw = f[lw]
        p_rw = f[rw]
        p_lh = f[lh]
        p_rh = f[rh]
        p_lk = f[lk]
        p_rk = f[rk]
        p_la = f[la]
        p_ra = f[ra]

        elbow_R = joint_angle(p_rs, p_re, p_rw)
        elbow_L = joint_angle(p_ls, p_le, p_lw)
        knee_R = joint_angle(p_rh, p_rk, p_ra)
        knee_L = joint_angle(p_lh, p_lk, p_la)

        shl = float(np.linalg.norm(p_ls - p_rs))
        rshl_rpalm = float(p_rs[1] - p_rw[1])
        rknee_rhip = float(p_rh[1] - p_rk[1])
        rhip_rfeet = float(p_ra[1] - p_rh[1])

        out.append(
            {
                "elbow_R": float(elbow_R),
                "elbow_L": float(elbow_L),
                "knee_R": float(knee_R),
                "knee_L": float(knee_L),
                "shl": shl,
                "rshl_rpalm": rshl_rpalm,
                "rknee_rhip": rknee_rhip,
                "rhip_rfeet": rhip_rfeet,
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Framewise MotionBERT (MoveNet2D) on Fit3D subset.")
    parser.add_argument(
        "--gt-meta",
        type=Path,
        default=Path("fit3d_subset/gt_meta.csv"),
        help="Path to gt_meta.csv.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("fit3d_subset/train"),
        help="Root containing sXX/videos/<view>/<exercise>.mp4.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="s03",
        help="Subject ID to analyse.",
    )
    parser.add_argument(
        "--exercises",
        nargs="+",
        default=["squat", "barbell_row"],
        help="Exercises to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pose3d/MB_ft_h36m.yaml"),
        help="MotionBERT pose3d config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint/pose3d/MB_ft_h36m/best_epoch.bin"),
        help="MotionBERT checkpoint path.",
    )
    parser.add_argument(
        "--movenet-service-url",
        type=str,
        default=os.getenv("MOVENET_SERVICE_URL", "http://127.0.0.1:8502/infer"),
        help="MoveNet 2D inference service URL (same as realtime backend).",
    )
    parser.add_argument(
        "--movenet-timeout",
        type=float,
        default=float(os.getenv("MOVENET_SERVICE_TIMEOUT", "2.0")),
        help="HTTP timeout (seconds) when calling MoveNet service.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("framewise_results/motionbert3d"),
        help="Directory to write JSON files.",
    )
    args = parser.parse_args()

    cfg = get_config(str(args.config))
    model = load_backbone(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model_pos"], strict=False)
    if missing or unexpected:
        print(f"[WARN] MotionBERT missing keys: {missing}")
        print(f"[WARN] MotionBERT unexpected keys: {unexpected}")
    model.eval()

    meta_rows = load_meta(args.gt_meta, args.subject, args.exercises)
    if not meta_rows:
        print("[WARN] No matching rows in gt_meta for subject/exercises.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import time

    for m in meta_rows:
        fit3d_name = EXERCISE_TO_FIT3D[m.exercise]
        video_path = args.video_root / m.subject / "videos" / m.view / f"{fit3d_name}.mp4"
        if not video_path.exists():
            print(f"[WARN] Video missing for MotionBERT: {video_path}")
            continue

        print(f"[INFO] MotionBERT (MoveNet2D) on {video_path.name} (ex={m.exercise}, view={m.view})")
        motion, frame_indices = movenet_to_h36m_sequence(
            video_path,
            m.start_frame,
            m.end_frame,
            args.movenet_service_url,
            args.movenet_timeout,
        )
        T = motion.shape[0]

        max_len = 243
        chunks: List[np.ndarray] = []
        start = 0
        t0 = time.perf_counter()
        while start < T:
            end = min(start + max_len, T)
            clip = motion[start:end]
            x = torch.from_numpy(clip)[None, ...]
            with torch.no_grad():
                out = model(x).cpu().numpy()[0]
            chunks.append(out)
            start = end
        pred_3d = np.concatenate(chunks, axis=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_frame_ms = elapsed_ms / T if T > 0 else float("nan")
        print(
            f"[INFO] MotionBERT latency: {elapsed_ms:.1f} ms total, "
            f"{per_frame_ms:.3f} ms/frame for T={T}"
        )

        metrics_list = metrics_from_3d_per_frame(pred_3d)

        frames_out: List[Dict[str, object]] = []
        for idx, fm in zip(frame_indices, metrics_list):
            rec: Dict[str, object] = {"frame_index": idx}
            rec.update(fm)
            frames_out.append(rec)

        sf = m.start_frame or (frame_indices[0] if frame_indices else 1)
        ef = m.end_frame or (frame_indices[-1] if frame_indices else sf)

        out = {
            "subject": m.subject,
            "exercise": m.exercise,
            "view": m.view,
            "backend": "motionbert3d",
            "gt_reps": m.gt_reps,
            "start_frame": sf,
            "end_frame": ef,
            "avg_latency_ms": elapsed_ms,
            "avg_latency_per_frame_ms": per_frame_ms,
            "frames": frames_out,
        }

        out_name = f"{m.exercise}__{m.subject}__{m.view}.json"
        out_path = args.output_dir / out_name
        with out_path.open("w") as f:
            json.dump(out, f)
        print(f"[INFO] wrote MotionBERT framewise -> {out_path}")


if __name__ == "__main__":
    main()
