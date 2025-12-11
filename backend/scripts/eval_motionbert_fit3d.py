"""
Run MotionBERT (3D pose from 2D keypoints) on a subset of Fit3D videos
and export metrics in the same format as offline_eval_fit3d_gt.csv.

Scope:
  - subject: s03
  - exercises: squat, barbell_row
  - all camera views for s03 in gt_meta.csv
  - back-end name: motionbert3d

Pipeline per clip:
  1) From gt_meta.csv get (subject=view_id, exercise, start_frame, end_frame).
  2) Load corresponding RGB video:
       fit3d_subset/train/s03/videos/<view>/<exercise_name>.mp4
     where exercise_name is mapped via EXERCISE_TO_FIT3D.
  3) For frames in [start_frame, end_frame], extract 2D keypoints with
     MediaPipe Pose, convert to H36M 17-joint format (x,y,conf) and
     normalise to [-1,1] as in MotionBERT's WildDetDataset.
  4) Run MotionBERT (configs/pose3d/MB_ft_h36m.yaml + checkpoint).
     Input shape: [1,T,17,3], output: [1,T,17,3] 3D joints (root-relative).
  5) From 3D joints compute the same angle/distance metrics as GT:
       - right/left elbow & knee angles
       - shoulder distance, hand/hip/knee distances (right side)
  6) Append rows to a new CSV that combines existing offline_eval_fit3d_gt.csv
     with MotionBERT rows.

This is an offline analysis script; it does NOT modify real-time backends.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# --- Paths for MotionBERT repo ------------------------------------------------

THIS_DIR = Path(__file__).resolve().parents[2]
MB_ROOT = THIS_DIR.parent / "MotionBERT"
if str(MB_ROOT) not in sys.path:
    sys.path.insert(0, str(MB_ROOT))

from lib.utils.tools import get_config  # type: ignore
from lib.utils.learning import load_backbone  # type: ignore


# --- Import mapping from exercise name to Fit3D filename ----------------------

from backend.scripts.compute_fit3d_gt_metrics import EXERCISE_TO_FIT3D  # type: ignore


@dataclass
class MetaRow:
    video_rel: str
    exercise: str
    gt_reps: int
    start_frame: Optional[int]
    end_frame: Optional[int]
    subject: str
    view: str


def load_meta(meta_csv: Path) -> List[MetaRow]:
    rows: List[MetaRow] = []
    with meta_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exercise = (r.get("exercise") or "").strip()
            if exercise not in EXERCISE_TO_FIT3D:
                continue
            subject = (r.get("subject") or "").strip()
            view = (r.get("view") or "").strip()
            video_rel = (r.get("video") or "").strip()
            if not subject or not view or not video_rel:
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
                    exercise=exercise,
                    gt_reps=gt_reps,
                    start_frame=sf,
                    end_frame=ef,
                    subject=subject,
                    view=view,
                )
            )
    return rows


# --- H36M joint index mapping -------------------------------------------------

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


def summarize(series: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [x for x in series if np.isfinite(x)]
    if not vals:
        return None, None, None
    a = np.array(vals, dtype=float)
    mean = float(a.mean())
    std = float(a.std())
    if len(a) > 1:
        mad = float(np.mean(np.abs(np.diff(a))))
    else:
        mad = None
    return mean, std, mad


def mediapipe_to_h36m_sequence(
    video_path: Path,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> np.ndarray:
    """Extract 2D keypoints from video using MediaPipe Pose and map to H36M 17 joints."""
    import mediapipe as mp  # local import to avoid hard dependency if unused

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = min(w, h) / 2.0
    center = np.array([w, h], dtype=float) / 2.0

    frames: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if start_frame is not None and frame_idx < start_frame:
            continue
        if end_frame is not None and frame_idx > end_frame:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(image_rgb)
        if not res.pose_landmarks:
            continue
        lm = res.pose_landmarks.landmark

        def pt(idx: int) -> Tuple[float, float, float]:
            p = lm[idx]
            x_pix = p.x * w
            y_pix = p.y * h
            conf = p.visibility
            return x_pix, y_pix, conf

        # MediaPipe indices
        L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        L_EL = mp_pose.PoseLandmark.LEFT_ELBOW.value
        R_EL = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        L_WR = mp_pose.PoseLandmark.LEFT_WRIST.value
        R_WR = mp_pose.PoseLandmark.RIGHT_WRIST.value
        L_HP = mp_pose.PoseLandmark.LEFT_HIP.value
        R_HP = mp_pose.PoseLandmark.RIGHT_HIP.value
        L_KN = mp_pose.PoseLandmark.LEFT_KNEE.value
        R_KN = mp_pose.PoseLandmark.RIGHT_KNEE.value
        L_AN = mp_pose.PoseLandmark.LEFT_ANKLE.value
        R_AN = mp_pose.PoseLandmark.RIGHT_ANKLE.value
        NOSE = mp_pose.PoseLandmark.NOSE.value

        pts = {}
        pts["left_shoulder"] = np.array(pt(L_SH), dtype=float)
        pts["right_shoulder"] = np.array(pt(R_SH), dtype=float)
        pts["left_elbow"] = np.array(pt(L_EL), dtype=float)
        pts["right_elbow"] = np.array(pt(R_EL), dtype=float)
        pts["left_wrist"] = np.array(pt(L_WR), dtype=float)
        pts["right_wrist"] = np.array(pt(R_WR), dtype=float)
        pts["left_hip"] = np.array(pt(L_HP), dtype=float)
        pts["right_hip"] = np.array(pt(R_HP), dtype=float)
        pts["left_knee"] = np.array(pt(L_KN), dtype=float)
        pts["right_knee"] = np.array(pt(R_KN), dtype=float)
        pts["left_ankle"] = np.array(pt(L_AN), dtype=float)
        pts["right_ankle"] = np.array(pt(R_AN), dtype=float)
        pts["nose"] = np.array(pt(NOSE), dtype=float)

        hip = (pts["left_hip"] + pts["right_hip"]) / 2.0
        thorax = (pts["left_shoulder"] + pts["right_shoulder"]) / 2.0
        spine = (hip + thorax) / 2.0

        kpts = np.zeros((17, 3), dtype=float)
        # H36M layout
        kpts[H36M_IDX["hip"]] = hip
        kpts[H36M_IDX["right_hip"]] = pts["right_hip"]
        kpts[H36M_IDX["right_knee"]] = pts["right_knee"]
        kpts[H36M_IDX["right_ankle"]] = pts["right_ankle"]
        kpts[H36M_IDX["left_hip"]] = pts["left_hip"]
        kpts[H36M_IDX["left_knee"]] = pts["left_knee"]
        kpts[H36M_IDX["left_ankle"]] = pts["left_ankle"]
        kpts[H36M_IDX["spine"]] = spine
        kpts[H36M_IDX["thorax"]] = thorax
        kpts[H36M_IDX["neck"]] = thorax  # approximate
        kpts[H36M_IDX["head"]] = pts["nose"]
        kpts[H36M_IDX["left_shoulder"]] = pts["left_shoulder"]
        kpts[H36M_IDX["left_elbow"]] = pts["left_elbow"]
        kpts[H36M_IDX["left_wrist"]] = pts["left_wrist"]
        kpts[H36M_IDX["right_shoulder"]] = pts["right_shoulder"]
        kpts[H36M_IDX["right_elbow"]] = pts["right_elbow"]
        kpts[H36M_IDX["right_wrist"]] = pts["right_wrist"]

        # Normalise x,y to [-1,1] as in dataset_wild (pixel mode)
        coords = kpts[:, :2]
        coords = coords - center
        coords = coords / scale
        kpts[:, :2] = coords
        frames.append(kpts)

    cap.release()

    if not frames:
        raise RuntimeError(f"No valid pose frames extracted from {video_path}")

    motion = np.stack(frames, axis=0)  # [T,17,3]
    return motion.astype(np.float32)


def metrics_from_3d(pred_3d: np.ndarray) -> Dict[str, object]:
    """Compute angle/distance metrics from MotionBERT 3D output."""
    frames = pred_3d  # [T,17,3]

    # indices
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

        elbow_R_series.append(joint_angle(p_rs, p_re, p_rw))
        elbow_L_series.append(joint_angle(p_ls, p_le, p_lw))
        knee_R_series.append(joint_angle(p_rh, p_rk, p_ra))
        knee_L_series.append(joint_angle(p_lh, p_lk, p_la))

        shl_series.append(float(np.linalg.norm(p_ls - p_rs)))
        rshl_rpalm_series.append(float(p_rs[1] - p_rw[1]))
        rshl_rhip_series.append(float(p_rh[1] - p_rs[1]))
        rpalm_rhip_series.append(float(p_rh[1] - p_rw[1]))
        rknee_rhip_series.append(float(p_rh[1] - p_rk[1]))
        rknee_rfeet_series.append(float(p_ra[1] - p_rk[1]))
        rhip_rfeet_series.append(float(p_ra[1] - p_rh[1]))

    elbow_R_mean, elbow_R_std, elbow_R_mad = summarize(elbow_R_series)
    elbow_L_mean, elbow_L_std, elbow_L_mad = summarize(elbow_L_series)
    knee_R_mean, knee_R_std, knee_R_mad = summarize(knee_R_series)
    knee_L_mean, knee_L_std, knee_L_mad = summarize(knee_L_series)
    shl_mean, shl_std, shl_mad = summarize(shl_series)
    rshl_rpalm_mean, rshl_rpalm_std, rshl_rpalm_mad = summarize(rshl_rpalm_series)
    rshl_rhip_mean, rshl_rhip_std, rshl_rhip_mad = summarize(rshl_rhip_series)
    rpalm_rhip_mean, rpalm_rhip_std, rpalm_rhip_mad = summarize(rpalm_rhip_series)
    rknee_rhip_mean, rknee_rhip_std, rknee_rhip_mad = summarize(rknee_rhip_series)
    rknee_rfeet_mean, rknee_rfeet_std, rknee_rfeet_mad = summarize(rknee_rfeet_series)
    rhip_rfeet_mean, rhip_rfeet_std, rhip_rfeet_mad = summarize(rhip_rfeet_series)

    return {
        "frames": len(frames),
        "elbow_mean": elbow_R_mean,
        "elbow_std": elbow_R_std,
        "elbow_mean_abs_delta": elbow_R_mad,
        "left_elbow_mean": elbow_L_mean,
        "left_elbow_std": elbow_L_std,
        "left_elbow_mean_abs_delta": elbow_L_mad,
        "knee_mean": knee_R_mean,
        "knee_std": knee_R_std,
        "knee_mean_abs_delta": knee_R_mad,
        "left_knee_mean": knee_L_mean,
        "left_knee_std": knee_L_std,
        "left_knee_mean_abs_delta": knee_L_mad,
        "shl_mean": shl_mean,
        "shl_std": shl_std,
        "shl_mean_abs_delta": shl_mad,
        "rshl_rpalm_mean": rshl_rpalm_mean,
        "rshl_rpalm_std": rshl_rpalm_std,
        "rshl_rpalm_mean_abs_delta": rshl_rpalm_mad,
        "rshl_rhip_mean": rshl_rhip_mean,
        "rshl_rhip_std": rshl_rhip_std,
        "rshl_rhip_mean_abs_delta": rshl_rhip_mad,
        "rpalm_rhip_mean": rpalm_rhip_mean,
        "rpalm_rhip_std": rpalm_rhip_std,
        "rpalm_rhip_mean_abs_delta": rpalm_rhip_mad,
        "rknee_rhip_mean": rknee_rhip_mean,
        "rknee_rhip_std": rknee_rhip_std,
        "rknee_rhip_mean_abs_delta": rknee_rhip_mad,
        "rknee_rfeet_mean": rknee_rfeet_mean,
        "rknee_rfeet_std": rknee_rfeet_std,
        "rknee_rfeet_mean_abs_delta": rknee_rfeet_mad,
        "rhip_rfeet_mean": rhip_rfeet_mean,
        "rhip_rfeet_std": rhip_rfeet_std,
        "rhip_rfeet_mean_abs_delta": rhip_rfeet_mad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MotionBERT on Fit3D subset (s03 squat/row).")
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
        help="Fit3D train root (containing s03/videos).",
    )
    parser.add_argument(
        "--base-model-csv",
        type=Path,
        default=Path("offline_eval_fit3d_gt.csv"),
        help="Existing model CSV to merge with (from offline_eval_gt).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("offline_eval_fit3d_with_motionbert.csv"),
        help="Output CSV path with MotionBERT rows appended.",
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
        "--subject",
        type=str,
        default="s03",
        help="Subject ID to analyse.",
    )
    parser.add_argument(
        "--exercises",
        nargs="+",
        default=["squat", "barbell_row"],
        help="Exercises to evaluate with MotionBERT.",
    )
    args = parser.parse_args()

    # Load MotionBERT model
    cfg = get_config(str(args.config))
    model = load_backbone(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # 有些预训练权重和当前代码库的键名略有差异，放宽 strict 以便加载。
    missing, unexpected = model.load_state_dict(checkpoint["model_pos"], strict=False)
    if missing or unexpected:
        print(f"[WARN] MotionBERT missing keys: {missing}")
        print(f"[WARN] MotionBERT unexpected keys: {unexpected}")
    model.eval()

    # Prepare meta rows for subject + selected exercises
    meta_all = load_meta(args.gt_meta)
    meta_rows = [
        m for m in meta_all if m.subject == args.subject and m.exercise in args.exercises
    ]
    if not meta_rows:
        print("[WARN] No matching rows in gt_meta for subject/exercises.")

    mb_rows: List[Dict[str, object]] = []
    for m in meta_rows:
        fit3d_name = EXERCISE_TO_FIT3D[m.exercise]
        video_path = args.video_root / m.subject / "videos" / m.view / f"{fit3d_name}.mp4"
        if not video_path.exists():
            print(f"[WARN] Video missing for MotionBERT: {video_path}")
            continue

        print(f"[INFO] MotionBERT on {video_path.name} (ex={m.exercise}, view={m.view})")
        motion = mediapipe_to_h36m_sequence(video_path, m.start_frame, m.end_frame)
        T = motion.shape[0]

        # MotionBERT 预训练支持的最大时间长度为 243 帧。
        # 为避免信息损失，这里采用分段方式：将整段序列切成若干段
        # (每段长度<=243)，分别前向，再在时间维上拼接 3D 结果。
        chunks: List[np.ndarray] = []
        max_len = 243
        start = 0
        import time
        t0 = time.perf_counter()
        while start < T:
            end = min(start + max_len, T)
            clip = motion[start:end]  # [Tc,17,3]
            x = torch.from_numpy(clip)[None, ...]  # [1,Tc,17,3]
            with torch.no_grad():
                out = model(x).cpu().numpy()[0]  # [Tc,17,3]
            chunks.append(out)
            start = end
        pred_3d = np.concatenate(chunks, axis=0)  # [T,17,3]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_frame_ms = elapsed_ms / T if T > 0 else float("nan")
        print(f"[INFO] MotionBERT latency: {elapsed_ms:.1f} ms total, {per_frame_ms:.3f} ms/frame for T={T}")

        metrics = metrics_from_3d(pred_3d)

        row: Dict[str, object] = {
            "video": str(video_path),
            "exercise": m.exercise,
            "pitch": "",        # not used in GT comparison
            "yaw": "",
            "take": "",
            "backend": "motionbert3d",
            "frames": metrics["frames"],
            "avg_latency_ms": elapsed_ms,
            "fps": (T / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else None,
            "curl_counter": None,
            "squat_counter": None,
        }
        # merge angle/distance metrics
        for k, v in metrics.items():
            if k == "frames":
                continue
            row[k] = v

        # GT-related fields (subject/view/gt_reps); rep counts left None
        row["subject"] = m.subject
        row["view"] = m.view
        row["gt_reps"] = m.gt_reps
        row["pred_reps"] = None
        row["rep_error"] = None
        row["abs_rep_error"] = None

        mb_rows.append(row)

    # Merge existing model CSV and MotionBERT rows into new file
    if args.base_model_csv.exists():
        with args.base_model_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            base_fieldnames = reader.fieldnames or []
            base_rows = list(reader)
    else:
        base_fieldnames = []
        base_rows = []

    # union of fieldnames
    fieldnames_set = set(base_fieldnames)
    for r in mb_rows:
        fieldnames_set.update(r.keys())
    fieldnames = list(fieldnames_set)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in base_rows:
            writer.writerow(r)
        for r in mb_rows:
            writer.writerow(r)

    print(f"[INFO] Wrote combined CSV with MotionBERT to {args.output_csv}")


if __name__ == "__main__":
    main()
