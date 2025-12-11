"""
Framewise model angle/distance export for FitCoachAR backends (s03).

对每个 (exercise, subject, view, backend) + 帧 t，记录：
  - 右/左肘角: elbow_R, elbow_L
  - 右/左膝角: knee_R, knee_L
  - 右侧距离: shl, rshl_rpalm, rknee_rhip, rhip_rfeet

结果按 “每个视频 × backend 一个 JSON” 导出：
  framewise_results/<backend>/<exercise>__<subject>__<view>.json

后续可与 framewise_gt.py 生成的 GT JSON 做逐帧误差分析。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import mediapipe as mp
import time

# 允许从仓库根目录以 `python -m backend.scripts.framewise_eval_backends` 运行
ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pose_backends  # type: ignore  # noqa: F401
from pose_backends import build_pose_backend  # type: ignore

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


def load_meta(meta_csv: Path, subject: str, exercises: List[str]) -> List[MetaRow]:
    rows: List[MetaRow] = []
    with meta_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exercise = (r.get("exercise") or "").strip()
            if exercise not in exercises:
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
                    exercise=exercise,
                    gt_reps=gt_reps,
                    start_frame=sf,
                    end_frame=ef,
                    subject=subj,
                    view=view,
                )
            )
    return rows


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle ABC (in degrees) for 2D/3D points."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export framewise model metrics for backends.")
    parser.add_argument(
        "--gt-meta",
        type=Path,
        default=Path("fit3d_subset/gt_meta.csv"),
        help="Path to gt_meta.csv (for frame ranges).",
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
        help="Subject ID to export (e.g. s03).",
    )
    parser.add_argument(
        "--exercises",
        nargs="+",
        default=["squat", "barbell_row"],
        help="Exercises to export (internal names).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["mediapipe_2d", "mediapipe_3d", "movenet_3d"],
        help="Backend names as registered in pose_backends.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("framewise_results"),
        help="Root directory to write per-backend JSON files.",
    )
    args = parser.parse_args()

    meta_rows = load_meta(args.gt_meta, args.subject, args.exercises)
    if not meta_rows:
        print("[WARN] No matching rows in gt_meta for given subject/exercises.")
        return

    mp_pose = mp.solutions.pose

    for backend_name in args.backends:
        backend = build_pose_backend(backend_name)
        out_dir = args.output_dir / backend_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for m in meta_rows:
            # Prefer meta-provided relative path (for custom dataset)
            candidate = args.video_root / m.subject / m.video_rel
            video_path = candidate
            if not candidate.exists():
                fit3d_name = EXERCISE_TO_FIT3D.get(m.exercise, m.exercise)
                fallback = (
                    args.video_root / m.subject / "videos" / m.view / f"{fit3d_name}.mp4"
                )
                video_path = fallback
            if not video_path.exists():
                print(f"[WARN] video missing: {video_path}")
                continue

            # 告诉 backend 当前动作（沿用 offline_eval 的逻辑）
            try:
                backend.handle_command({"command": "select_exercise", "exercise": m.exercise})
            except Exception:
                pass

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[WARN] failed to open video: {video_path}")
                continue

            frames_out: List[Dict[str, Any]] = []
            frame_idx = 0
            T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sf = m.start_frame or 1
            ef = m.end_frame or T
            sf = max(1, sf)
            ef = min(T, ef)

            print(
                f"[INFO] {backend_name} on {video_path.name} "
                f"(ex={m.exercise}, view={m.view}, frames {sf}-{ef})"
            )

            t0 = time.perf_counter()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx < sf or frame_idx > ef:
                    continue

                try:
                    result = backend.process_frame(frame)
                except Exception:
                    result = None

                if not result:
                    frames_out.append({"frame_index": frame_idx})
                    continue

                # 对齐 GT JSON 的字段命名
                rec: Dict[str, Any] = {"frame_index": frame_idx}
                # 直接从 backend 的标量结果中取已有角度 / 距离
                rec["elbow_R"] = (
                    float(result.get("right_elbow_angle"))
                    if result.get("right_elbow_angle") is not None
                    else None
                )
                rec["elbow_L"] = (
                    float(result.get("left_elbow_angle"))
                    if result.get("left_elbow_angle") is not None
                    else None
                )
                rec["knee_R"] = (
                    float(result.get("right_knee_angle"))
                    if result.get("right_knee_angle") is not None
                    else None
                )
                rec["knee_L"] = (
                    float(result.get("left_knee_angle"))
                    if result.get("left_knee_angle") is not None
                    else None
                )
                rec["shl"] = (
                    float(result.get("metric_shl_dist"))
                    if result.get("metric_shl_dist") is not None
                    else None
                )
                rec["rshl_rpalm"] = (
                    float(result.get("metric_rshl_rpalm"))
                    if result.get("metric_rshl_rpalm") is not None
                    else None
                )
                rec["rknee_rhip"] = (
                    float(result.get("metric_rknee_rhip"))
                    if result.get("metric_rknee_rhip") is not None
                    else None
                )
                rec["rhip_rfeet"] = (
                    float(result.get("metric_rhip_rfeet"))
                    if result.get("metric_rhip_rfeet") is not None
                    else None
                )

                # 需要的额外距离
                rec["rshl_rhip"] = (
                    float(result.get("metric_rshl_rhip"))
                    if result.get("metric_rshl_rhip") is not None
                    else None
                )
                rec["rpalm_rhip"] = (
                    float(result.get("metric_rpalm_rhip"))
                    if result.get("metric_rpalm_rhip") is not None
                    else None
                )
                rec["rknee_rfeet"] = (
                    float(result.get("metric_rknee_rfeet"))
                    if result.get("metric_rknee_rfeet") is not None
                    else None
                )

                # 从 landmarks 里补算 hip / shoulder 角
                landmarks = result.get("landmarks")
                if landmarks:
                    # mediapipe2d/3d: 33 点, 使用 PoseLandmark 索引;
                    # movenet3d: landmarks 也是一个列表, 我们假设下标兼容主要关节。
                    try:
                        l_sh_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                        r_sh_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                        l_el_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
                        r_el_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                        l_hp_idx = mp_pose.PoseLandmark.LEFT_HIP.value
                        r_hp_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
                        l_kn_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
                        r_kn_idx = mp_pose.PoseLandmark.RIGHT_KNEE.value
                        l_an_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
                        r_an_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value

                        def pt(idx_m: int) -> np.ndarray:
                            lm = landmarks[idx_m]
                            x = float(lm.get("x", 0.0))
                            y = float(lm.get("y", 0.0))
                            z = float(lm.get("z", 0.0))
                            return np.array([x, y, z], dtype=float)

                        p_ls = pt(l_sh_idx)
                        p_rs = pt(r_sh_idx)
                        p_le = pt(l_el_idx)
                        p_re = pt(r_el_idx)
                        p_lh = pt(l_hp_idx)
                        p_rh = pt(r_hp_idx)
                        p_lk = pt(l_kn_idx)
                        p_rk = pt(r_kn_idx)
                        p_la = pt(l_an_idx)
                        p_ra = pt(r_an_idx)

                        hip_R = joint_angle(p_rs, p_rh, p_rk)
                        hip_L = joint_angle(p_ls, p_lh, p_lk)
                        shoulder_R = joint_angle(p_rh, p_rs, p_re)
                        shoulder_L = joint_angle(p_lh, p_ls, p_le)

                        rec["hip_R"] = float(hip_R)
                        rec["hip_L"] = float(hip_L)
                        rec["shoulder_R"] = float(shoulder_R)
                        rec["shoulder_L"] = float(shoulder_L)
                    except Exception:
                        rec.setdefault("hip_R", None)
                        rec.setdefault("hip_L", None)
                        rec.setdefault("shoulder_R", None)
                        rec.setdefault("shoulder_L", None)

                frames_out.append(rec)

            cap.release()

            # latency / fps 统计（只针对处理了的帧区间）
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            frames_proc = len(frames_out)
            fps = (
                frames_proc / (elapsed_ms / 1000.0)
                if frames_proc > 0 and elapsed_ms > 0
                else None
            )

            out = {
                "subject": m.subject,
                "exercise": m.exercise,
                "view": m.view,
                "backend": backend_name,
                "gt_reps": m.gt_reps,
                "start_frame": sf,
                "end_frame": ef,
                "frames_processed": frames_proc,
                "latency_ms_total": elapsed_ms,
                "latency_ms_per_frame": elapsed_ms / frames_proc
                if frames_proc > 0
                else None,
                "fps": fps,
                "frames": frames_out,
            }

            out_name = f"{m.exercise}__{m.subject}__{m.view}.json"
            out_path = out_dir / out_name
            with out_path.open("w") as f:
                json.dump(out, f)
            print(f"[INFO] wrote backend frames -> {out_path}")

        # reset backend state between backends
        try:
            backend.close()  # type: ignore[attr-defined]
        except Exception:
            pass


if __name__ == "__main__":
    main()
