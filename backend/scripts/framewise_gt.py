"""
Framewise GT angle/distance export for Fit3D (s03).

For每个 (exercise, subject, view) + 帧 t，计算：
  - 右/左肘角: elbow_R, elbow_L
  - 右/左膝角: knee_R, knee_L
  - 一些右侧距离: shl, rshl_rpalm, rknee_rhip, rhip_rfeet

结果按“每个视频一个 JSON”导出，结构大致为：
  framewise_results/gt/<exercise>__<subject>__<view>.json

后续各个模型的逐帧结果可以和这些 GT JSON 对齐，再做误差分析。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.scripts.compute_fit3d_gt_metrics import EXERCISE_TO_FIT3D, JOINT_INDEX  # type: ignore


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
    v1 = a - b
    v2 = c - b
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0.0 or v2_norm == 0.0:
        return float("nan")
    cos_theta = float(np.dot(v1, v2) / (v1_norm * v2_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export framewise GT angles/distances for Fit3D.")
    parser.add_argument(
        "--gt-meta",
        type=Path,
        default=Path("fit3d_subset/gt_meta.csv"),
        help="Path to gt_meta.csv.",
    )
    parser.add_argument(
        "--joints-root",
        type=Path,
        default=Path("fit3d_subset/train"),
        help="Root containing sXX/joints3d_25/*.json.",
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
        "--output-dir",
        type=Path,
        default=Path("framewise_results/gt"),
        help="Directory to write per-video GT JSON files.",
    )
    args = parser.parse_args()

    meta_rows = load_meta(args.gt_meta, args.subject, args.exercises)
    if not meta_rows:
        print("[WARN] No matching rows in gt_meta for given subject/exercises.")
        return

    ls = JOINT_INDEX["left_shoulder"]
    rs = JOINT_INDEX["right_shoulder"]
    le = JOINT_INDEX["left_elbow"]
    re = JOINT_INDEX["right_elbow"]
    lw = JOINT_INDEX["left_wrist"]
    rw = JOINT_INDEX["right_wrist"]
    lh = JOINT_INDEX["left_hip"]
    rh = JOINT_INDEX["right_hip"]
    lk = JOINT_INDEX["left_knee"]
    rk = JOINT_INDEX["right_knee"]
    la = JOINT_INDEX["left_ankle"]
    ra = JOINT_INDEX["right_ankle"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for m in meta_rows:
        fit3d_name = EXERCISE_TO_FIT3D[m.exercise]
        joints_path = args.joints_root / m.subject / "joints3d_25" / f"{fit3d_name}.json"
        if not joints_path.exists():
            print(f"[WARN] joints3d file missing: {joints_path}")
            continue

        with joints_path.open() as f:
            data = json.load(f)
        frames = np.array(data["joints3d_25"], dtype=float)  # [T,25,3]
        T = frames.shape[0]

        sf = m.start_frame or 1
        ef = m.end_frame or T
        sf = max(1, sf)
        ef = min(T, ef)

        result_frames: List[Dict[str, float]] = []
        for idx in range(sf - 1, ef):
            f3d = frames[idx]
            p_ls = f3d[ls]
            p_rs = f3d[rs]
            p_le = f3d[le]
            p_re = f3d[re]
            p_lw = f3d[lw]
            p_rw = f3d[rw]
            p_lh = f3d[lh]
            p_rh = f3d[rh]
            p_lk = f3d[lk]
            p_rk = f3d[rk]
            p_la = f3d[la]
            p_ra = f3d[ra]

            elbow_R = joint_angle(p_rs, p_re, p_rw)
            elbow_L = joint_angle(p_ls, p_le, p_lw)
            knee_R = joint_angle(p_rh, p_rk, p_ra)
            knee_L = joint_angle(p_lh, p_lk, p_la)

            # Hip joint angles: shoulder-hip-knee
            hip_R = joint_angle(p_rs, p_rh, p_rk)
            hip_L = joint_angle(p_ls, p_lh, p_lk)

            # Shoulder joint angles: hip-shoulder-elbow
            shoulder_R = joint_angle(p_rh, p_rs, p_re)
            shoulder_L = joint_angle(p_lh, p_ls, p_le)

            shl = float(np.linalg.norm(p_ls - p_rs))
            rshl_rpalm = float(p_rs[1] - p_rw[1])
            rshl_rhip = float(p_rh[1] - p_rs[1])
            rpalm_rhip = float(p_rh[1] - p_rw[1])
            rknee_rhip = float(p_rh[1] - p_rk[1])
            rknee_rfeet = float(p_ra[1] - p_rk[1])
            rhip_rfeet = float(p_ra[1] - p_rh[1])

            result_frames.append(
                {
                    "frame_index": idx + 1,  # 1-based
                    "elbow_R": float(elbow_R),
                    "elbow_L": float(elbow_L),
                    "knee_R": float(knee_R),
                    "knee_L": float(knee_L),
                    "hip_R": float(hip_R),
                    "hip_L": float(hip_L),
                    "shoulder_R": float(shoulder_R),
                    "shoulder_L": float(shoulder_L),
                    "shl": shl,
                    "rshl_rpalm": rshl_rpalm,
                    "rshl_rhip": rshl_rhip,
                    "rpalm_rhip": rpalm_rhip,
                    "rknee_rhip": rknee_rhip,
                    "rknee_rfeet": rknee_rfeet,
                    "rhip_rfeet": rhip_rfeet,
                }
            )

        out = {
            "subject": m.subject,
            "exercise": m.exercise,
            "view": m.view,
            "gt_reps": m.gt_reps,
            "start_frame": sf,
            "end_frame": ef,
            "frames": result_frames,
        }

        out_name = f"{m.exercise}__{m.subject}__{m.view}.json"
        out_path = args.output_dir / out_name
        with out_path.open("w") as f:
            json.dump(out, f)
        print(f"[INFO] wrote GT frames -> {out_path}")


if __name__ == "__main__":
    main()
