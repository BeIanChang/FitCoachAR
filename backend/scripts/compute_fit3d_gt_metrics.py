"""
Compute ground-truth (GT) angle / distance metrics from Fit3D joints3d_25.

Pipeline:
  - Read the same gt_meta.csv that we used for offline_eval_gt.py.
  - For each row (subject, view, exercise, start/end_frame), load the
    corresponding joints3d_25/<exercise>.json file.
  - On the frame range [start_frame, end_frame], compute:
      * right_elbow_angle / left_elbow_angle trajectory (in degrees)
      * right_knee_angle  / left_knee_angle  trajectory
      * the same distance-style metrics used in offline_eval.py:
          - metric_shl_dist      (shoulder distance)
          - metric_rshl_rpalm    (right_shoulder.y - right_wrist.y)
          - metric_rshl_rhip     (right_hip.y - right_shoulder.y)
          - metric_rpalm_rhip    (right_hip.y - right_wrist.y)
          - metric_rknee_rhip    (right_hip.y - right_knee.y)
          - metric_rknee_rfeet   (right_ankle.y - right_knee.y)
          - metric_rhip_rfeet    (right_ankle.y - right_hip.y)
    and summarize each series with mean / std.

Result CSV (default: fit3d_subset/fit3d_gt_metrics.csv) has columns:
  subject,view,exercise,
  elbow_gt_mean,elbow_gt_std, left_elbow_gt_mean,left_elbow_gt_std,
  knee_gt_mean,knee_gt_std,   left_knee_gt_mean,left_knee_gt_std,
  shl_gt_mean,shl_gt_std,
  rshl_rpalm_gt_mean,rshl_rpalm_gt_std,
  ...

IMPORTANT: You MUST set the JOINT_INDEX mapping below to match the
Fit3D joints3d_25 skeleton definition before trusting any numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---- JOINT INDEX MAPPING (MUST BE VERIFIED MANUALLY) -----------------------
#
# Fill these according to the Fit3D "joints3d_25" joint ordering.
# Below are placeholders and will raise at runtime until you update them.

JOINT_INDEX: Dict[str, Optional[int]] = {
    # 上半身
    "left_shoulder": 11,
    "right_shoulder": 14,
    "left_elbow": 12,
    "right_elbow": 15,
    # wrist 上有多个点（13/22 左手，16/24 右手），这里取靠近手腕后部的一个点
    # 用于肘-腕夹角和手相对肩/髋的距离，实际选择 13/16 或 22/24 差异不大。
    "left_wrist": 13,
    "right_wrist": 16,
    # 下半身
    # 0: 下肢中心（骨盆），1: 左腿根，4: 右腿根
    "left_hip": 1,
    "right_hip": 4,
    # 2: 左膝，5: 右膝
    "left_knee": 2,
    "right_knee": 5,
    # 3/17/18 在左脚上，6/19/20 在右脚上；取靠近踝关节的 3、6
    "left_ankle": 3,
    "right_ankle": 6,
}


def _idx(name: str) -> int:
    idx = JOINT_INDEX.get(name)
    if idx is None:
        raise RuntimeError(
            f"JOINT_INDEX['{name}'] is not set. "
            "Please edit compute_fit3d_gt_metrics.py to fill correct indices "
            "for Fit3D joints3d_25 before running."
        )
    return idx


# Map internal exercise names -> Fit3D file names
EXERCISE_TO_FIT3D: Dict[str, str] = {
    "bicep_curl": "dumbbell_biceps_curls",
    "squat": "squat",
    "push_up": "pushup",
    "lateral_raise": "side_lateral_raise",
    "barbell_row": "barbell_row",
}


@dataclass
class MetaRow:
    video: str
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
                # Ignore exercises we don't map.
                continue
            subject = (r.get("subject") or "").strip()
            view = (r.get("view") or "").strip()
            video = (r.get("video") or "").strip()
            if not subject or not view or not video:
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
                    video=video,
                    exercise=exercise,
                    gt_reps=gt_reps,
                    start_frame=sf,
                    end_frame=ef,
                    subject=subject,
                    view=view,
                )
            )
    return rows


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute angle ABC (in degrees) where a,b,c are 3D points; angle at b.
    """
    v1 = a - b
    v2 = c - b
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0.0 or v2_norm == 0.0:
        return float("nan")
    cos_theta = float(np.dot(v1, v2) / (v1_norm * v2_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def summarize(series: List[float]) -> Dict[str, Optional[float]]:
    vals = [x for x in series if np.isfinite(x)]
    if not vals:
        return {"mean": None, "std": None}
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def compute_metrics_for_clip(
    joints_path: Path,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> Dict[str, Optional[float]]:
    with joints_path.open() as f:
        data = json.load(f)

    frames = np.array(data["joints3d_25"], dtype=float)  # [T, 25, 3]
    T = frames.shape[0]

    sf = start_frame or 1
    ef = end_frame or T
    sf = max(1, sf)
    ef = min(T, ef)

    # 1-based -> 0-based slice
    sl = slice(sf - 1, ef)
    clip = frames[sl]  # [T', 25, 3]

    # Pre-fetch indices (will raise if not configured)
    ls = _idx("left_shoulder")
    rs = _idx("right_shoulder")
    le = _idx("left_elbow")
    re = _idx("right_elbow")
    lw = _idx("left_wrist")
    rw = _idx("right_wrist")
    lh = _idx("left_hip")
    rh = _idx("right_hip")
    lk = _idx("left_knee")
    rk = _idx("right_knee")
    la = _idx("left_ankle")
    ra = _idx("right_ankle")

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
    hip_R_series: List[float] = []
    hip_L_series: List[float] = []
    shoulder_R_series: List[float] = []
    shoulder_L_series: List[float] = []

    for frame in clip:
        # 3D points
        p_ls = frame[ls]
        p_rs = frame[rs]
        p_le = frame[le]
        p_re = frame[re]
        p_lw = frame[lw]
        p_rw = frame[rw]
        p_lh = frame[lh]
        p_rh = frame[rh]
        p_lk = frame[lk]
        p_rk = frame[rk]
        p_la = frame[la]
        p_ra = frame[ra]

        # Angles (right & left)
        elbow_R = joint_angle(p_rs, p_re, p_rw)
        elbow_L = joint_angle(p_ls, p_le, p_lw)
        knee_R = joint_angle(p_rh, p_rk, p_ra)
        knee_L = joint_angle(p_lh, p_lk, p_la)
        elbow_R_series.append(elbow_R)
        elbow_L_series.append(elbow_L)
        knee_R_series.append(knee_R)
        knee_L_series.append(knee_L)

        # Hip joint angles: shoulder-hip-knee (right / left)
        hip_R_series.append(joint_angle(p_rs, p_rh, p_rk))
        hip_L_series.append(joint_angle(p_ls, p_lh, p_lk))

        # Shoulder joint angles: hip-shoulder-elbow (right / left)
        shoulder_R_series.append(joint_angle(p_rh, p_rs, p_re))
        shoulder_L_series.append(joint_angle(p_lh, p_ls, p_le))

        # Distances / relative heights (using y-coordinate as vertical)
        shl_dist = float(np.linalg.norm(p_ls - p_rs))
        shl_series.append(shl_dist)

        rshl_rpalm_series.append(float(p_rs[1] - p_rw[1]))
        rshl_rhip_series.append(float(p_rh[1] - p_rs[1]))
        rpalm_rhip_series.append(float(p_rh[1] - p_rw[1]))
        rknee_rhip_series.append(float(p_rh[1] - p_rk[1]))
        rknee_rfeet_series.append(float(p_ra[1] - p_rk[1]))
        rhip_rfeet_series.append(float(p_ra[1] - p_rh[1]))

    # Summaries
    elbow_R = summarize(elbow_R_series)
    elbow_L = summarize(elbow_L_series)
    knee_R = summarize(knee_R_series)
    knee_L = summarize(knee_L_series)
    hip_R = summarize(hip_R_series)
    hip_L = summarize(hip_L_series)
    shoulder_R = summarize(shoulder_R_series)
    shoulder_L = summarize(shoulder_L_series)
    shl = summarize(shl_series)
    rshl_rpalm = summarize(rshl_rpalm_series)
    rshl_rhip = summarize(rshl_rhip_series)
    rpalm_rhip = summarize(rpalm_rhip_series)
    rknee_rhip = summarize(rknee_rhip_series)
    rknee_rfeet = summarize(rknee_rfeet_series)
    rhip_rfeet = summarize(rhip_rfeet_series)

    return {
        "elbow_gt_mean": elbow_R["mean"],
        "elbow_gt_std": elbow_R["std"],
        "left_elbow_gt_mean": elbow_L["mean"],
        "left_elbow_gt_std": elbow_L["std"],
        "knee_gt_mean": knee_R["mean"],
        "knee_gt_std": knee_R["std"],
        "left_knee_gt_mean": knee_L["mean"],
        "left_knee_gt_std": knee_L["std"],
        "hip_gt_mean": hip_R["mean"],
        "hip_gt_std": hip_R["std"],
        "left_hip_gt_mean": hip_L["mean"],
        "left_hip_gt_std": hip_L["std"],
        "shoulder_gt_mean": shoulder_R["mean"],
        "shoulder_gt_std": shoulder_R["std"],
        "left_shoulder_gt_mean": shoulder_L["mean"],
        "left_shoulder_gt_std": shoulder_L["std"],
        "shl_gt_mean": shl["mean"],
        "shl_gt_std": shl["std"],
        "rshl_rpalm_gt_mean": rshl_rpalm["mean"],
        "rshl_rpalm_gt_std": rshl_rpalm["std"],
        "rshl_rhip_gt_mean": rshl_rhip["mean"],
        "rshl_rhip_gt_std": rshl_rhip["std"],
        "rpalm_rhip_gt_mean": rpalm_rhip["mean"],
        "rpalm_rhip_gt_std": rpalm_rhip["std"],
        "rknee_rhip_gt_mean": rknee_rhip["mean"],
        "rknee_rhip_gt_std": rknee_rhip["std"],
        "rknee_rfeet_gt_mean": rknee_rfeet["mean"],
        "rknee_rfeet_gt_std": rknee_rfeet["std"],
        "rhip_rfeet_gt_mean": rhip_rfeet["mean"],
        "rhip_rfeet_gt_std": rhip_rfeet["std"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute GT angle/distance metrics from Fit3D joints3d_25."
    )
    parser.add_argument(
        "--gt-meta",
        type=Path,
        default=Path("fit3d_subset/gt_meta.csv"),
        help="Path to gt_meta.csv generated by gen_fit3d_meta.py.",
    )
    parser.add_argument(
        "--joints-root",
        type=Path,
        default=Path("fit3d_subset/train"),
        help="Root directory containing sXX/joints3d_25/*.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fit3d_subset/fit3d_gt_metrics.csv"),
        help="Output CSV path for GT metrics.",
    )
    args = parser.parse_args()

    meta_rows = load_meta(args.gt_meta)
    if not meta_rows:
        print(f"No usable rows found in {args.gt_meta}")
        return

    rows_out: List[Dict[str, object]] = []

    for m in meta_rows:
        fit3d_name = EXERCISE_TO_FIT3D[m.exercise]
        joints_path = args.joints_root / m.subject / "joints3d_25" / f"{fit3d_name}.json"
        if not joints_path.exists():
            print(f"[WARN] joints file missing: {joints_path}")
            continue

        metrics = compute_metrics_for_clip(joints_path, m.start_frame, m.end_frame)
        row: Dict[str, object] = {
            "subject": m.subject,
            "view": m.view,
            "exercise": m.exercise,
            "gt_reps": m.gt_reps,
        }
        row.update(metrics)
        rows_out.append(row)

    # Write CSV
    fieldnames = [
        "subject",
        "view",
        "exercise",
        "gt_reps",
        "elbow_gt_mean",
        "elbow_gt_std",
        "left_elbow_gt_mean",
        "left_elbow_gt_std",
        "knee_gt_mean",
        "knee_gt_std",
        "left_knee_gt_mean",
        "left_knee_gt_std",
        "hip_gt_mean",
        "hip_gt_std",
        "left_hip_gt_mean",
        "left_hip_gt_std",
        "shoulder_gt_mean",
        "shoulder_gt_std",
        "left_shoulder_gt_mean",
        "left_shoulder_gt_std",
        "shl_gt_mean",
        "shl_gt_std",
        "rshl_rpalm_gt_mean",
        "rshl_rpalm_gt_std",
        "rshl_rhip_gt_mean",
        "rshl_rhip_gt_std",
        "rpalm_rhip_gt_mean",
        "rpalm_rhip_gt_std",
        "rknee_rhip_gt_mean",
        "rknee_rhip_gt_std",
        "rknee_rfeet_gt_mean",
        "rknee_rfeet_gt_std",
        "rhip_rfeet_gt_mean",
        "rhip_rfeet_gt_std",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"[INFO] Wrote {len(rows_out)} GT rows to {args.output}")


if __name__ == "__main__":
    main()
