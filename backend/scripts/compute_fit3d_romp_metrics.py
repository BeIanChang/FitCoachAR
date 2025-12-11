"""
Compute angle/distance error metrics for ROMP 3D joints on Fit3D clips.

思路和 `compute_fit3d_lifter_metrics.py` 相同，只是预测来自 ROMP 的
`fit3d_romp_preds/<exercise>__<subject>__<view>/romp_joints3d.json`，
每帧有 71 个关节点：

  - 0–23:   SMPL_24
  - 24–53:  SMPL_EXTRA_30
  - 54–70:  H36M_17

为了一致性，这里只用 SMPL_24 里解剖含义明确的关节点：

  - 肩：  L_Shoulder (16),  R_Shoulder (17)
  - 肘：  L_Elbow (18),     R_Elbow (19)
  - 腕：  L_Wrist (20),     R_Wrist (21)
  - 髋：  L_Hip_SMPL (1),   R_Hip_SMPL (2)
  - 膝：  L_Knee (4),       R_Knee (5)
  - 踝：  L_Ankle (7),      R_Ankle (8)

GT 仍然使用 Fit3D joints3d_25 和 compute_fit3d_gt_metrics 里定义的
JOINT_INDEX 映射。

输出 CSV 结构类似 lifter 版本：

  fit3d_results/fit3d_romp_model_vs_gt.csv

每行：
  exercise, subject, view, backend=romp3d,
  frames_used, latency_ms, fps_eq,
  elbow_R_mean_abs_err, elbow_R_std_abs_err, elbow_R_n, ...
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.scripts.compute_fit3d_gt_metrics import (  # type: ignore
    EXERCISE_TO_FIT3D,
    JOINT_INDEX,
    joint_angle,
    load_meta,
)


def _idx(name: str) -> int:
    idx = JOINT_INDEX.get(name)
    if idx is None:
        raise RuntimeError(
            f"JOINT_INDEX['{name}'] is not set. "
            "Please edit compute_fit3d_gt_metrics.py before running ROMP metrics."
        )
    return idx


def load_gt_clip(
    joints_root: Path,
    subject: str,
    exercise: str,
    view: str,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> np.ndarray:
    """Load GT 3D joints for a clip and slice to [sf,ef]."""
    fit3d_name = EXERCISE_TO_FIT3D[exercise]
    joints_path = joints_root / subject / "joints3d_25" / f"{fit3d_name}.json"
    if not joints_path.exists():
        raise FileNotFoundError(f"GT joints file missing: {joints_path}")
    with joints_path.open() as f:
        data = json.load(f)
    frames = np.asarray(data["joints3d_25"], dtype=float)  # [T,25,3]
    T = frames.shape[0]
    sf = start_frame or 1
    ef = end_frame or T
    sf = max(1, sf)
    ef = min(T, ef)
    return frames[sf - 1 : ef]


def load_romp_pred(
    preds_root: Path,
    exercise: str,
    subject: str,
    view: str,
) -> np.ndarray:
    """Load ROMP 3D joints [T,71,3] from romp_joints3d.json."""
    clip_dir = preds_root / f"{exercise}__{subject}__{view}"
    pred_file = clip_dir / "romp_joints3d.json"
    if not pred_file.exists():
        raise FileNotFoundError(f"ROMP joints file missing: {pred_file}")
    with pred_file.open() as f:
        data = json.load(f)
    arr = np.asarray(data["joints3d"], dtype=float)  # [T,71,3]
    if arr.ndim != 3 or arr.shape[1] != 71:
        raise RuntimeError(f"Unexpected ROMP joints shape: {arr.shape}")
    return arr


def compute_errors_for_clip(gt: np.ndarray, pred71: np.ndarray) -> Dict[str, object]:
    """
    Compute per-metric absolute errors between GT and ROMP prediction.

    gt     : [Tg,25,3]  Fit3D joints3d_25
    pred71 : [Tp,71,3]  ROMP joints (SMPL_24 + EXTRA_30 + H36M_17)
    """
    T = min(gt.shape[0], pred71.shape[0])
    if T <= 0:
        raise RuntimeError("No overlapping frames between GT and prediction.")

    gt_clip = gt[:T]
    pred_clip = pred71[:T]

    # GT indices
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

    # ROMP SMPL_24 indices (0–23) we care about:
    # 1: L_Hip_SMPL, 2: R_Hip_SMPL, 4: L_Knee, 5: R_Knee,
    # 7: L_Ankle, 8: R_Ankle,
    # 16: L_Shoulder, 17: R_Shoulder,
    # 18: L_Elbow, 19: R_Elbow,
    # 20: L_Wrist, 21: R_Wrist
    r_ls, r_rs = 16, 17
    r_le, r_re = 18, 19
    r_lw, r_rw = 20, 21
    r_lh, r_rh = 1, 2
    r_lk, r_rk = 4, 5
    r_la, r_ra = 7, 8

    elbow_R_err: List[float] = []
    elbow_L_err: List[float] = []
    knee_R_err: List[float] = []
    knee_L_err: List[float] = []
    hip_R_err: List[float] = []
    hip_L_err: List[float] = []
    shoulder_R_err: List[float] = []
    shoulder_L_err: List[float] = []
    shl_err: List[float] = []
    rshl_rpalm_err: List[float] = []
    rshl_rhip_err: List[float] = []
    rpalm_rhip_err: List[float] = []
    rknee_rhip_err: List[float] = []
    rknee_rfeet_err: List[float] = []
    rhip_rfeet_err: List[float] = []

    for g3d, p3d in zip(gt_clip, pred_clip):
        # GT joints
        g_ls = g3d[ls]
        g_rs = g3d[rs]
        g_le = g3d[le]
        g_re = g3d[re]
        g_lw = g3d[lw]
        g_rw = g3d[rw]
        g_lh = g3d[lh]
        g_rh = g3d[rh]
        g_lk = g3d[lk]
        g_rk = g3d[rk]
        g_la = g3d[la]
        g_ra = g3d[ra]

        # ROMP joints
        p_ls = p3d[r_ls]
        p_rs = p3d[r_rs]
        p_le = p3d[r_le]
        p_re = p3d[r_re]
        p_lw = p3d[r_lw]
        p_rw = p3d[r_rw]
        p_lh = p3d[r_lh]
        p_rh = p3d[r_rh]
        p_lk = p3d[r_lk]
        p_rk = p3d[r_rk]
        p_la = p3d[r_la]
        p_ra = p3d[r_ra]

        # Angles
        g_elbow_R = joint_angle(g_rs, g_re, g_rw)
        g_elbow_L = joint_angle(g_ls, g_le, g_lw)
        g_knee_R = joint_angle(g_rh, g_rk, g_ra)
        g_knee_L = joint_angle(g_lh, g_lk, g_la)
        g_hip_R = joint_angle(g_rs, g_rh, g_rk)
        g_hip_L = joint_angle(g_ls, g_lh, g_lk)
        g_shoulder_R = joint_angle(g_rh, g_rs, g_re)
        g_shoulder_L = joint_angle(g_lh, g_ls, g_le)

        p_elbow_R = joint_angle(p_rs, p_re, p_rw)
        p_elbow_L = joint_angle(p_ls, p_le, p_lw)
        p_knee_R = joint_angle(p_rh, p_rk, p_ra)
        p_knee_L = joint_angle(p_lh, p_lk, p_la)
        p_hip_R = joint_angle(p_rs, p_rh, p_rk)
        p_hip_L = joint_angle(p_ls, p_lh, p_lk)
        p_shoulder_R = joint_angle(p_rh, p_rs, p_re)
        p_shoulder_L = joint_angle(p_lh, p_ls, p_le)

        # Distances (use y as vertical，如同 lifter 版本)
        g_shl = float(np.linalg.norm(g_ls - g_rs))
        p_shl = float(np.linalg.norm(p_ls - p_rs))

        g_rshl_rpalm = float(g_rs[1] - g_rw[1])
        p_rshl_rpalm = float(p_rs[1] - p_rw[1])

        g_rshl_rhip = float(g_rh[1] - g_rs[1])
        p_rshl_rhip = float(p_rh[1] - p_rs[1])

        g_rpalm_rhip = float(g_rh[1] - g_rw[1])
        p_rpalm_rhip = float(p_rh[1] - p_rw[1])

        g_rknee_rhip = float(g_rh[1] - g_rk[1])
        p_rknee_rhip = float(p_rh[1] - p_rk[1])

        g_rknee_rfeet = float(g_ra[1] - g_rk[1])
        p_rknee_rfeet = float(p_ra[1] - p_rk[1])

        g_rhip_rfeet = float(g_ra[1] - g_rh[1])
        p_rhip_rfeet = float(p_ra[1] - p_rh[1])

        def acc(lst: List[float], gv: float, pv: float) -> None:
            if np.isfinite(gv) and np.isfinite(pv):
                lst.append(abs(pv - gv))

        acc(elbow_R_err, g_elbow_R, p_elbow_R)
        acc(elbow_L_err, g_elbow_L, p_elbow_L)
        acc(knee_R_err, g_knee_R, p_knee_R)
        acc(knee_L_err, g_knee_L, p_knee_L)
        acc(hip_R_err, g_hip_R, p_hip_R)
        acc(hip_L_err, g_hip_L, p_hip_L)
        acc(shoulder_R_err, g_shoulder_R, p_shoulder_R)
        acc(shoulder_L_err, g_shoulder_L, p_shoulder_L)
        acc(shl_err, g_shl, p_shl)
        acc(rshl_rpalm_err, g_rshl_rpalm, p_rshl_rpalm)
        acc(rshl_rhip_err, g_rshl_rhip, p_rshl_rhip)
        acc(rpalm_rhip_err, g_rpalm_rhip, p_rpalm_rhip)
        acc(rknee_rhip_err, g_rknee_rhip, p_rknee_rhip)
        acc(rknee_rfeet_err, g_rknee_rfeet, p_rknee_rfeet)
        acc(rhip_rfeet_err, g_rhip_rfeet, p_rhip_rfeet)

    def summarize(xs: List[float]) -> (float, float, int):
        if not xs:
            return float("nan"), float("nan"), 0
        arr = np.asarray(xs, dtype=float)
        return float(arr.mean()), float(arr.std()), int(arr.size)

    out: Dict[str, object] = {}
    for name, vals in [
        ("elbow_R", elbow_R_err),
        ("elbow_L", elbow_L_err),
        ("knee_R", knee_R_err),
        ("knee_L", knee_L_err),
        ("hip_R", hip_R_err),
        ("hip_L", hip_L_err),
        ("shoulder_R", shoulder_R_err),
        ("shoulder_L", shoulder_L_err),
        ("shl", shl_err),
        ("rshl_rpalm", rshl_rpalm_err),
        ("rshl_rhip", rshl_rhip_err),
        ("rpalm_rhip", rpalm_rhip_err),
        ("rknee_rhip", rknee_rhip_err),
        ("rknee_rfeet", rknee_rfeet_err),
        ("rhip_rfeet", rhip_rfeet_err),
    ]:
        m, s, n = summarize(vals)
        out[f"{name}_mean_abs_err"] = m
        out[f"{name}_std_abs_err"] = s
        out[f"{name}_n"] = n

    out["frames_used"] = int(T)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Fit3D angle/distance errors for ROMP 3D joints."
    )
    parser.add_argument(
        "--gt-meta",
        type=Path,
        default=Path("fit3d_subset/gt_meta_s03.csv"),
        help="Meta CSV with exercise/subject/view/start_frame/end_frame.",
    )
    parser.add_argument(
        "--joints-root",
        type=Path,
        default=Path("fit3d_subset/train"),
        help="Root of Fit3D train data (contains s03/joints3d_25).",
    )
    parser.add_argument(
        "--preds-root",
        type=Path,
        default=Path("fit3d_romp_preds"),
        help="Root containing ROMP predictions (romp_joints3d.json).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="s03",
        help="Subject id (e.g., s03).",
    )
    parser.add_argument(
        "--exercises",
        nargs="+",
        default=["bicep_curl", "squat", "push_up", "lateral_raise", "barbell_row"],
        help="Exercises to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fit3d_results/fit3d_romp_model_vs_gt.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    meta_rows = load_meta(args.gt_meta)

    rows_out: List[Dict[str, object]] = []

    for meta in meta_rows:
        exercise = meta.exercise
        subject = meta.subject
        view = meta.view
        if subject != args.subject or exercise not in args.exercises:
            continue
        try:
            gt_clip = load_gt_clip(
                joints_root=args.joints_root,
                subject=subject,
                exercise=exercise,
                view=view,
                start_frame=meta.start_frame,
                end_frame=meta.end_frame,
            )
            pred = load_romp_pred(
                preds_root=args.preds_root,
                exercise=exercise,
                subject=subject,
                view=view,
            )
        except FileNotFoundError:
            continue

        metrics = compute_errors_for_clip(gt_clip, pred)
        row: Dict[str, object] = {
            "exercise": exercise,
            "subject": subject,
            "view": view,
            "backend": "romp3d",
        }
        row.update(metrics)

        # 附加 ROMP 延迟（如果存在）
        metrics_path = (
            args.preds_root / f"{exercise}__{subject}__{view}" / "metrics.json"
        )
        if metrics_path.exists():
            with metrics_path.open() as f:
                m = json.load(f)
            row["latency_ms"] = m.get("latency_ms_total")
            row["fps_eq"] = (
                1000.0 / m.get("latency_ms_per_frame")
                if m.get("latency_ms_per_frame")
                else None
            )

        rows_out.append(row)

    if not rows_out:
        print("[WARN] No ROMP metrics computed.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[INFO] Wrote {len(rows_out)} ROMP rows to {args.output}")


if __name__ == "__main__":
    main()
