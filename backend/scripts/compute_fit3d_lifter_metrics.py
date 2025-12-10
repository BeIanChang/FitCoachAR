"""
Compute angle/distance error metrics for 3D lifter models on Fit3D clips.

输入：
  - Fit3D GT joints3d_25 及 gt_meta.csv（与 compute_fit3d_gt_metrics 相同）
  - mmpose/tools/eval_fit3d_with_lifters.py 生成的预测结果：
      fit3d_lifter_preds/<backend>/<exercise>__<subject>__<view>/predictions/<exercise>.json
    以及 per-clip 延迟信息：
      fit3d_lifter_preds/<backend>/<exercise>__<subject>__<view>/metrics.json

输出：
  - CSV 文件，例如：
      fit3d_results/fit3d_lifter_model_vs_gt.csv
    每行包含：
      exercise, subject, view, backend,
      per-metric mean_abs_err / std_abs_err / n,
      frames, latency_ms, fps_eq
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
            "Please edit compute_fit3d_gt_metrics.py to fill correct indices "
            "for Fit3D joints3d_25 before running lifter metrics."
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
    return frames[sf - 1 : ef]  # [T',25,3]


def load_lifter_pred(
    preds_root: Path,
    backend: str,
    exercise: str,
    subject: str,
    view: str,
) -> np.ndarray:
    """Load lifter 3D joints (H36M-17 style) from predictions JSON."""
    clip_dir = preds_root / backend / f"{exercise}__{subject}__{view}" / "predictions"
    if not clip_dir.exists():
        raise FileNotFoundError(f"Predictions directory missing: {clip_dir}")

    # 默认文件名为 <exercise>.json，例如 squat.json / barbell_row.json
    pred_file = clip_dir / f"{exercise}.json"
    if not pred_file.exists():
        # 回退：目录下找第一个 .json
        json_files = list(clip_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No prediction JSON found in {clip_dir}")
        pred_file = json_files[0]

    with pred_file.open() as f:
        data = json.load(f)

    # mmpose 的输出是一个列表，每个元素对应一帧：
    # [{ "frame_id": int, "instances": [ { "keypoints": [ [x,y,z], ... ] } ] }, ...]
    frames_3d: List[np.ndarray] = []
    for fr in data:
        insts = fr.get("instances") or []
        if not insts:
            continue
        kpts = np.asarray(insts[0].get("keypoints"), dtype=float)  # [J,3]
        frames_3d.append(kpts)

    if not frames_3d:
        raise RuntimeError(f"No 3D keypoints parsed from {pred_file}")

    return np.stack(frames_3d, axis=0)  # [T, J, 3]


def compute_errors_for_clip(gt: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    """
    Compute per-metric absolute errors between GT and prediction.

    gt   : [Tg, 25, 3] joints3d_25
    pred : [Tp, J, 3]  (H36M-17 style, but我们只用需要的关节)
    """
    T = min(gt.shape[0], pred.shape[0])
    if T <= 0:
        raise RuntimeError("No overlapping frames between GT and prediction.")

    gt_clip = gt[:T]
    pred_clip = pred[:T]

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

    # 对于 lifter 的 skeleton，我们尽量复用相同的解剖含义：这里假设
    # H36M-17 顺序（0:hip,1:r_hip,2:r_knee,3:r_ankle,4:l_hip,5:l_knee,6:l_ankle,
    # 7:spine,8:thorax,9:neck,10:head,11:l_sh,12:l_el,13:l_wr,14:r_sh,15:r_el,16:r_wr）
    h36m_ls, h36m_rs = 11, 14
    h36m_le, h36m_re = 12, 15
    h36m_lw, h36m_rw = 13, 16
    h36m_lh, h36m_rh = 4, 1
    h36m_lk, h36m_rk = 5, 2
    h36m_la, h36m_ra = 6, 3

    elbow_R_err: List[float] = []
    elbow_L_err: List[float] = []
    knee_R_err: List[float] = []
    knee_L_err: List[float] = []
    shl_err: List[float] = []
    rshl_rpalm_err: List[float] = []
    rknee_rhip_err: List[float] = []
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

        # Pred joints (假设 H36M-17)
        p_ls = p3d[h36m_ls]
        p_rs = p3d[h36m_rs]
        p_le = p3d[h36m_le]
        p_re = p3d[h36m_re]
        p_lw = p3d[h36m_lw]
        p_rw = p3d[h36m_rw]
        p_lh = p3d[h36m_lh]
        p_rh = p3d[h36m_rh]
        p_lk = p3d[h36m_lk]
        p_rk = p3d[h36m_rk]
        p_la = p3d[h36m_la]
        p_ra = p3d[h36m_ra]

        # Angles
        g_elbow_R = joint_angle(g_rs, g_re, g_rw)
        g_elbow_L = joint_angle(g_ls, g_le, g_lw)
        g_knee_R = joint_angle(g_rh, g_rk, g_ra)
        g_knee_L = joint_angle(g_lh, g_lk, g_la)

        p_elbow_R = joint_angle(p_rs, p_re, p_rw)
        p_elbow_L = joint_angle(p_ls, p_le, p_lw)
        p_knee_R = joint_angle(p_rh, p_rk, p_ra)
        p_knee_L = joint_angle(p_lh, p_lk, p_la)

        # Distances (use y as vertical)
        g_shl = float(np.linalg.norm(g_ls - g_rs))
        p_shl = float(np.linalg.norm(p_ls - p_rs))

        g_rshl_rpalm = float(g_rs[1] - g_rw[1])
        p_rshl_rpalm = float(p_rs[1] - p_rw[1])

        g_rknee_rhip = float(g_rh[1] - g_rk[1])
        p_rknee_rhip = float(p_rh[1] - p_rk[1])

        g_rhip_rfeet = float(g_ra[1] - g_rh[1])
        p_rhip_rfeet = float(p_ra[1] - p_rh[1])

        if np.isfinite(g_elbow_R) and np.isfinite(p_elbow_R):
            elbow_R_err.append(abs(p_elbow_R - g_elbow_R))
        if np.isfinite(g_elbow_L) and np.isfinite(p_elbow_L):
            elbow_L_err.append(abs(p_elbow_L - g_elbow_L))
        if np.isfinite(g_knee_R) and np.isfinite(p_knee_R):
            knee_R_err.append(abs(p_knee_R - g_knee_R))
        if np.isfinite(g_knee_L) and np.isfinite(p_knee_L):
            knee_L_err.append(abs(p_knee_L - g_knee_L))

        shl_err.append(abs(p_shl - g_shl))
        rshl_rpalm_err.append(abs(p_rshl_rpalm - g_rshl_rpalm))
        rknee_rhip_err.append(abs(p_rknee_rhip - g_rknee_rhip))
        rhip_rfeet_err.append(abs(p_rhip_rfeet - g_rhip_rfeet))

    def summarize_err(xs: List[float]) -> Dict[str, object]:
        if not xs:
            return {"mean": None, "std": None, "n": 0}
        arr = np.asarray(xs, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": int(arr.size),
        }

    return {
        "elbow_R": summarize_err(elbow_R_err),
        "elbow_L": summarize_err(elbow_L_err),
        "knee_R": summarize_err(knee_R_err),
        "knee_L": summarize_err(knee_L_err),
        "shl": summarize_err(shl_err),
        "rshl_rpalm": summarize_err(rshl_rpalm_err),
        "rknee_rhip": summarize_err(rknee_rhip_err),
        "rhip_rfeet": summarize_err(rhip_rfeet_err),
        "frames_used": int(T),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Fit3D angle/distance errors for 3D lifter models."
    )
    parser.add_argument(
        "--gt-meta",
        type=Path,
        default=Path("fit3d_subset/gt_meta_s03.csv"),
        help="Meta CSV for Fit3D clips (subject/exercise/view/start/end).",
    )
    parser.add_argument(
        "--joints-root",
        type=Path,
        default=Path("fit3d_subset/train"),
        help="Root containing sXX/joints3d_25/*.json.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("fit3d_lifter_preds"),
        help="Root containing lifter predictions (motionbert/videopose*/...).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["motionbert", "videopose243", "videopose1"],
        help="Lifter backend names (subdirectories under pred-root).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="s03",
        help="Subject ID to filter (e.g., s03).",
    )
    parser.add_argument(
        "--exercises",
        nargs="+",
        default=["squat", "barbell_row"],
        help="Exercises to include.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("fit3d_results/fit3d_lifter_model_vs_gt.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    meta_rows = load_meta(args.gt_meta)
    if not meta_rows:
        print(f"[WARN] No rows in {args.gt_meta}")
        return

    rows_out: List[Dict[str, object]] = []

    for m in meta_rows:
        if m.subject != args.subject:
            continue
        if m.exercise not in args.exercises:
            continue
        try:
            gt_clip = load_gt_clip(
                args.joints_root,
                m.subject,
                m.exercise,
                m.view,
                m.start_frame,
                m.end_frame,
            )
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue

        for backend in args.backends:
            try:
                pred = load_lifter_pred(
                    args.pred_root, backend, m.exercise, m.subject, m.view
                )
            except Exception as exc:
                print(f"[WARN] skip {backend} {m.exercise} {m.view}: {exc}")
                continue

            try:
                errs = compute_errors_for_clip(gt_clip, pred)
            except Exception as exc:
                print(f"[WARN] error computing metrics for {backend} {m.exercise} {m.view}: {exc}")
                continue

            # 读取延迟信息
            metrics_path = (
                args.pred_root
                / backend
                / f"{m.exercise}__{m.subject}__{m.view}"
                / "metrics.json"
            )
            latency_ms = None
            fps_eq = None
            if metrics_path.exists():
                try:
                    with metrics_path.open() as f:
                        mjson = json.load(f)
                    latency_ms = mjson.get("latency_ms")
                    fps_eq = mjson.get("fps_eq")
                except Exception:
                    latency_ms = None
                    fps_eq = None

            row: Dict[str, object] = {
                "exercise": m.exercise,
                "subject": m.subject,
                "view": m.view,
                "backend": backend,
                "frames_used": errs["frames_used"],
                "latency_ms": latency_ms,
                "fps_eq": fps_eq,
            }

            for key in ["elbow_R", "elbow_L", "knee_R", "knee_L", "shl", "rshl_rpalm", "rknee_rhip", "rhip_rfeet"]:
                stats = errs[key]
                row[f"{key}_mean_abs_err"] = stats["mean"]
                row[f"{key}_std_abs_err"] = stats["std"]
                row[f"{key}_n"] = stats["n"]

            rows_out.append(row)

    if not rows_out:
        print("[WARN] No lifter metrics computed.")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    # 构造字段顺序
    base_fields = [
        "exercise",
        "subject",
        "view",
        "backend",
        "frames_used",
        "latency_ms",
        "fps_eq",
    ]
    metric_fields: List[str] = []
    for key in ["elbow_R", "elbow_L", "knee_R", "knee_L", "shl", "rshl_rpalm", "rknee_rhip", "rhip_rfeet"]:
        metric_fields.extend(
            [f"{key}_mean_abs_err", f"{key}_std_abs_err", f"{key}_n"]
        )

    fieldnames = base_fields + metric_fields

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"[INFO] Wrote {len(rows_out)} lifter rows to {args.output_csv}")


if __name__ == "__main__":
    main()

