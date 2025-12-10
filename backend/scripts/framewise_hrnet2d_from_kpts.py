"""
Export framewise HRNet-2D metrics from saved 2D keypoints.

数据来源：
  mmpose/tools/export_fit3d_hrnet2d_kpts.py 生成的：

    hrnet2d_kpts/<exercise>__<subject>__<view>/kpts2d_hrnet.json

  其中包含：
    {
      "exercise": "...",
      "subject": "s03",
      "view": "50591643",
      "video": ".../s03/videos/50591643/squat.mp4",
      "keypoints": [  # T 帧
        [[x0,y0], ..., [x16,y16]],
        ...
      ]
    }

本脚本：
  - 读取这些 2D keypoints；
  - 使用 gt_meta_s03.csv 中的 start_frame/end_frame 做裁剪；
  - 对每一帧计算与 GT 相同的一整套指标：
      * 右/左肘、膝、髋、肩的角度
      * shl, rshl_rpalm, rshl_rhip, rpalm_rhip,
        rknee_rhip, rknee_rfeet, rhip_rfeet
  - 按 framewise_gt.py 的结构输出：

    framewise_results/hrnet2d/<exercise>__<subject>__<view>.json

后续可以用 analyze_framewise_errors.py 做逐帧与 GT 的差值分析。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2


@dataclass
class MetaRow:
    exercise: str
    subject: str
    view: str
    gt_reps: int
    start_frame: Optional[int]
    end_frame: Optional[int]


def load_meta(meta_csv: Path, subject: str, exercises: List[str]) -> Dict[Tuple[str, str, str], MetaRow]:
    """Load gt_meta.csv into dict keyed by (exercise, subject, view)."""
    out: Dict[Tuple[str, str, str], MetaRow] = {}
    with meta_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ex = (r.get("exercise") or "").strip()
            if exercises and ex not in exercises:
                continue
            subj = (r.get("subject") or "").strip()
            if subj != subject:
                continue
            view = (r.get("view") or "").strip()
            if not ex or not subj or not view:
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
            out[(ex, subj, view)] = MetaRow(
                exercise=ex,
                subject=subj,
                view=view,
                gt_reps=gt_reps,
                start_frame=sf,
                end_frame=ef,
            )
    return out


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """2D angle ABC (degrees)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cos_th = float(np.dot(v1, v2) / (n1 * n2))
    cos_th = max(-1.0, min(1.0, cos_th))
    return float(np.degrees(np.arccos(cos_th)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export framewise HRNet2D metrics from 2D keypoints.")
    parser.add_argument(
        "--kpts-root",
        type=Path,
        default=Path("hrnet2d_kpts"),
        help="Root containing <exercise>__<subject>__<view>/kpts2d_hrnet.json.",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=Path("fit3d_subset/gt_meta_s03.csv"),
        help="Meta CSV (gt_meta_s03.csv) for frame ranges.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="s03",
        help="Subject ID (default: s03).",
    )
    parser.add_argument(
        "--exercises",
        nargs="+",
        default=["bicep_curl", "squat", "push_up", "lateral_raise", "barbell_row"],
        help="Exercises to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("framewise_results/hrnet2d"),
        help="Directory to write per-video HRNet2D JSON files.",
    )
    args = parser.parse_args()

    meta_map = load_meta(args.meta_csv, args.subject, args.exercises)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for subdir in sorted(args.kpts_root.glob("*__*__*")):
        if not subdir.is_dir():
            continue
        kpts_path = subdir / "kpts2d_hrnet.json"
        if not kpts_path.exists():
            continue
        with kpts_path.open() as f:
            data = json.load(f)

        ex = (data.get("exercise") or "").strip()
        subj = (data.get("subject") or "").strip()
        view = (data.get("view") or "").strip()
        if ex not in args.exercises or subj != args.subject:
            continue

        key = (ex, subj, view)
        meta = meta_map.get(key)
        if meta is None:
            print(f"[WARN] No meta row for {key}")
            continue

        kpts = np.asarray(data.get("keypoints", []), dtype=float)  # [T,17,2]
        if kpts.ndim != 3 or kpts.shape[1] != 17:
            print(f"[WARN] Unexpected keypoints shape in {kpts_path}: {kpts.shape}")
            continue

        # Normalize coordinates by video width/height so distance-like signals are comparable
        vid_path = data.get("video")
        width = height = None
        if vid_path and Path(vid_path).exists():
            cap = cv2.VideoCapture(str(vid_path))
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
        if width and height and width > 0 and height > 0:
            kpts = kpts / np.array([[width, height]])
        else:
            print(f"[WARN] Video size unknown for {vid_path}, keeping raw pixel distances.")

        T = kpts.shape[0]
        sf = meta.start_frame or 1
        ef = meta.end_frame or T
        sf = max(1, sf)
        ef = min(T, ef)

        # COCO indices
        NOSE = 0
        L_SH, R_SH = 5, 6
        L_EL, R_EL = 7, 8
        L_WR, R_WR = 9, 10
        L_HP, R_HP = 11, 12
        L_KN, R_KN = 13, 14
        L_AN, R_AN = 15, 16

        frames_out: List[Dict[str, Any]] = []
        for idx in range(sf - 1, ef):
            p = kpts[idx]
            nose = p[NOSE]
            l_sh, r_sh = p[L_SH], p[R_SH]
            l_el, r_el = p[L_EL], p[R_EL]
            l_wr, r_wr = p[L_WR], p[R_WR]
            l_hp, r_hp = p[L_HP], p[R_HP]
            l_kn, r_kn = p[L_KN], p[R_KN]
            l_an, r_an = p[L_AN], p[R_AN]

            elbow_R = joint_angle(r_sh, r_el, r_wr)
            elbow_L = joint_angle(l_sh, l_el, l_wr)
            knee_R = joint_angle(r_hp, r_kn, r_an)
            knee_L = joint_angle(l_hp, l_kn, l_an)

            hip_R = joint_angle(r_sh, r_hp, r_kn)
            hip_L = joint_angle(l_sh, l_hp, l_kn)

            shoulder_R = joint_angle(r_hp, r_sh, r_el)
            shoulder_L = joint_angle(l_hp, l_sh, l_el)

            shl = float(np.linalg.norm(l_sh - r_sh))
            rshl_rpalm = float(r_sh[1] - r_wr[1])
            rshl_rhip = float(r_hp[1] - r_sh[1])
            rpalm_rhip = float(r_hp[1] - r_wr[1])
            rknee_rhip = float(r_hp[1] - r_kn[1])
            rknee_rfeet = float(r_an[1] - r_kn[1])
            rhip_rfeet = float(r_an[1] - r_hp[1])

            rec: Dict[str, Any] = {
                "frame_index": idx + 1,
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
            frames_out.append(rec)

        out = {
            "subject": meta.subject,
            "exercise": meta.exercise,
            "view": meta.view,
            "backend": "hrnet2d",
            "gt_reps": meta.gt_reps,
            "start_frame": sf,
            "end_frame": ef,
            "frames": frames_out,
        }

        out_name = f"{meta.exercise}__{meta.subject}__{meta.view}.json"
        out_path = args.output_dir / out_name
        with out_path.open("w") as f:
            json.dump(out, f)
        print(f"[INFO] wrote HRNet2D framewise -> {out_path}")


if __name__ == "__main__":
    main()
