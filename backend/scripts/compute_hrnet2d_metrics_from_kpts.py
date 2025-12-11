"""
Compute HRNet-2D based metrics from previously exported 2D keypoints.

Background
----------
`mmpose/tools/export_fit3d_hrnet2d_kpts.py` writes, for each clip:

  hrnet2d_kpts/<exercise>__<subject>__<view>/
      kpts2d_hrnet.json
      metrics.json

where `kpts2d_hrnet.json` contains:

  {
    "exercise": "...",
    "subject": "s03",
    "view": "50591643",
    "video": ".../s03/videos/50591643/squat.mp4",
    "keypoints": [  // length T
      [[x0,y0], ..., [x16,y16]],
      ...
    ]
  }

This script:
  - reads those 2D sequences,
  - optionally crops frames using Fit3D gt_meta.csv (start_frame/end_frame),
  - computes the same angle / distance metrics as we do for GT:
      * right/left elbow & knee angles
      * right/left hip angles (shoulder-hip-knee)
      * right/left shoulder angles (hip-shoulder-elbow)
      * shl_dist (shoulder distance)
      * rshl_rpalm, rshl_rhip, rpalm_rhip, rknee_rhip, rknee_rfeet, rhip_rfeet
  - writes a CSV similar to offline_eval_fit3d_hrnet2d.csv.

It does *not* run HRNet again; it only operates on stored 2D keypoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


@dataclass
class MetaRow:
    exercise: str
    subject: str
    view: str
    start_frame: Optional[int]
    end_frame: Optional[int]


def load_meta(meta_csv: Path) -> Dict[Tuple[str, str, str], MetaRow]:
    """Load gt_meta.csv into a dict keyed by (exercise, subject, view)."""
    out: Dict[Tuple[str, str, str], MetaRow] = {}
    with meta_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ex = (r.get("exercise") or "").strip()
            subj = (r.get("subject") or "").strip()
            view = (r.get("view") or "").strip()
            if not ex or not subj or not view:
                continue
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
                start_frame=sf,
                end_frame=ef,
            )
    return out


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """2D version: angle ABC in degrees."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def summarize(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [x for x in xs if np.isfinite(x)]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std())


def compute_metrics_from_kpts(
    kpts: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compute angle / distance metrics from [T,17,2] COCO keypoints."""
    # COCO-17 indices
    NOSE = 0
    L_SH, R_SH = 5, 6
    L_EL, R_EL = 7, 8
    L_WR, R_WR = 9, 10
    L_HP, R_HP = 11, 12
    L_KN, R_KN = 13, 14
    L_AN, R_AN = 15, 16

    elbow_R, elbow_L = [], []
    knee_R, knee_L = [], []
    hip_R, hip_L = [], []
    shoulder_R, shoulder_L = [], []
    shl = []
    rshl_rpalm, rshl_rhip = [], []
    rpalm_rhip = []
    rknee_rhip, rknee_rfeet, rhip_rfeet = [], [], []

    for p in kpts:  # [17,2]
        nose = p[NOSE]
        l_sh, r_sh = p[L_SH], p[R_SH]
        l_el, r_el = p[L_EL], p[R_EL]
        l_wr, r_wr = p[L_WR], p[R_WR]
        l_hp, r_hp = p[L_HP], p[R_HP]
        l_kn, r_kn = p[L_KN], p[R_KN]
        l_an, r_an = p[L_AN], p[R_AN]

        # elbow & knee angles
        elbow_R.append(joint_angle(r_sh, r_el, r_wr))
        elbow_L.append(joint_angle(l_sh, l_el, l_wr))
        knee_R.append(joint_angle(r_hp, r_kn, r_an))
        knee_L.append(joint_angle(l_hp, l_kn, l_an))

        # hip angles: shoulder-hip-knee
        hip_R.append(joint_angle(r_sh, r_hp, r_kn))
        hip_L.append(joint_angle(l_sh, l_hp, l_kn))

        # shoulder angles: hip-shoulder-elbow
        shoulder_R.append(joint_angle(r_hp, r_sh, r_el))
        shoulder_L.append(joint_angle(l_hp, l_sh, l_el))

        # distances / vertical differences
        shl.append(float(np.linalg.norm(l_sh - r_sh)))

        rshl_rpalm.append(float(r_sh[1] - r_wr[1]))
        rshl_rhip.append(float(r_hp[1] - r_sh[1]))
        rpalm_rhip.append(float(r_hp[1] - r_wr[1]))
        rknee_rhip.append(float(r_hp[1] - r_kn[1]))
        rknee_rfeet.append(float(r_an[1] - r_kn[1]))
        rhip_rfeet.append(float(r_an[1] - r_hp[1]))

    el_m, el_s = summarize(elbow_R)
    ell_m, ell_s = summarize(elbow_L)
    kn_m, kn_s = summarize(knee_R)
    knl_m, knl_s = summarize(knee_L)
    hip_m, hip_s = summarize(hip_R)
    hipL_m, hipL_s = summarize(hip_L)
    sh_m, sh_s = summarize(shoulder_R)
    shL_m, shL_s = summarize(shoulder_L)
    shl_m, shl_s = summarize(shl)
    rshl_rpalm_m, rshl_rpalm_s = summarize(rshl_rpalm)
    rshl_rhip_m, rshl_rhip_s = summarize(rshl_rhip)
    rpalm_rhip_m, rpalm_rhip_s = summarize(rpalm_rhip)
    rknee_rhip_m, rknee_rhip_s = summarize(rknee_rhip)
    rknee_rfeet_m, rknee_rfeet_s = summarize(rknee_rfeet)
    rhip_rfeet_m, rhip_rfeet_s = summarize(rhip_rfeet)

    return {
        "elbow_mean": el_m,
        "elbow_std": el_s,
        "left_elbow_mean": ell_m,
        "left_elbow_std": ell_s,
        "knee_mean": kn_m,
        "knee_std": kn_s,
        "left_knee_mean": knl_m,
        "left_knee_std": knl_s,
        "hip_mean": hip_m,
        "hip_std": hip_s,
        "left_hip_mean": hipL_m,
        "left_hip_std": hipL_s,
        "shoulder_mean": sh_m,
        "shoulder_std": sh_s,
        "left_shoulder_mean": shL_m,
        "left_shoulder_std": shL_s,
        "shl_mean": shl_m,
        "shl_std": shl_s,
        "rshl_rpalm_mean": rshl_rpalm_m,
        "rshl_rpalm_std": rshl_rpalm_s,
        "rshl_rhip_mean": rshl_rhip_m,
        "rshl_rhip_std": rshl_rhip_s,
        "rpalm_rhip_mean": rpalm_rhip_m,
        "rpalm_rhip_std": rpalm_rhip_s,
        "rknee_rhip_mean": rknee_rhip_m,
        "rknee_rhip_std": rknee_rhip_s,
        "rknee_rfeet_mean": rknee_rfeet_m,
        "rknee_rfeet_std": rknee_rfeet_s,
        "rhip_rfeet_mean": rhip_rfeet_m,
        "rhip_rfeet_std": rhip_rfeet_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute HRNet2D metrics from saved 2D keypoints."
    )
    parser.add_argument(
        "--kpts-root",
        type=Path,
        default=Path("hrnet2d_kpts"),
        help="Root directory with <exercise>__<subject>__<view>/kpts2d_hrnet.json.",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=Path("fit3d_subset/gt_meta_s03.csv"),
        help="Meta CSV (e.g., gt_meta_s03.csv) with start_frame/end_frame.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fit3d_results/offline_eval_fit3d_hrnet2d_from_kpts.csv"),
        help="Output CSV path for HRNet2D metrics.",
    )
    args = parser.parse_args()

    meta_map = load_meta(args.meta_csv)

    rows_out: List[Dict[str, object]] = []

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
        video = (data.get("video") or "").strip()
        keypoints = np.asarray(data.get("keypoints", []), dtype=float)  # [T,17,2]
        if keypoints.ndim != 3 or keypoints.shape[1] != 17:
            print(f"[WARN] Unexpected keypoints shape in {kpts_path}: {keypoints.shape}")
            continue

        # Normalize to [0,1] by video width/height so distances are comparable
        width = height = None
        if video and Path(video).exists():
            cap = cv2.VideoCapture(video)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
        if width and height and width > 0 and height > 0:
            keypoints = keypoints / np.array([[width, height]])
        else:
            print(f"[WARN] Video size unknown for {video}, keeping raw pixel distances.")

        meta = meta_map.get((ex, subj, view))
        if meta is not None and meta.start_frame and meta.end_frame:
            sf = max(1, meta.start_frame)
            ef = min(keypoints.shape[0], meta.end_frame)
            clip = keypoints[sf - 1 : ef]
        else:
            clip = keypoints

        metrics = compute_metrics_from_kpts(clip)
        frames = clip.shape[0]

        row: Dict[str, object] = {
            "video": video,
            "exercise": ex,
            "subject": subj,
            "view": view,
            "backend": "hrnet2d",
            "frames": frames,
        }
        row.update(metrics)
        rows_out.append(row)

    if not rows_out:
        print(f"[WARN] No rows produced from {args.kpts_root}")
        return

    # Fieldnames: union of keys
    fieldnames: List[str] = []
    seen = set()
    for r in rows_out:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[INFO] Wrote {len(rows_out)} HRNet2D metric rows to {args.output}")


if __name__ == "__main__":
    main()
