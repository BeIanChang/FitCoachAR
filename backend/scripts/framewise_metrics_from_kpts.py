"""
Compute per-frame angle/distance metrics from keypoint JSON files (no GT).

Input JSON (per video), produced by export_movenet2d_kpts.py or smoothed results:
{
  "exercise": "...",
  "subject": "...",
  "view": "...",
  "backend": "...",
  "frames": [
    {"frame_index": 1, "keypoints": [[y,x,score], ... 17 points]},
    ...
  ]
}

Output JSON mirrors input files under output root:
frames array where each frame keeps frame_index and adds:
  elbow_R, elbow_L, knee_R, knee_L, hip_R, hip_L, shoulder_R, shoulder_L,
  shl, rshl_rpalm, rshl_rhip, rpalm_rhip, rknee_rhip, rknee_rfeet, rhip_rfeet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cos_th = float(np.dot(v1, v2) / (n1 * n2))
    cos_th = max(-1.0, min(1.0, cos_th))
    return float(np.degrees(np.arccos(cos_th)))


def process_file(src: Path, dst_root: Path) -> None:
    with src.open() as f:
        data = json.load(f)
    frames: List[Dict] = data.get("frames", [])
    if not frames:
        return

    # COCO indices
    NOSE = 0
    L_SH, R_SH = 5, 6
    L_EL, R_EL = 7, 8
    L_WR, R_WR = 9, 10
    L_HP, R_HP = 11, 12
    L_KN, R_KN = 13, 14
    L_AN, R_AN = 15, 16

    out_frames: List[Dict] = []
    for fr in frames:
        kpts = fr.get("keypoints")
        if kpts is None:
            continue
        p = np.asarray(kpts, dtype=float)
        if p.shape != (17, 3):
            continue

        l_sh, r_sh = p[L_SH, :2], p[R_SH, :2]
        l_el, r_el = p[L_EL, :2], p[R_EL, :2]
        l_wr, r_wr = p[L_WR, :2], p[R_WR, :2]
        l_hp, r_hp = p[L_HP, :2], p[R_HP, :2]
        l_kn, r_kn = p[L_KN, :2], p[R_KN, :2]
        l_an, r_an = p[L_AN, :2], p[R_AN, :2]

        rec = {
            "frame_index": fr.get("frame_index"),
            "elbow_R": joint_angle(r_sh, r_el, r_wr),
            "elbow_L": joint_angle(l_sh, l_el, l_wr),
            "knee_R": joint_angle(r_hp, r_kn, r_an),
            "knee_L": joint_angle(l_hp, l_kn, l_an),
            "hip_R": joint_angle(r_sh, r_hp, r_kn),
            "hip_L": joint_angle(l_sh, l_hp, l_kn),
            "shoulder_R": joint_angle(r_hp, r_sh, r_el),
            "shoulder_L": joint_angle(l_hp, l_sh, l_el),
            "shl": float(np.linalg.norm(l_sh - r_sh)),
            "rshl_rpalm": float(r_sh[1] - r_wr[1]),
            "rshl_rhip": float(r_hp[1] - r_sh[1]),
            "rpalm_rhip": float(r_hp[1] - r_wr[1]),
            "rknee_rhip": float(r_hp[1] - r_kn[1]),
            "rknee_rfeet": float(r_an[1] - r_kn[1]),
            "rhip_rfeet": float(r_an[1] - r_hp[1]),
        }
        out_frames.append(rec)

    # write
    rel = src.relative_to(src.parents[1])  # <backend>/<file>.json
    out_path = dst_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "exercise": data.get("exercise"),
        "subject": data.get("subject"),
        "view": data.get("view"),
        "azimuth": data.get("azimuth"),
        "view_full": data.get("view_full", data.get("view")),
        "backend": data.get("backend"),
        "frames": out_frames,
    }
    with out_path.open("w") as f:
        json.dump(out, f)
    print(f"[INFO] wrote framewise metrics -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-frame metrics from keypoint JSONs (no GT).")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root containing backend subfolders with keypoint JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root to store per-video framewise metrics JSONs.",
    )
    args = parser.parse_args()

    candidates = list(args.input_root.glob("*.json")) + list(args.input_root.glob("*/*.json"))
    if not candidates:
        print(f"[WARN] No input JSONs under {args.input_root}")
        return

    for src in candidates:
        process_file(src, args.output_root)


if __name__ == "__main__":
    main()
