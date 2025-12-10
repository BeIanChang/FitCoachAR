"""
Compute angle/distance metrics from per-frame keypoints JSON (no GT).

Input JSON structure (produced by export_movenet2d_kpts.py):
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

Output CSV (per video):
  video,exercise,subject,view,backend,frames,
  elbow_mean,elbow_std,left_elbow_mean,left_elbow_std,
  knee_mean,knee_std,left_knee_mean,left_knee_std,
  hip_mean,hip_std,left_hip_mean,left_hip_std,
  shoulder_mean,shoulder_std,left_shoulder_mean,left_shoulder_std,
  shl_mean,shl_std,rshl_rpalm_mean,rshl_rpalm_std,
  rshl_rhip_mean,rshl_rhip_std,rpalm_rhip_mean,rpalm_rhip_std,
  rknee_rhip_mean,rknee_rhip_std,rknee_rfeet_mean,rknee_rfeet_std,
  rhip_rfeet_mean,rhip_rfeet_std
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def compute_metrics(kpts: np.ndarray) -> Dict[str, float]:
    """
    kpts: [T,17,3] (y,x,score) assumed normalized coordinates.
    """
    # indices (COCO style)
    NOSE = 0
    L_SH, R_SH = 5, 6
    L_EL, R_EL = 7, 8
    L_WR, R_WR = 9, 10
    L_HP, R_HP = 11, 12
    L_KN, R_KN = 13, 14
    L_AN, R_AN = 15, 16

    angles: Dict[str, List[float]] = {
        "elbow_R": [],
        "elbow_L": [],
        "knee_R": [],
        "knee_L": [],
        "hip_R": [],
        "hip_L": [],
        "shoulder_R": [],
        "shoulder_L": [],
    }
    dists: Dict[str, List[float]] = {
        "shl": [],
        "rshl_rpalm": [],
        "rshl_rhip": [],
        "rpalm_rhip": [],
        "rknee_rhip": [],
        "rknee_rfeet": [],
        "rhip_rfeet": [],
    }

    for p in kpts:
        nose = p[NOSE, :2]
        l_sh, r_sh = p[L_SH, :2], p[R_SH, :2]
        l_el, r_el = p[L_EL, :2], p[R_EL, :2]
        l_wr, r_wr = p[L_WR, :2], p[R_WR, :2]
        l_hp, r_hp = p[L_HP, :2], p[R_HP, :2]
        l_kn, r_kn = p[L_KN, :2], p[R_KN, :2]
        l_an, r_an = p[L_AN, :2], p[R_AN, :2]

        angles["elbow_R"].append(joint_angle(r_sh, r_el, r_wr))
        angles["elbow_L"].append(joint_angle(l_sh, l_el, l_wr))
        angles["knee_R"].append(joint_angle(r_hp, r_kn, r_an))
        angles["knee_L"].append(joint_angle(l_hp, l_kn, l_an))
        angles["hip_R"].append(joint_angle(r_sh, r_hp, r_kn))
        angles["hip_L"].append(joint_angle(l_sh, l_hp, l_kn))
        angles["shoulder_R"].append(joint_angle(r_hp, r_sh, r_el))
        angles["shoulder_L"].append(joint_angle(l_hp, l_sh, l_el))

        dists["shl"].append(float(np.linalg.norm(l_sh - r_sh)))
        dists["rshl_rpalm"].append(float(r_sh[1] - r_wr[1]))
        dists["rshl_rhip"].append(float(r_hp[1] - r_sh[1]))
        dists["rpalm_rhip"].append(float(r_hp[1] - r_wr[1]))
        dists["rknee_rhip"].append(float(r_hp[1] - r_kn[1]))
        dists["rknee_rfeet"].append(float(r_an[1] - r_kn[1]))
        dists["rhip_rfeet"].append(float(r_an[1] - r_hp[1]))

    def mean_std(vals: List[float]) -> Tuple[float, float]:
        arr = np.array(vals, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    out: Dict[str, float] = {}
    # angles
    r_el_m, r_el_s = mean_std(angles["elbow_R"])
    l_el_m, l_el_s = mean_std(angles["elbow_L"])
    r_kn_m, r_kn_s = mean_std(angles["knee_R"])
    l_kn_m, l_kn_s = mean_std(angles["knee_L"])
    r_hp_m, r_hp_s = mean_std(angles["hip_R"])
    l_hp_m, l_hp_s = mean_std(angles["hip_L"])
    r_sh_m, r_sh_s = mean_std(angles["shoulder_R"])
    l_sh_m, l_sh_s = mean_std(angles["shoulder_L"])

    out.update(
        {
            "elbow_mean": np.nanmean([r_el_m, l_el_m]),
            "elbow_std": np.nanmean([r_el_s, l_el_s]),
            "right_elbow_mean": r_el_m,
            "right_elbow_std": r_el_s,
            "left_elbow_mean": l_el_m,
            "left_elbow_std": l_el_s,
            "knee_mean": np.nanmean([r_kn_m, l_kn_m]),
            "knee_std": np.nanmean([r_kn_s, l_kn_s]),
            "right_knee_mean": r_kn_m,
            "right_knee_std": r_kn_s,
            "left_knee_mean": l_kn_m,
            "left_knee_std": l_kn_s,
            "hip_mean": np.nanmean([r_hp_m, l_hp_m]),
            "hip_std": np.nanmean([r_hp_s, l_hp_s]),
            "right_hip_mean": r_hp_m,
            "right_hip_std": r_hp_s,
            "left_hip_mean": l_hp_m,
            "left_hip_std": l_hp_s,
            "shoulder_mean": np.nanmean([r_sh_m, l_sh_m]),
            "shoulder_std": np.nanmean([r_sh_s, l_sh_s]),
            "right_shoulder_mean": r_sh_m,
            "right_shoulder_std": r_sh_s,
            "left_shoulder_mean": l_sh_m,
            "left_shoulder_std": l_sh_s,
        }
    )
    # distances
    for k, vals in dists.items():
        m, s = mean_std(vals)
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = s
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute metrics from per-frame keypoints JSON (no GT).")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root containing backend subfolders with keypoint JSON files (e.g., framewise_kpts_fit3d/movenet_3d/*.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows_out: List[Dict[str, object]] = []
    # allow files directly under input_root or one-level nested
    candidates = list(args.input_root.glob("*.json")) + list(args.input_root.glob("*/*.json"))
    for json_path in candidates:
        with json_path.open() as f:
            data = json.load(f)
        frames = data.get("frames", [])
        if not frames:
            continue
        kpts_list = []
        for fr in frames:
            k = fr.get("keypoints")
            if k is None:
                continue
            kpts_list.append(k)
        if not kpts_list:
            continue
        kpts = np.asarray(kpts_list, dtype=float)  # [T,17,3]
        metrics = compute_metrics(kpts)
        row: Dict[str, object] = {
            "video": data.get("video", json_path.name),
            "exercise": data.get("exercise"),
            "subject": data.get("subject"),
            "view": data.get("view_full", data.get("view")),
            "backend": data.get("backend"),
            "frames": kpts.shape[0],
        }
        row.update(metrics)
        rows_out.append(row)

    if not rows_out:
        print("[WARN] No metrics produced.")
        return

    # collect fieldnames
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
    print(f"[INFO] wrote {len(rows_out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
