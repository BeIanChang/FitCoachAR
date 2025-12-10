"""
Compute per-frame error between model metrics (from keypoints) and GT metrics.

Inputs:
  - GT framewise metrics JSONs (from framewise_gt.py):
      framewise_results/gt/<exercise>__<subject>__<view>.json
  - Model framewise metrics JSONs (from framewise_metrics_from_kpts.py):
      framewise_metrics_fit3d_smoothed/movenet_3d/<exercise>__<subject>__<view>[_...].json

Each JSON has:
  {
    "exercise": "...",
    "subject": "...",
    "view": "...",
    "frames": [
      {"frame_index": 1, "elbow_R": ..., "knee_L": ..., "shl": ..., ...},
      ...
    ]
  }

Outputs:
  - framewise_error_summary.csv (per view/backend): mean_abs_err for each metric.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_framewise_map(root: Path) -> Dict[Tuple[str, str, str], Dict[int, Dict[str, float]]]:
    """
    Return map: (exercise, subject, view_full) -> {frame_index: metric dict}
    """
    out: Dict[Tuple[str, str, str], Dict[int, Dict[str, float]]] = {}
    for path in root.glob("*.json"):
        with path.open() as f:
            data = json.load(f)
        ex = data.get("exercise")
        subj = data.get("subject")
        view = data.get("view_full", data.get("view"))
        if not ex or not subj or not view:
            continue
        key = (ex, subj, str(view))
        fmap: Dict[int, Dict[str, float]] = {}
        for fr in data.get("frames", []):
            idx = fr.get("frame_index")
            if idx is None:
                continue
            fmap[int(idx)] = fr
        out[key] = fmap
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model framewise metrics vs GT.")
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("framewise_results/gt"),
        help="Root containing GT framewise JSONs.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Root containing model framewise metrics JSONs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("fit3d_results/framewise_model_vs_gt.csv"),
        help="Output CSV.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="movenet_3d",
        help="Backend name for output labeling.",
    )
    args = parser.parse_args()

    gt_map = load_framewise_map(args.gt_root)
    model_map = load_framewise_map(args.model_root)

    if not model_map:
        print(f"[WARN] No model files found under {args.model_root}")
        return

    rows: List[Dict[str, object]] = []
    metrics_keys = [
        "elbow_R",
        "elbow_L",
        "knee_R",
        "knee_L",
        "hip_R",
        "hip_L",
        "shoulder_R",
        "shoulder_L",
        "shl",
        "rshl_rpalm",
        "rshl_rhip",
        "rpalm_rhip",
        "rknee_rhip",
        "rknee_rfeet",
        "rhip_rfeet",
    ]

    for key, fmap_model in model_map.items():
        ex, subj, view = key
        fmap_gt = gt_map.get(key)
        # allow fallback: if view contains extra suffix (e.g., "50591643_exercise"), try prefix before "_"
        if fmap_gt is None and "_" in view:
            view_prefix = view.split("_")[0]
            fmap_gt = gt_map.get((ex, subj, view_prefix))
        if not fmap_gt:
            print(f"[WARN] No GT for {key}")
            continue
        errs: Dict[str, List[float]] = {m: [] for m in metrics_keys}
        for idx, mrec in fmap_model.items():
            grec = fmap_gt.get(idx)
            if not grec:
                continue
            for m in metrics_keys:
                mv = mrec.get(m)
                gv = grec.get(m)
                if mv is None or gv is None:
                    continue
                errs[m].append(abs(float(mv) - float(gv)))
        if not any(errs.values()):
            continue
        row: Dict[str, object] = {
            "backend": args.backend,
            "exercise": ex,
            "subject": subj,
            "view": view,
        }
        for m in metrics_keys:
            if errs[m]:
                row[f"{m}_mean_abs_err"] = sum(errs[m]) / len(errs[m])
            else:
                row[f"{m}_mean_abs_err"] = ""
        rows.append(row)

    if not rows:
        print("[WARN] No rows produced.")
        return

    # Fieldnames
    fieldnames = ["backend", "exercise", "subject", "view"] + [
        f"{m}_mean_abs_err" for m in metrics_keys
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
