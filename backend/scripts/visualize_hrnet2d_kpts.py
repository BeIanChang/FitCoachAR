"""
Quickly visualize stored HRNet-2D keypoints on top of the original Fit3D video.

目的：用已经导出的 `hrnet2d_kpts/.../kpts2d_hrnet.json` 来还原到视频帧里，
检查 2D 关键点与原始画面是否对齐，而不再重新跑 HRNet。

用法示例（在 FitCoachAR 根目录）::

    # 随便挑一个 clip，比如 barbell_row
    python -m backend.scripts.visualize_hrnet2d_kpts \\
      --tag barbell_row__s03__50591643 \\
      --kpts-root hrnet2d_kpts \\
      --meta-csv fit3d_subset/gt_meta_s03.csv \\
      --out-dir debug_hrnet2d_overlays \\
      --indices 0 200 500

这样会：
  - 从 hrnet2d_kpts/barbell_row__s03__50591643/kpts2d_hrnet.json 读取 2D 关键点；
  - 从 gt_meta_s03.csv 里找到 start_frame/end_frame，对齐到原视频帧号；
  - 对于 indices 里给出的第 t 帧，在原视频第 (start_frame + t) 帧画 17 个点；
  - 把结果保存为若干 PNG，便于你肉眼检查。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MetaRow:
    exercise: str
    subject: str
    view: str
    start_frame: Optional[int]
    end_frame: Optional[int]


def load_meta(meta_csv: Path) -> Dict[Tuple[str, str, str], MetaRow]:
    """Load gt_meta.csv into a dict keyed by (exercise, subject, view)."""
    meta: Dict[Tuple[str, str, str], MetaRow] = {}
    if not meta_csv.exists():
        return meta
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
            meta[(ex, subj, view)] = MetaRow(
                exercise=ex,
                subject=subj,
                view=view,
                start_frame=sf,
                end_frame=ef,
            )
    return meta


def draw_keypoints_on_frame(
    frame: np.ndarray, kpts_2d: np.ndarray
) -> np.ndarray:
    """在一帧图像上画 17 个 COCO 关键点."""
    out = frame.copy()
    for x, y in kpts_2d:
        cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize stored HRNet2D keypoints on original Fit3D frames."
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Clip tag in the form <exercise>__<subject>__<view>, e.g. barbell_row__s03__50591643.",
    )
    parser.add_argument(
        "--kpts-root",
        type=Path,
        default=Path("hrnet2d_kpts"),
        help="Root directory containing <tag>/kpts2d_hrnet.json.",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=Path("fit3d_subset/gt_meta_s03.csv"),
        help="Meta CSV for mapping local index -> global video frame index.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("debug_hrnet2d_overlays"),
        help="Directory to save overlay PNGs.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Keypoint frame indices (0-based within the clip) to visualize. "
        "If omitted,会自动选首/中/尾三帧。",
    )
    args = parser.parse_args()

    kpts_path = args.kpts_root / args.tag / "kpts2d_hrnet.json"
    if not kpts_path.exists():
        raise SystemExit(f"Keypoint file not found: {kpts_path}")

    with kpts_path.open() as f:
        data = json.load(f)

    exercise = (data.get("exercise") or "").strip()
    subject = (data.get("subject") or "").strip()
    view = (data.get("view") or "").strip()
    video_path = Path(data.get("video") or "")
    keypoints = np.asarray(data.get("keypoints", []), dtype=float)  # [T,17,2]

    if keypoints.ndim != 3 or keypoints.shape[1] != 17:
        raise SystemExit(f"Unexpected keypoints shape: {keypoints.shape}")
    T = keypoints.shape[0]

    meta_map = load_meta(args.meta_csv)
    meta = meta_map.get((exercise, subject, view))
    if meta and meta.start_frame:
        start_frame = max(1, meta.start_frame)
    else:
        start_frame = 1  # fallback：假设从第一帧开始

    if args.indices:
        indices = [i for i in args.indices if 0 <= i < T]
    else:
        # 默认取首/中/尾三帧
        mid = T // 2
        indices = sorted({0, mid, T - 1})

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        global_frame = start_frame + idx  # gt_meta 中是 1-based
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(global_frame - 1, 0))
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Failed to read frame {global_frame} from {video_path}")
            continue

        overlay = draw_keypoints_on_frame(frame, keypoints[idx])
        out_name = f"{exercise}__{subject}__{view}_t{idx}_f{global_frame}.png"
        out_path = args.out_dir / out_name
        cv2.imwrite(str(out_path), overlay)
        print(f"[INFO] Saved overlay -> {out_path}")

    cap.release()


if __name__ == "__main__":
    main()

