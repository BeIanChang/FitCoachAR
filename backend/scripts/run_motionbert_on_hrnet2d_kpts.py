"""
Run MotionBERT 3D lifter on precomputed HRNet2D keypoints (Fit3D, subject s03).

This script does NOT touch ground-truth metrics or angles. It only:

  1. Reads per-frame 2D keypoints exported by
       mmpose/tools/export_fit3d_hrnet2d_kpts.py
     under a root like:
       FitCoachAR/hrnet2d_kpts/<exercise>__s03__<view>/kpts2d_hrnet.json

  2. Converts COCO-17 keypoints to a simple H36M-17 layout (x,y,0),
     without extra normalisation (no [-1,1] scaling).

  3. Feeds the 2D sequence to MotionBERT using a sliding temporal window
     of fixed length (default 243) and stride=1:
        - For each frame t, we form a window centred at t,
          pad at the boundaries by repeating edge frames,
          and run one forward pass.
        - We take the centre frame output from the window as the 3D
          prediction for time t.

  4. Writes, for each clip:
       hrnet2d_kpts/<tag>/motionbert3d_kpts.json
       hrnet2d_kpts/<tag>/metrics_motionbert3d.json

  where <tag> is "<exercise>__<subject>__<view>".

The output JSON format for 3D keypoints:

  {
    "exercise": "squat",
    "subject": "s03",
    "view": "50591643",
    "backend": "motionbert3d",
    "video": "absolute/path/to/video.mp4",
    "keypoints3d": [  // length T
      [[x0,y0,z0], ..., [x16,y16,z16]],
      ...
    ]
  }

Latency JSON:

  {
    "exercise": "squat",
    "subject": "s03",
    "view": "50591643",
    "backend": "motionbert3d",
    "video": "...",
    "frames": T,
    "latency_ms": 1234.5,
    "fps_eq": 27.3
  }

This is an offline analysis tool; it does not affect real-time backends.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# --- Wire in MotionBERT repo --------------------------------------------------

THIS_DIR = Path(__file__).resolve().parents[2]
MB_ROOT = THIS_DIR.parent / "MotionBERT"
if str(MB_ROOT) not in sys.path:
    sys.path.insert(0, str(MB_ROOT))

from lib.utils.tools import get_config  # type: ignore
from lib.utils.learning import load_backbone  # type: ignore


def coco17_to_h36m17(kpts2d: np.ndarray) -> np.ndarray:
    """Rudimentary mapping from COCO-17 2D to H36M-17 layout.

    Args:
        kpts2d: [T,17,2] array of (x,y) in pixels.

    Returns:
        motion: [T,17,3] array (x,y,0) in a H36M-style ordering:
            0: hip (pelvis, mid-hip)
            1: right_hip
            2: right_knee
            3: right_ankle
            4: left_hip
            5: left_knee
            6: left_ankle
            7: spine (mid of hips and shoulders)
            8: thorax (mid-shoulder)
            9: neck (use nose)
            10: head (use nose as proxy)
            11: left_shoulder
            12: left_elbow
            13: left_wrist
            14: right_shoulder
            15: right_elbow
            16: right_wrist
    """
    T = kpts2d.shape[0]
    motion = np.zeros((T, 17, 3), dtype=np.float32)

    # COCO indices
    NOSE = 0
    L_SH, R_SH = 5, 6
    L_EL, R_EL = 7, 8
    L_WR, R_WR = 9, 10
    L_HP, R_HP = 11, 12
    L_KN, R_KN = 13, 14
    L_AN, R_AN = 15, 16

    nose = kpts2d[:, NOSE]
    l_sh = kpts2d[:, L_SH]
    r_sh = kpts2d[:, R_SH]
    l_el = kpts2d[:, L_EL]
    r_el = kpts2d[:, R_EL]
    l_wr = kpts2d[:, L_WR]
    r_wr = kpts2d[:, R_WR]
    l_hp = kpts2d[:, L_HP]
    r_hp = kpts2d[:, R_HP]
    l_kn = kpts2d[:, L_KN]
    r_kn = kpts2d[:, R_KN]
    l_an = kpts2d[:, L_AN]
    r_an = kpts2d[:, R_AN]

    pelvis = (l_hp + r_hp) * 0.5  # [T,2]
    thorax = (l_sh + r_sh) * 0.5
    spine = (pelvis + thorax) * 0.5

    motion[:, 0, :2] = pelvis
    motion[:, 1, :2] = r_hp
    motion[:, 2, :2] = r_kn
    motion[:, 3, :2] = r_an
    motion[:, 4, :2] = l_hp
    motion[:, 5, :2] = l_kn
    motion[:, 6, :2] = l_an
    motion[:, 7, :2] = spine
    motion[:, 8, :2] = thorax
    motion[:, 9, :2] = nose  # neck approx
    motion[:, 10, :2] = nose  # head approx
    motion[:, 11, :2] = l_sh
    motion[:, 12, :2] = l_el
    motion[:, 13, :2] = l_wr
    motion[:, 14, :2] = r_sh
    motion[:, 15, :2] = r_el
    motion[:, 16, :2] = r_wr

    # third coordinate set to zero
    return motion


def sliding_motionbert(
    model: torch.nn.Module,
    motion_2d: np.ndarray,
    window: int,
) -> np.ndarray:
    """Run MotionBERT on a 2D sequence via sliding window.

    Args:
        model: MotionBERT model (DSTformer) in eval() mode.
        motion_2d: [T,17,3] 2D input (x,y,0) in H36M layout.
        window: temporal window length (e.g., 243).

    Returns:
        pred_3d: [T,17,3] 3D predictions.
    """
    T = motion_2d.shape[0]
    if T == 0:
        return np.zeros((0, 17, 3), dtype=np.float32)

    W = max(1, window)
    half = W // 2
    pred_3d = np.zeros((T, 17, 3), dtype=np.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(T):
            # indices with edge padding by replication
            idx = np.arange(t - half, t + half + 1)
            idx = np.clip(idx, 0, T - 1)
            clip = motion_2d[idx]  # [W,17,3]
            x = torch.from_numpy(clip[None]).float()  # [1,W,17,3]
            out = model(x)  # [1,W,17,3]
            center = out.shape[1] // 2
            pred_3d[t] = out[0, center].cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[motionbert3d] sliding window: T={T}, window={W}, "
        f"latency={elapsed_ms:.1f} ms, ~{T / (elapsed_ms / 1000.0):.2f} fps"
    )
    return pred_3d, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MotionBERT on HRNet2D keypoints (Fit3D s03)."
    )
    parser.add_argument(
        "--kpts-root",
        type=Path,
        required=True,
        help="Root directory containing hrnet2d_kpts/<tag>/kpts2d_hrnet.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="MotionBERT config file (e.g., ../MotionBERT/configs/pose3d/MB_ft_h36m.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="MotionBERT checkpoint (e.g., ../MotionBERT/checkpoint/pose3d/MB_ft_h36m/best_epoch.bin).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=243,
        help="Temporal window size for sliding inference (default: 243).",
    )
    args = parser.parse_args()

    cfg = get_config(str(args.config))
    model = load_backbone(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_pos"], strict=False)
    if missing:
        print(f"[WARN] MotionBERT missing keys: {missing}")
    if unexpected:
        print(f"[WARN] MotionBERT unexpected keys: {unexpected}")
    model.eval()

    # Iterate all hrnet2d_kpts/* directories
    kpts_root = args.kpts_root
    if not kpts_root.exists():
        print(f"[ERROR] kpts_root does not exist: {kpts_root}")
        return

    for sub in sorted(kpts_root.iterdir()):
        if not sub.is_dir():
            continue
        kpts_path = sub / "kpts2d_hrnet.json"
        if not kpts_path.exists():
            continue
        with kpts_path.open() as f:
            data = json.load(f)
        exercise = data.get("exercise", "")
        subject = data.get("subject", "")
        view = data.get("view", "")
        video = data.get("video", "")
        keypoints = np.asarray(data.get("keypoints", []), dtype=float)  # [T,17,2]
        if keypoints.ndim != 3 or keypoints.shape[1] != 17:
            print(f"[WARN] Unexpected keypoints shape in {kpts_path}: {keypoints.shape}")
            continue
        T = keypoints.shape[0]

        print(f"[motionbert3d] {sub.name} ({exercise}, {view}), T={T}")

        motion_2d = coco17_to_h36m17(keypoints)  # [T,17,3]
        pred_3d, elapsed_ms = sliding_motionbert(model, motion_2d, args.window)

        # Save 3D keypoints
        out_kpts = {
            "exercise": exercise,
            "subject": subject,
            "view": view,
            "backend": "motionbert3d",
            "video": video,
            "keypoints3d": pred_3d.tolist(),
        }
        with (sub / "motionbert3d_kpts.json").open("w") as f:
            json.dump(out_kpts, f)

        # Save latency info
        fps_eq = T / (elapsed_ms / 1000.0) if elapsed_ms > 0 and T > 0 else None
        latency = {
            "exercise": exercise,
            "subject": subject,
            "view": view,
            "backend": "motionbert3d",
            "video": video,
            "frames": T,
            "latency_ms": elapsed_ms,
            "fps_eq": fps_eq,
        }
        with (sub / "metrics_motionbert3d.json").open("w") as f:
            json.dump(latency, f, indent=2)


if __name__ == "__main__":
    main()

