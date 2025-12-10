"""
Apply signal processing to framewise metric JSON files produced by framewise_eval_backends.py.

Supported methods:
  - ema (default): exponential moving average with factor alpha
  - kalman: per-dimension 1D Kalman filter (state = value; tunable R/Q)
Optional: z-score clipping after smoothing.

Input structure (per file):
{
  "exercise": "...",
  "subject": "...",
  "view": "...",
  "backend": "...",
  "frames": [
     {"frame_index": 1, "elbow_R": ..., "knee_L": ..., "shl": ..., ...},
     ...
  ]
}

Processing:
  - For each numeric field (excluding frame_index), apply exponential moving
    average with factor alpha.
  - Optional z-score clipping: values with |z| > clip_sigma are clamped to
    the clip threshold around the mean of the processed sequence.

Output:
  Writes the same structure to a mirror directory, default:
    framewise_results_smoothed/<backend>/<file>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def ema_smooth(seq: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(seq)
    out[0] = seq[0]
    for i in range(1, len(seq)):
        out[i] = alpha * seq[i] + (1 - alpha) * out[i - 1]
    return out


def kalman_smooth(seq: np.ndarray, R: float, Q: float) -> np.ndarray:
    """Simple 1D Kalman filter per sequence (state = value)."""
    out = np.empty_like(seq)
    x = seq[0]  # state estimate
    P = 1.0     # initial covariance
    out[0] = x
    for i in range(1, len(seq)):
        # predict
        x_pred = x
        P_pred = P + Q
        # update
        K = P_pred / (P_pred + R)
        z = seq[i]
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        out[i] = x
    return out


def clip_by_sigma(seq: np.ndarray, sigma: float) -> np.ndarray:
    if len(seq) == 0 or not np.isfinite(seq).any():
        return seq
    mu = np.nanmean(seq)
    sd = np.nanstd(seq)
    if sd == 0 or not np.isfinite(sd):
        return seq
    lower = mu - sigma * sd
    upper = mu + sigma * sd
    return np.clip(seq, lower, upper)


def process_file(
    path: Path,
    out_root: Path,
    alpha: float,
    clip_sigma: float | None,
    method: str,
    kalman_R: float,
    kalman_Q: float,
) -> None:
    with path.open() as f:
        data = json.load(f)

    frames: List[Dict[str, float]] = data.get("frames", [])
    if not frames:
        return

    # Collect numeric fields (exclude frame_index)
    keys = [k for k in frames[0].keys() if k != "frame_index"]

    # Build arrays; handle keypoints separately
    arr: Dict[str, np.ndarray] = {}
    if "keypoints" in keys:
        klist = []
        for f in frames:
            kp = f.get("keypoints")
            if kp is None:
                kp = np.full((17, 3), np.nan)
            klist.append(np.asarray(kp, dtype=float))
        try:
            arr["keypoints"] = np.stack(klist, axis=0)  # [T,17,3]
        except Exception:
            arr["keypoints"] = np.asarray(klist, dtype=object)
        keys = [k for k in keys if k != "keypoints"]
    for k in keys:
        vals = [f.get(k, np.nan) for f in frames]
        try:
            arr[k] = np.asarray(vals, dtype=float)
        except Exception:
            arr[k] = np.asarray(vals, dtype=object)

    # Apply smoothing
    for k, v in arr.items():
        if k == "keypoints" and v.ndim == 3 and v.shape[1:] == (17, 3):
            sm = np.empty_like(v)
            for j in range(17):
                for c in range(3):
                    seq = v[:, j, c]
                    sm_seq = kalman_smooth(seq, kalman_R, kalman_Q) if method == "kalman" else ema_smooth(seq, alpha)
                    if clip_sigma is not None:
                        sm_seq = clip_by_sigma(sm_seq, clip_sigma)
                    sm[:, j, c] = sm_seq
            arr[k] = sm
            continue
        if v.ndim == 1:
            smoothed = kalman_smooth(v, kalman_R, kalman_Q) if method == "kalman" else ema_smooth(v, alpha)
            if clip_sigma is not None:
                smoothed = clip_by_sigma(smoothed, clip_sigma)
            arr[k] = smoothed
        else:
            # Skip non-1D arrays (e.g., vector keypoints); leave as-is
            arr[k] = v

    # Write back
    for i, frame in enumerate(frames):
        # keypoints first if present
        if "keypoints" in arr:
            frame["keypoints"] = arr["keypoints"][i].tolist()
        for k in [kk for kk in arr.keys() if kk != "keypoints"]:
            val = arr[k][i]
            if np.isscalar(val) and np.isfinite(val):
                frame[k] = float(val)
            else:
                frame[k] = val.tolist() if isinstance(val, np.ndarray) else val

    rel = path.relative_to(path.parents[1])  # framewise_results/<backend>/<file>.json -> <backend>/<file>.json
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f)
    print(f"[INFO] wrote smoothed -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smooth framewise metrics with EMA and optional sigma clipping.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("framewise_results"),
        help="Root containing backend subfolders with framewise JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("framewise_results_smoothed"),
        help="Root to write smoothed JSON files (mirrors input structure).",
    )
    parser.add_argument(
        "--method",
        choices=["ema", "kalman"],
        default="ema",
        help="Smoothing method.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="EMA smoothing factor (0-1, higher = less smoothing).",
    )
    parser.add_argument(
        "--kalman-R",
        type=float,
        default=0.05,
        help="Measurement noise covariance for Kalman (smaller = trust measurements less smoothing).",
    )
    parser.add_argument(
        "--kalman-Q",
        type=float,
        default=0.001,
        help="Process noise covariance for Kalman (larger = allow faster changes).",
    )
    parser.add_argument(
        "--clip-sigma",
        type=float,
        default=None,
        help="If set, z-score clip to +/- sigma after smoothing.",
    )
    args = parser.parse_args()

    if not args.input_root.exists():
        print(f"[WARN] input root not found: {args.input_root}")
        return

    for json_path in args.input_root.glob("*/*.json"):
        process_file(
            json_path,
            args.output_root,
            args.alpha,
            args.clip_sigma,
            args.method,
            args.kalman_R,
            args.kalman_Q,
        )


if __name__ == "__main__":
    main()
