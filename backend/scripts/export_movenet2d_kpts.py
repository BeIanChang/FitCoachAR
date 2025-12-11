"""
Export per-frame 2D keypoints from MoveNet service for a custom dataset.

Inputs
------
Meta CSV: columns [exercise,subject,view,video,start_frame,end_frame]
Video root: base directory containing videos (video paths are relative to this root,
            and are usually "subject/<exercise>/<view>/<file>.MOV").
MoveNet service: HTTP endpoint (default http://127.0.0.1:8502/infer) that takes
                 base64 JPEG and returns {"keypoints":[[y,x,score]...], "score":...}

Outputs
-------
framewise_kpts/<backend>/<exercise>__<subject>__<view>.json
{
  "exercise": "...",
  "subject": "...",
  "view": "...",
  "backend": "movenet_3d",
  "frames": [
     {"frame_index": 1, "keypoints": [[y,x,score], ...]},
     ...
  ]
}
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import requests


def load_meta(meta_csv: Path, subject: str, exercises: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
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
            video_rel = (r.get("video") or "").strip()
            if not ex or not subj or not view or not video_rel:
                continue
            rows.append(
                {
                    "exercise": ex,
                    "subject": subj,
                    "view": view,
                    "video": video_rel,
                    "azimuth": Path(video_rel).stem,  # front/back/side/front45/back45
                    "start_frame": (r.get("start_frame") or "").strip(),
                    "end_frame": (r.get("end_frame") or "").strip(),
                }
            )
    return rows


def encode_frame(frame_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return base64.b64encode(buf).decode("utf-8")


def call_movenet(service_url: str, frame_bgr, timeout: float) -> Optional[List[List[float]]]:
    payload = {"frame": encode_frame(frame_bgr)}
    try:
        resp = requests.post(service_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        kpts = data.get("keypoints")
        if kpts is None:
            return None
        return kpts
    except Exception as e:
        print(f"[WARN] MoveNet call failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-frame MoveNet 2D keypoints.")
    parser.add_argument("--meta-csv", type=Path, required=True, help="Path to meta CSV.")
    parser.add_argument("--video-root", type=Path, default=Path("dataset"), help="Root dir of videos.")
    parser.add_argument("--subject", type=str, required=True, help="Subject to process (e.g., diy).")
    parser.add_argument(
        "--exercises",
        nargs="+",
        default=[],
        help="Exercises to include. Empty means all found in meta.",
    )
    parser.add_argument(
        "--service-url",
        type=str,
        default=os.getenv("MOVENET_SERVICE_URL", "http://127.0.0.1:8502/infer"),
        help="MoveNet service URL.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP timeout seconds for MoveNet service.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("framewise_kpts"),
        help="Root to store per-video keypoint JSONs.",
    )
    args = parser.parse_args()

    meta_rows = load_meta(args.meta_csv, args.subject, args.exercises)
    if not meta_rows:
        print("[WARN] No meta rows matched.")
        return

    backend_name = "movenet_3d"
    for m in meta_rows:
        # Prefer direct video_rel under video_root; fallback to subject/video_rel;
        # also accept prefixed subject/ and "train/<...>" (Fit3D layout).
        candidates = [
            args.video_root / m["video"],
            args.video_root / args.subject / m["video"],
        ]
        vid_rel = m["video"]
        if vid_rel.startswith(f"{args.subject}/"):
            candidates.append(args.video_root / vid_rel)
        # Fit3D style: train/s03/videos/<view>/<exercise>.mp4
        candidates.append(args.video_root / "train" / vid_rel)
        if vid_rel.startswith("train/"):
            candidates.append(args.video_root / vid_rel)
        video_path = next((c for c in candidates if c.exists()), None)
        if video_path is None:
            print(f"[WARN] video missing for {m['video']}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] failed to open video: {video_path}")
            continue

        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sf = int(m["start_frame"]) if m["start_frame"].isdigit() else 1
        ef = int(m["end_frame"]) if m["end_frame"].isdigit() else T
        sf = max(1, sf)
        ef = min(T, ef)

        frames_out = []
        frame_idx = 0
        print(f"[INFO] MoveNet kpts on {video_path.name} frames {sf}-{ef}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx < sf or frame_idx > ef:
                continue
            kpts = call_movenet(args.service_url, frame, args.timeout)
            if kpts is None:
                continue
            frames_out.append({"frame_index": frame_idx, "keypoints": kpts})
        cap.release()

        out_dir = args.output_root / backend_name
        out_dir.mkdir(parents=True, exist_ok=True)
        view_full = f"{m['view']}_{m['azimuth']}"
        out_path = out_dir / f"{m['exercise']}__{m['subject']}__{view_full}.json"
        with out_path.open("w") as f:
            json.dump(
                {
                    "exercise": m["exercise"],
                    "subject": m["subject"],
                    "view": m["view"],
                    "azimuth": m["azimuth"],
                    "view_full": view_full,
                    "backend": backend_name,
                    "frames": frames_out,
                },
                f,
            )
        print(f"[INFO] wrote {out_path} ({len(frames_out)} frames)")


if __name__ == "__main__":
    main()
