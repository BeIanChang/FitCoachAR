"""
Aggregate latency metrics from different backends on Fit3D clips.

当前汇总两部分数据：
  1) HRNet-W32 2D 的离线结果（mmpose/tools/eval_fit3d_hrnet2d.py 生成）：
       fit3d_results/offline_eval_fit3d_hrnet2d.csv
  2) 各个 3D lifter（MotionBERT / VideoPose3D 等）在 mmpose 中
     跑出的 metrics.json：
       fit3d_lifter_preds/<backend>/<exercise>__<subject>__<view>/metrics.json

输出：
  一个统一的 CSV，例如：
    fit3d_results/fit3d_latency_summary.csv

  每行字段：
    exercise, subject, view, backend, frames, latency_ms, fps
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_hrnet2d_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                exercise = (r.get("exercise") or "").strip()
                subject = (r.get("subject") or "").strip()
                view = (r.get("view") or "").strip()
            except Exception:
                continue
            if not exercise or not subject or not view:
                continue
            try:
                frames = int(float(r.get("frames", "0")))
            except Exception:
                frames = 0
            try:
                latency_ms = float(r.get("avg_latency_ms", "nan"))
            except Exception:
                latency_ms = None  # type: ignore[assignment]
            try:
                fps = float(r.get("fps", "nan"))
            except Exception:
                fps = None  # type: ignore[assignment]

            rows.append(
                {
                    "exercise": exercise,
                    "subject": subject,
                    "view": view,
                    "backend": r.get("backend") or "hrnet2d",
                    "frames": frames,
                    "latency_ms": latency_ms,
                    "fps": fps,
                }
            )
    return rows


def load_lifter_metrics(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not root.exists():
        return rows

    for backend_dir in root.iterdir():
        if not backend_dir.is_dir():
            continue
        backend = backend_dir.name
        for clip_dir in backend_dir.iterdir():
            if not clip_dir.is_dir():
                continue
            metrics_path = clip_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                with metrics_path.open() as f:
                    m = json.load(f)
            except Exception:
                continue

            exercise = (m.get("exercise") or "").strip()
            subject = (m.get("subject") or "").strip()
            view = (m.get("view") or "").strip()
            if not exercise or not subject or not view:
                # 尝试从目录名解析：<exercise>__<subject>__<view>
                parts = clip_dir.name.split("__")
                if len(parts) == 3:
                    exercise, subject, view = parts
            if not exercise or not subject or not view:
                continue

            frames = m.get("frames")
            latency_ms = m.get("latency_ms")
            fps = m.get("fps_eq")

            rows.append(
                {
                    "exercise": exercise,
                    "subject": subject,
                    "view": view,
                    "backend": backend,
                    "frames": frames,
                    "latency_ms": latency_ms,
                    "fps": fps,
                }
            )
    return rows


def load_romp_metrics(root: Path) -> List[Dict[str, object]]:
    """Load ROMP latency metrics from fit3d_romp_preds/*/metrics.json."""
    rows: List[Dict[str, object]] = []
    if not root.exists():
        return rows

    for clip_dir in root.iterdir():
        if not clip_dir.is_dir():
            continue
        metrics_path = clip_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with metrics_path.open() as f:
                m = json.load(f)
        except Exception:
            continue

        # 目录名约定：<exercise>__<subject>__<view>
        parts = clip_dir.name.split("__")
        if len(parts) != 3:
            continue
        exercise, subject, view = parts

        frames = m.get("frames")
        total_ms = m.get("latency_ms_total")
        per_frame = m.get("latency_ms_per_frame")
        fps = 1000.0 / per_frame if per_frame and per_frame > 0 else None

        rows.append(
            {
                "exercise": exercise,
                "subject": subject,
                "view": view,
                "backend": "romp3d",
                "frames": frames,
                "latency_ms": total_ms,
                "fps": fps,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Fit3D latency metrics.")
    parser.add_argument(
        "--hrnet2d-csv",
        type=Path,
        default=Path("fit3d_results/offline_eval_fit3d_hrnet2d.csv"),
        help="CSV produced by eval_fit3d_hrnet2d.py.",
    )
    parser.add_argument(
        "--lifter-root",
        type=Path,
        default=Path("fit3d_lifter_preds"),
        help="Root containing lifter metrics (motionbert/videopose*/metrics.json).",
    )
    parser.add_argument(
        "--romp-root",
        type=Path,
        default=Path("fit3d_romp_preds"),
        help="Root containing ROMP metrics (fit3d_romp_preds/*/metrics.json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("fit3d_results/fit3d_latency_summary.csv"),
        help="Path to write aggregated latency CSV.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    rows.extend(load_hrnet2d_csv(args.hrnet2d_csv))
    rows.extend(load_lifter_metrics(args.lifter_root))
    rows.extend(load_romp_metrics(args.romp_root))

    if not rows:
        print("[WARN] No latency rows found; check paths.")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["exercise", "subject", "view", "backend", "frames", "latency_ms", "fps"]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Wrote latency summary to {args.output_csv}")


if __name__ == "__main__":
    main()
