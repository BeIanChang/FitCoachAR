"""
Summarize first-wave Fit3D experiments into per-backend CSV tables.

目标：给每个模型一张“按视频一行”的表，包含：
  - exercise, subject, view
  - 角度 / 距离误差（mean_abs_err）
  - 延迟（latency_ms_total 或等效 fps）

现有数据来源：
  1) framewise_results/framewise_error_summary.csv
       - mediapipe_2d / mediapipe_3d / movenet_3d / hrnet2d
       - 每行一个 (exercise,subject,view,backend) 的逐帧 vs GT 误差汇总
  2) fit3d_results/fit3d_lifter_model_vs_gt.csv
       - motionbert / videopose1 / videopose243
  3) fit3d_results/fit3d_romp_model_vs_gt.csv
       - romp3d
  4) fit3d_results/fit3d_latency_summary.csv
       - 所有 backend 的延迟（frames, latency_ms, fps）

本脚本会在 fit3d_results 下生成：
  - table_mediapipe_2d.csv
  - table_mediapipe_3d.csv
  - table_movenet_3d.csv
  - table_hrnet2d.csv
  - table_motionbert.csv
  - table_videopose1.csv
  - table_videopose243.csv
  - table_romp3d.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def load_latency(lat_csv: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """Map (exercise, subject, view, backend) -> latency info."""
    info: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    if not lat_csv.exists():
        return info
    with lat_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (
                (r.get("exercise") or "").strip(),
                (r.get("subject") or "").strip(),
                (r.get("view") or "").strip(),
                (r.get("backend") or "").strip(),
            )
            info[key] = r
    return info


def summarize_framewise_backend(
    backend: str,
    err_csv: Path,
    lat_map: Dict[Tuple[str, str, str], Dict[str, str]],
    out_csv: Path,
    subject_filter: str = "s03",
) -> None:
    """从 framewise_error_summary 选出某个 backend 的行，并加上延迟."""
    if not err_csv.exists():
        return
    with err_csv.open() as f:
        reader = csv.DictReader(f)
        rows = [
            r
            for r in reader
            if (r.get("backend") or "").strip() == backend
            and (r.get("subject") or "").strip() == subject_filter
        ]
    if not rows:
        return

    # 只保留误差相关列和基本标识
    base_cols = ["exercise", "subject", "view", "backend"]
    mean_cols = [c for c in rows[0].keys() if c.endswith("_mean_abs_err")]
    std_cols = [c for c in rows[0].keys() if c.endswith("_std_abs_err")]
    n_cols = [c for c in rows[0].keys() if c.endswith("_n")]
    metric_cols = mean_cols + std_cols + n_cols

    out_rows: List[Dict[str, object]] = []
    for r in rows:
        ex = (r.get("exercise") or "").strip()
        subj = (r.get("subject") or "").strip()
        view = (r.get("view") or "").strip()
        key = (ex, subj, view, backend)
        lat = lat_map.get(key, {})

        row: Dict[str, object] = {c: r.get(c) for c in base_cols}
        for mc in metric_cols:
            row[mc] = r.get(mc)
        # 附加延迟
        row["frames"] = lat.get("frames")
        row["latency_ms"] = lat.get("latency_ms")
        row["fps"] = lat.get("fps")
        out_rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = base_cols + metric_cols + ["frames", "latency_ms", "fps"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def summarize_simple_source(
    src_csv: Path, backend: str, out_csv: Path, subject_filter: str = "s03"
) -> None:
    """
    对于 lifter 和 ROMP，源 CSV 已经包含误差和延迟，只需按 backend/subject 过滤即可.
    """
    if not src_csv.exists():
        return
    with src_csv.open() as f:
        reader = csv.DictReader(f)
        rows = [
            r
            for r in reader
            if (r.get("backend") or "").strip() == backend
            and (r.get("subject") or "").strip() == subject_filter
        ]
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce per-backend summary tables for Fit3D experiments."
    )
    parser.add_argument(
        "--framewise-errors",
        type=Path,
        default=Path("framewise_results/framewise_error_summary.csv"),
    )
    parser.add_argument(
        "--latency-summary",
        type=Path,
        default=Path("fit3d_results/fit3d_latency_summary.csv"),
    )
    parser.add_argument(
        "--lifter-metrics",
        type=Path,
        default=Path("fit3d_results/fit3d_lifter_model_vs_gt.csv"),
    )
    parser.add_argument(
        "--romp-metrics",
        type=Path,
        default=Path("fit3d_results/fit3d_romp_model_vs_gt.csv"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("fit3d_results"),
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="s03",
        help="Subject id to include (default s03).",
    )
    args = parser.parse_args()

    lat_map = load_latency(args.latency_summary)

    # 1) raw backends from framewise_error_summary
    for backend in ["mediapipe_2d", "mediapipe_3d", "movenet_3d", "hrnet2d"]:
        out_csv = args.out_root / f"table_{backend}.csv"
        summarize_framewise_backend(
            backend=backend,
            err_csv=args.framewise_errors,
            lat_map=lat_map,
            out_csv=out_csv,
            subject_filter=args.subject,
        )

    # 2) lifters
    for backend in ["motionbert", "videopose1", "videopose243"]:
        out_csv = args.out_root / f"table_{backend}.csv"
        summarize_simple_source(
            src_csv=args.lifter_metrics,
            backend=backend,
            out_csv=out_csv,
            subject_filter=args.subject,
        )

    # 3) ROMP
    summarize_simple_source(
        src_csv=args.romp_metrics,
        backend="romp3d",
        out_csv=args.out_root / "table_romp3d.csv",
        subject_filter=args.subject,
    )

    print("[INFO] Wrote per-backend tables under", args.out_root)


if __name__ == "__main__":
    main()
