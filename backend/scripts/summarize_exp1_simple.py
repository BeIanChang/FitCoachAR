"""
Summarize Exp#1 model results into upper/lower body angle + distance metrics.

输入：fit3d_results/exp1_model 下的各个 CSV：
  - table_mediapipe_2d.csv
  - table_mediapipe_3d.csv
  - table_movenet_3d.csv
  - table_hrnet2d.csv
  - fit3d_lifter_model_vs_gt.csv
  - table_romp3d.csv

对每一行 (exercise, subject, view, backend)：
  - 识别角度误差列：elbow/shoulder/knee/hip 的 mean_abs_err/std_abs_err
    * 上半身角度：左/右肘+肩的 mean 合并平均 => upper_mean；std 合并平均 => upper_std
    * 下半身角度：左/右膝+髋的 mean 合并平均 => lower_mean；std 合并平均 => lower_std
    * upper_better_mean/std：取左右上半身 mean 更小的一侧，std 用对应侧
    * lower_better_mean/std：同理
  - 距离误差：其它 *_mean_abs_err 列的平均值，std 同理
  - 延迟：优先使用 fps / fps_eq，如果只有 latency_ms + frames，就换算成 fps

输出：fit3d_results/exp1_model/summary_simple.csv
  字段：
    exercise, subject, view, backend,
    upper_angle_mean, upper_angle_std,
    lower_angle_mean, lower_angle_std,
    dist_mean, dist_std,
    frames, latency_ms, fps
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def load_rows(path: Path, backend_name: str | None = None) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if backend_name is None:
        return rows
    out: List[Dict[str, str]] = []
    for r in rows:
        b = (r.get("backend") or "").strip()
        if not b:
            b = backend_name
            r = dict(r)
            r["backend"] = backend_name
        if b == backend_name:
            out.append(r)
    return out


def to_float(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None


def summarize_row(r: Dict[str, str]) -> Dict[str, object]:
    backend = (r.get("backend") or "").strip()
    exercise = (r.get("exercise") or "").strip()
    subject = (r.get("subject") or "").strip()
    view = (r.get("view") or "").strip()

    cols = list(r.keys())
    def col_exists(name: str) -> bool:
        return name in cols

    def prefer_abs_err(base: str) -> str | None:
        """Return column name: *_mean_abs_err if exists, else *_mean."""
        abs_err = f"{base}_mean_abs_err"
        plain = f"{base}_mean"
        if col_exists(abs_err):
            return abs_err
        if col_exists(plain):
            return plain
        return None

    def prefer_abs_err_std(base: str) -> str | None:
        """Return column name: *_std_abs_err if exists, else *_std."""
        abs_err = f"{base}_std_abs_err"
        plain = f"{base}_std"
        if col_exists(abs_err):
            return abs_err
        if col_exists(plain):
            return plain
        return None

    angle_mean_cols = [
        c
        for c in cols
        if (c.endswith("_mean_abs_err") or c.endswith("_mean"))
        and any(key in c for key in ["elbow", "shoulder", "knee", "hip"])
    ]
    angle_std_cols = [
        c
        for c in cols
        if (c.endswith("_std_abs_err") or c.endswith("_std"))
        and any(key in c for key in ["elbow", "shoulder", "knee", "hip"])
    ]
    # 上半身：elbow + shoulder；下半身：knee + hip（左右列固定命名）
    left_upper_mean_cols = [
        c for c in [prefer_abs_err("elbow_L"), prefer_abs_err("shoulder_L")] if c
    ]
    right_upper_mean_cols = [
        c for c in [prefer_abs_err("elbow_R"), prefer_abs_err("shoulder_R")] if c
    ]
    left_lower_mean_cols = [
        c for c in [prefer_abs_err("knee_L"), prefer_abs_err("hip_L")] if c
    ]
    right_lower_mean_cols = [
        c for c in [prefer_abs_err("knee_R"), prefer_abs_err("hip_R")] if c
    ]

    left_upper_std_cols = [
        c for c in [prefer_abs_err_std("elbow_L"), prefer_abs_err_std("shoulder_L")] if c
    ]
    right_upper_std_cols = [
        c for c in [prefer_abs_err_std("elbow_R"), prefer_abs_err_std("shoulder_R")] if c
    ]
    left_lower_std_cols = [
        c for c in [prefer_abs_err_std("knee_L"), prefer_abs_err_std("hip_L")] if c
    ]
    right_lower_std_cols = [
        c for c in [prefer_abs_err_std("knee_R"), prefer_abs_err_std("hip_R")] if c
    ]

    dist_mean_cols = [
        c
        for c in cols
        if (c.endswith("_mean_abs_err") or c.endswith("_mean"))
        and c not in angle_mean_cols
    ]
    dist_std_cols = [
        c
        for c in cols
        if (c.endswith("_std_abs_err") or c.endswith("_std"))
        and c not in angle_std_cols
    ]

    def mean_of(cols: List[str]) -> float | None:
        vals: List[float] = []
        for c in cols:
            v = to_float(r.get(c))
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        return sum(vals) / len(vals)

    left_upper_mean = mean_of(left_upper_mean_cols)
    right_upper_mean = mean_of(right_upper_mean_cols)
    left_lower_mean = mean_of(left_lower_mean_cols)
    right_lower_mean = mean_of(right_lower_mean_cols)
    dist_mean = mean_of(dist_mean_cols)

    left_upper_std = mean_of(left_upper_std_cols)
    right_upper_std = mean_of(right_upper_std_cols)
    left_lower_std = mean_of(left_lower_std_cols)
    right_lower_std = mean_of(right_lower_std_cols)
    dist_std = mean_of(dist_std_cols)

    # better：左右取更小的 mean，std 用对应那一侧
    def pick_better(l_mean, r_mean, l_std, r_std):
        if l_mean is None and r_mean is None:
            return None, None
        if r_mean is None or (l_mean is not None and l_mean <= r_mean):
            return l_mean, l_std
        return r_mean, r_std

    upper_better_mean, upper_better_std = pick_better(
        left_upper_mean, right_upper_mean, left_upper_std, right_upper_std
    )
    lower_better_mean, lower_better_std = pick_better(
        left_lower_mean, right_lower_mean, left_lower_std, right_lower_std
    )

    # latency / fps
    frames = to_float(r.get("frames") or r.get("frames_used"))
    latency_ms = to_float(r.get("latency_ms"))
    fps = to_float(r.get("fps") or r.get("fps_eq"))
    if fps is None and latency_ms and latency_ms > 0 and frames:
        fps = frames / (latency_ms / 1000.0)

    return {
        "exercise": exercise,
        "subject": subject,
        "view": view,
        "backend": backend,
        "upper_mean": round(
            (mean_of(left_upper_mean_cols) + mean_of(right_upper_mean_cols)) / 2, 2
        )
        if left_upper_mean is not None or right_upper_mean is not None
        else "",
        "upper_std": round(
            (mean_of(left_upper_std_cols) + mean_of(right_upper_std_cols)) / 2, 2
        )
        if left_upper_std is not None or right_upper_std is not None
        else "",
        "upper_better_mean": round(upper_better_mean, 2) if upper_better_mean is not None else "",
        "upper_better_std": round(upper_better_std, 2) if upper_better_std is not None else "",
        "lower_mean": round(
            (mean_of(left_lower_mean_cols) + mean_of(right_lower_mean_cols)) / 2, 2
        )
        if left_lower_mean is not None or right_lower_mean is not None
        else "",
        "lower_std": round(
            (mean_of(left_lower_std_cols) + mean_of(right_lower_std_cols)) / 2, 2
        )
        if left_lower_std is not None or right_lower_std is not None
        else "",
        "lower_better_mean": round(lower_better_mean, 2) if lower_better_mean is not None else "",
        "lower_better_std": round(lower_better_std, 2) if lower_better_std is not None else "",
        "dist_mean": round(dist_mean, 4) if dist_mean is not None else "",
        "dist_std": round(dist_std, 4) if dist_std is not None else "",
        "frames": int(frames) if frames is not None else "",
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else "",
        "fps": round(fps, 2) if fps is not None else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Exp#1 results into upper/lower angle + distance metrics."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("fit3d_results/exp1_model"),
        help="Directory containing exp1_model CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fit3d_results/exp1_model/summary_simple.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    all_rows: List[Dict[str, object]] = []

    # raw backends
    for name in ["mediapipe_2d", "mediapipe_3d", "movenet_3d", "hrnet2d"]:
        path = args.root / f"table_{name}.csv"
        for r in load_rows(path, backend_name=name):
            all_rows.append(summarize_row(r))

    # lifters
    lifter_path = args.root / "fit3d_lifter_model_vs_gt.csv"
    for backend in ["motionbert", "videopose1", "videopose243"]:
        for r in load_rows(lifter_path, backend_name=backend):
            all_rows.append(summarize_row(r))

    # ROMP
    romp_path = args.root / "table_romp3d.csv"
    for r in load_rows(romp_path, backend_name="romp3d"):
        all_rows.append(summarize_row(r))

    if not all_rows:
        print("[WARN] No rows found to summarize.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exercise",
        "subject",
        "view",
        "backend",
        "upper_mean",
        "upper_std",
        "upper_better_mean",
        "upper_better_std",
        "lower_mean",
        "lower_std",
        "lower_better_mean",
        "lower_better_std",
        "dist_mean",
        "dist_std",
        "frames",
        "latency_ms",
        "fps",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[INFO] wrote summary to {args.output}")


if __name__ == "__main__":
    main()
