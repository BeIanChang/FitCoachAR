"""
Offline evaluation with ground-truth repetition counts.

This script is similar to `offline_eval.py`, but instead of自动根据目录
扫描所有视频，它依赖一个带标注信息的 CSV：

  video,exercise,gt_reps,start_frame,end_frame,subject,view,pitch,yaw,take

必需字段：
  - video: 视频相对 `--video-root` 的路径，或者绝对路径
  - exercise: 动作名称（必须能被后端识别，例如 bicep_curl / squat / push_up）
  - gt_reps: 该片段的 ground truth 次数

可选字段：
  - start_frame / end_frame: 只在给定帧区间内做评估（1-based，闭区间）
  - subject / view / pitch / yaw / take: 仅用于分组分析，原样写回结果表

使用示例（在仓库根目录）：

  python -m backend.scripts.offline_eval_gt \\
    --meta-csv fit3d_subset/gt_meta.csv \\
    --video-root fit3d_subset/train \\
    --backends mediapipe_2d mediapipe_3d movenet3d \\
    --output-csv offline_eval_fit3d_gt.csv

然后可以用现有的 `analyze_offline_results.py` 或自己写脚本，
按 subject / view / backend 对比 rep 误差和角度/距离稳定性。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 允许从仓库根目录以 `python -m backend.scripts.offline_eval_gt` 运行
ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.offline_eval import VideoMeta, evaluate_video_with_backend  # type: ignore


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline evaluation with ground-truth reps for FitCoachAR pose backends."
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        required=True,
        help="CSV 文件，包含 video/exercise/gt_reps 等标注信息。",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("."),
        help="所有 video 相对路径的根目录（默认当前目录）。",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["mediapipe_2d", "mediapipe_3d", "movenet3d"],
        help="要评估的 backend 名称列表（需与 registry 中一致）。",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("offline_eval_gt_results.csv"),
        help="输出结果 CSV 路径。",
    )
    args = parser.parse_args()

    # 读取 meta CSV
    metas: List[Dict[str, str]] = []
    with args.meta_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"video", "exercise", "gt_reps"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"meta CSV 缺少必要列: {', '.join(sorted(missing))}")
        for row in reader:
            metas.append(row)

    if not metas:
        print(f"[WARN] meta CSV {args.meta_csv} 为空，没有任何样本。")
        return

    rows_out: List[Dict[str, object]] = []

    for row in metas:
        video_str = (row.get("video") or "").strip()
        if not video_str:
            print("[WARN] 跳过一行：video 为空", file=sys.stderr)
            continue

        video_path = Path(video_str)
        if not video_path.is_absolute():
            video_path = (args.video_root / video_path).resolve()

        exercise = (row.get("exercise") or "").strip()
        if not exercise:
            print(f"[WARN] 跳过 {video_str}: exercise 为空", file=sys.stderr)
            continue

        try:
            gt_reps = int((row.get("gt_reps") or "").strip())
        except Exception:
            print(f"[WARN] 跳过 {video_str}: gt_reps 非法", file=sys.stderr)
            continue

        start_frame = _parse_int(row.get("start_frame"))
        end_frame = _parse_int(row.get("end_frame"))

        # 这些字段如果存在，就带回去，便于后续分析
        subject = row.get("subject") or ""
        view = row.get("view") or ""
        pitch = row.get("pitch") or ""
        yaw = row.get("yaw") or ""
        take = row.get("take") or ""

        meta = VideoMeta(
            path=video_path,
            exercise=exercise,
            pitch=pitch,
            yaw=yaw,
            take=take,
        )

        for backend_name in args.backends:
            print(f"[INFO] Evaluating {video_path.name} (gt={gt_reps}) with backend={backend_name} ...")
            try:
                eval_row = evaluate_video_with_backend(
                    meta,
                    backend_name,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            except Exception as exc:
                print(f"[WARN] Failed on {video_path} with {backend_name}: {exc}", file=sys.stderr)
                eval_row = {
                    "video": str(video_path),
                    "exercise": exercise,
                    "pitch": pitch,
                    "yaw": yaw,
                    "take": take,
                    "backend": backend_name,
                    "frames": 0,
                    "avg_latency_ms": None,
                    "fps": None,
                    "curl_counter": None,
                    "squat_counter": None,
                }

            # 选用哪个 counter 与 gt 对比：
            pred_reps: Optional[int]
            curl = eval_row.get("curl_counter")
            squat = eval_row.get("squat_counter")
            if isinstance(curl, int):
                pred_reps = curl
            elif isinstance(squat, int):
                pred_reps = squat
            else:
                pred_reps = None

            if pred_reps is None:
                rep_error = None
                abs_rep_error = None
            else:
                rep_error = pred_reps - gt_reps
                abs_rep_error = abs(rep_error)

            out_row = dict(eval_row)  # type: ignore[arg-type]
            out_row.update(
                {
                    "subject": subject,
                    "view": view,
                    "gt_reps": gt_reps,
                    "pred_reps": pred_reps,
                    "rep_error": rep_error,
                    "abs_rep_error": abs_rep_error,
                }
            )
            rows_out.append(out_row)

    # 确定输出字段：沿用 offline_eval 的字段，再附加 GT 相关字段
    base_fields = [
        "video",
        "exercise",
        "pitch",
        "yaw",
        "take",
        "backend",
        "frames",
        "avg_latency_ms",
        "fps",
        "curl_counter",
        "squat_counter",
        "elbow_mean",
        "elbow_std",
        "elbow_mean_abs_delta",
        "left_elbow_mean",
        "left_elbow_std",
        "left_elbow_mean_abs_delta",
        "knee_mean",
        "knee_std",
        "knee_mean_abs_delta",
        "left_knee_mean",
        "left_knee_std",
        "left_knee_mean_abs_delta",
        "shl_mean",
        "shl_std",
        "shl_mean_abs_delta",
        "rshl_rpalm_mean",
        "rshl_rpalm_std",
        "rshl_rpalm_mean_abs_delta",
        "rshl_rhip_mean",
        "rshl_rhip_std",
        "rshl_rhip_mean_abs_delta",
        "rpalm_rhip_mean",
        "rpalm_rhip_std",
        "rpalm_rhip_mean_abs_delta",
        "rknee_rhip_mean",
        "rknee_rhip_std",
        "rknee_rhip_mean_abs_delta",
        "rknee_rfeet_mean",
        "rknee_rfeet_std",
        "rknee_rfeet_mean_abs_delta",
        "rhip_rfeet_mean",
        "rhip_rfeet_std",
        "rhip_rfeet_mean_abs_delta",
    ]
    extra_fields = ["subject", "view", "gt_reps", "pred_reps", "rep_error", "abs_rep_error"]
    fieldnames = base_fields + extra_fields

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"[INFO] Evaluation with GT written to {args.output_csv}")


if __name__ == "__main__":
    main()
