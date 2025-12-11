"""
Generate meta CSV for Fit3D subset with ground-truth reps.

假设目录结构为：

  fit3d_subset/train/sXX/
    rep_ann.json                      # 每个动作的关键帧标注
    videos/<cam_id>/<exercise>.mp4    # 同一主体的多视角视频

其中 rep_ann.json 里，像这样：

  {
    "dumbbell_biceps_curls": [105, 201, 312, 408, 512, 630],
    "squat": [...],
    ...
  }

我们简单地把：
  - start_frame = 列表第一个元素
  - end_frame   = 列表最后一个元素
  - gt_reps     = len(list) - 1        # 相邻关键帧之间视为一组动作

并对同一 subject 下所有 camera 复用这一段帧区间和 gt_reps，
输出成一个 meta CSV，用于 `offline_eval_gt.py`。

使用示例（从仓库根目录）：

  python -m backend.scripts.gen_fit3d_meta \\
    --fit3d-root fit3d_subset/train \\
    --output fit3d_subset/gt_meta.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


EXERCISE_MAP: Dict[str, str] = {
    # Fit3D 名称 -> 本项目内部动作名
    "dumbbell_biceps_curls": "bicep_curl",
    "squat": "squat",
    "pushup": "push_up",
    "side_lateral_raise": "lateral_raise",
    "barbell_row": "barbell_row",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate meta CSV with GT reps from Fit3D subset.")
    parser.add_argument(
        "--fit3d-root",
        type=Path,
        default=Path("fit3d_subset/train"),
        help="Fit3D 子集的 train 根目录（包含 sXX 子目录）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fit3d_subset/gt_meta.csv"),
        help="输出的 meta CSV 路径。",
    )
    args = parser.parse_args()

    subjects: List[Path] = [
        p for p in args.fit3d_root.iterdir() if p.is_dir()
    ]
    subjects.sort()

    rows: List[Dict[str, object]] = []

    for subj_dir in subjects:
        subject_id = subj_dir.name
        rep_ann_path = subj_dir / "rep_ann.json"
        if not rep_ann_path.exists():
            continue

        with rep_ann_path.open() as f:
            ann = json.load(f)

        videos_root = subj_dir / "videos"
        if not videos_root.is_dir():
            continue

        # cam_id 子目录
        cam_dirs = [p for p in videos_root.iterdir() if p.is_dir()]
        cam_dirs.sort()

        for fit3d_name, internal_ex in EXERCISE_MAP.items():
            frames = ann.get(fit3d_name)
            if not frames or len(frames) < 2:
                continue

            try:
                start_frame = int(frames[0])
                end_frame = int(frames[-1])
            except Exception:
                continue

            gt_reps = max(len(frames) - 1, 1)

            for cam_dir in cam_dirs:
                cam_id = cam_dir.name
                video_path = cam_dir / f"{fit3d_name}.mp4"
                if not video_path.exists():
                    continue

                # video 相对于 fit3d-root 的路径，让后续 --video-root 指向 train 即可
                rel_video = video_path.relative_to(args.fit3d_root)

                rows.append(
                    {
                        "video": str(rel_video).replace("\\", "/"),
                        "exercise": internal_ex,
                        "gt_reps": gt_reps,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "subject": subject_id,
                        "view": cam_id,
                        # pitch/yaw/take 暂时留空，如有需要可以手动补
                        "pitch": "",
                        "yaw": "",
                        "take": "",
                    }
                )

    # 写 CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import csv

    fieldnames = [
        "video",
        "exercise",
        "gt_reps",
        "start_frame",
        "end_frame",
        "subject",
        "view",
        "pitch",
        "yaw",
        "take",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[INFO] Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

