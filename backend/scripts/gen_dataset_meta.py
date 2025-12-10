"""
Generate a lightweight meta CSV for the custom `dataset/` videos.

Folder layout expected:
  dataset/<exercise>/<view>/<video_file>
Where view can be eye, up45, down45, etc., and inside each view there are
five files like front.MOV, front45.MOV, side.MOV, back45.MOV, back.MOV.

Output CSV columns (similar to fit3d gt_meta):
  exercise,subject,view,video,gt_reps,start_frame,end_frame
video is a path relative to the dataset root.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate meta CSV for custom dataset/")
    parser.add_argument("--root", type=Path, default=Path("dataset"), help="Dataset root")
    parser.add_argument(
        "--subject", type=str, default="diy", help="Subject ID to write into the CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_meta.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows = []
    for exercise_dir in sorted(args.root.iterdir()):
        if not exercise_dir.is_dir():
            continue
        exercise = exercise_dir.name
        for view_dir in sorted(exercise_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            view = view_dir.name
            for video_path in sorted(view_dir.iterdir()):
                if not video_path.is_file():
                    continue
                if video_path.suffix.lower() not in {".mp4", ".mov"}:
                    continue
                rel = video_path.relative_to(args.root)
                rows.append(
                    {
                        "exercise": exercise,
                        "subject": args.subject,
                        "view": view,
                        "video": str(rel),
                        "gt_reps": "",
                        "start_frame": "",
                        "end_frame": "",
                    }
                )

    if not rows:
        print(f"[WARN] No videos found under {args.root}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exercise",
                "subject",
                "view",
                "video",
                "gt_reps",
                "start_frame",
                "end_frame",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
