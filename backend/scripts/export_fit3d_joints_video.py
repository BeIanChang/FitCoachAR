"""
Export a simple keypoint-only video from Fit3D joints3d_25.

Goal: 帮助你通过“看视频”来识别每个关节点的 index，而不是一帧一帧看静态图。

功能：
  - 读取 joints3d_25/<exercise>.json
  - 选定一个投影平面（xy/xz/yz），把 3D 点投到 2D
  - 做归一化缩放到固定分辨率画布上
  - 每帧画 25 个点 + 索引编号
  - 输出一个 mp4 视频，方便用播放器逐帧看

用法示例（在仓库根目录）：

  python -m backend.scripts.export_fit3d_joints_video \\
    --joints fit3d_subset/train/s03/joints3d_25/dumbbell_biceps_curls.json \\
    --output debug_biceps_s03_xz.mp4 \\
    --proj xz \\
    --fps 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def project_points(points: np.ndarray, proj: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    选择一个平面把 3D 点投到 2D：
      - xy: (x, y)
      - xz: (x, z)
      - yz: (y, z)
    """
    if proj == "xy":
        return points[:, 0], points[:, 1]
    if proj == "xz":
        return points[:, 0], points[:, 2]
    if proj == "yz":
        return points[:, 1], points[:, 2]
    if proj == "zy":
        return points[:, 2], points[:, 1]
    raise ValueError(f"Unknown proj mode: {proj}")


def export_video(
    joints_path: Path,
    output_path: Path,
    proj: str = "xz",
    fps: int = 25,
    size: int = 640,
) -> None:
    with joints_path.open() as f:
        data = json.load(f)

    frames = np.array(data["joints3d_25"], dtype=float)  # [T, 25, 3]
    T = frames.shape[0]

    # 先把所有帧投到 2D，算出整体的 min/max，这样视频里人物不会跳来跳去
    all_x = []
    all_y = []
    for t in range(T):
        xs, ys = project_points(frames[t], proj)
        all_x.append(xs)
        all_y.append(ys)
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())

    # 给一点边距
    margin = size * 0.1
    draw_min_x, draw_max_x = x_min, x_max
    draw_min_y, draw_max_y = y_min, y_max

    def world_to_img(x: float, y: float) -> Tuple[int, int]:
        # 线性缩放到 [margin, size - margin]
        if draw_max_x == draw_min_x:
            sx = 0.5
        else:
            sx = (x - draw_min_x) / (draw_max_x - draw_min_x)
        if draw_max_y == draw_min_y:
            sy = 0.5
        else:
            sy = (y - draw_min_y) / (draw_max_y - draw_min_y)

        px = int(margin + sx * (size - 2 * margin))
        py = int(margin + sy * (size - 2 * margin))
        # 翻一下 y，让上方在画面上方
        py = size - py
        return px, py

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (size, size))

    for t in range(T):
        xs, ys = project_points(frames[t], proj)
        frame = np.zeros((size, size, 3), dtype=np.uint8)

        # 画骨架点
        for i, (x, y) in enumerate(zip(xs, ys)):
            px, py = world_to_img(float(x), float(y))
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(i),
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # 写入帧
        writer.write(frame)

    writer.release()
    print(f"[INFO] wrote {T} frames to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export joints3d_25 as keypoint-only video.")
    parser.add_argument(
        "--joints",
        type=Path,
        required=True,
        help="Path to joints3d_25 JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output mp4 path.",
    )
    parser.add_argument(
        "--proj",
        type=str,
        default="xz",
        choices=["xy", "xz", "yz", "zy"],
        help="Projection plane to use (xy/xz/yz/zy).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Output video resolution (square).",
    )
    args = parser.parse_args()

    export_video(args.joints, args.output, proj=args.proj, fps=args.fps, size=args.size)


if __name__ == "__main__":
    main()
