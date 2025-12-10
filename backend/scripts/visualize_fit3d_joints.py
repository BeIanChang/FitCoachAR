"""
Quick-and-dirty visualizer for Fit3D joints3d_25 indices.

Purpose: 帮你直接“看见”每个关节点的索引，然后手动标注
哪些 index 是 right_shoulder / right_elbow / right_wrist / ...

用法（在仓库根目录）：

  python -m backend.scripts.visualize_fit3d_joints \\
    --joints fit3d_subset/train/s03/joints3d_25/dumbbell_biceps_curls.json \\
    --frame 200 \\
    --proj xz \\
    --output joints_frame200_xz.png

参数说明：
  --joints : joints3d_25 的 json 路径
  --frame  : 使用第几帧（1-based），不写默认中间那一帧
  --proj   : 选择投影平面，xy / xz / yz，默认 xz
  --output : 如果指定，则保存 PNG 到文件；否则直接弹出窗口显示
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
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


def project_to_image(
    points_cam: np.ndarray,
    cam_params_path: Path,
    coord_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    把 joints3d_25 的 3D 关节点投到图像平面。

    coord_mode 有三种选择：
      - 'cam'         : joints3d_25 已经是相机坐标系 (Xc,Yc,Zc)
      - 'world_w2c'   : extrinsics.R, T 是 world->camera，先做 Xc = R Xw + T
      - 'world_c2w'   : extrinsics.R, T 是 camera->world，先做 Xc = R^T (Xw - T)

    然后统一用 intrinsics_wo_distortion 做 pinhole 投影：

        [u]   [fx  0 cx] [X/Z]
        [v] = [ 0 fy cy] [Y/Z]
    """
    with cam_params_path.open() as f:
        cam = json.load(f)

    fx, fy = cam["intrinsics_wo_distortion"]["f"]
    cx, cy = cam["intrinsics_wo_distortion"]["c"]

    pts = np.array(points_cam, dtype=float)

    if coord_mode == "world_w2c":
        R = np.array(cam["extrinsics"]["R"], dtype=float)
        T = np.array(cam["extrinsics"]["T"], dtype=float).reshape(3)
        pts = (R @ pts.T).T + T
    elif coord_mode == "world_c2w":
        R = np.array(cam["extrinsics"]["R"], dtype=float)
        T = np.array(cam["extrinsics"]["T"], dtype=float).reshape(3)
        # Xw = R Xc + T  =>  Xc = R^T (Xw - T)
        pts = (R.T @ (pts - T).T).T
    elif coord_mode == "cam":
        pass
    else:
        raise ValueError(f"Unknown coord_mode: {coord_mode}")

    z = pts[:, 2]
    # 避免除零
    z_safe = np.where(z == 0, 1e-6, z)
    x_norm = pts[:, 0] / z_safe
    y_norm = pts[:, 1] / z_safe

    u = fx * x_norm + cx
    v = fy * y_norm + cy
    return u, v


def visualize(
    joints_path: Path,
    frame_idx: int | None = None,
    proj: str = "xz",
    output: Path | None = None,
    video_path: Path | None = None,
    cam_params_path: Path | None = None,
    coord_mode: str = "cam",
) -> None:
    with joints_path.open() as f:
        data = json.load(f)

    frames = np.array(data["joints3d_25"], dtype=float)  # [T, 25, 3]
    T = frames.shape[0]

    if frame_idx is None:
        frame_idx = T // 2  # middle frame (0-based)
    else:
        frame_idx = max(0, min(T - 1, frame_idx))

    pts = frames[frame_idx]  # [25, 3]

    # 如果提供了 video + camera_parameters，就投影到原视频坐标并叠加。
    if video_path is not None and cam_params_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx+1} from {video_path}")

        h, w = frame.shape[:2]
        us, vs = project_to_image(pts, cam_params_path, coord_mode)

        # 画到原始 BGR 帧上
        for i, (u, v) in enumerate(zip(us, vs)):
            x = int(round(u))
            y = int(round(v))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(i),
                    (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # 转成 RGB 再用 matplotlib 显示/保存
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))
        plt.imshow(frame_rgb)
        plt.axis("off")
        plt.title(f"{video_path.name}  frame={frame_idx+1}")

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output, dpi=200, bbox_inches="tight")
            print(f"[INFO] Saved overlay visualization to {output}")
        else:
            plt.show()
        return

    # 否则，只在 3D 投影平面上画散点 + 索引
    xs, ys = project_points(pts, proj)

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c="blue")
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x, y, str(i), color="red", fontsize=8)

    plt.title(f"{joints_path.name}  frame={frame_idx+1}  proj={proj}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200)
        print(f"[INFO] Saved visualization to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Fit3D joints3d_25 indices.")
    parser.add_argument(
        "--joints",
        type=Path,
        required=True,
        help="Path to joints3d_25 JSON file.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="1-based frame index to visualize (default: middle frame).",
    )
    parser.add_argument(
        "--proj",
        type=str,
        default="xz",
        choices=["xy", "xz", "yz", "zy"],
        help="Projection plane to use (xy/xz/yz/zy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, save PNG to this path instead of showing a window.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional: path to corresponding RGB video (for overlay).",
    )
    parser.add_argument(
        "--cam-params",
        type=Path,
        default=None,
        help="Optional: path to camera_parameters JSON for this video.",
    )
    parser.add_argument(
        "--coord-mode",
        type=str,
        default="cam",
        choices=["cam", "world_w2c", "world_c2w"],
        help="Coordinate convention for joints3d_25 vs extrinsics.",
    )
    args = parser.parse_args()

    frame_idx_0 = args.frame - 1 if args.frame is not None else None
    visualize(
        args.joints,
        frame_idx_0,
        args.proj,
        args.output,
        video_path=args.video,
        cam_params_path=args.cam_params,
        coord_mode=args.coord_mode,
    )


if __name__ == "__main__":
    main()
