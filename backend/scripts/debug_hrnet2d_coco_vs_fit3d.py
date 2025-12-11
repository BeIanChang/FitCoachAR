"""
Visualize HRNet2D keypoints on a COCO-style image and on one Fit3D frame.

Usage example (from FitCoachAR root, in an env that has mmpose installed):

  python3 -m backend.scripts.debug_hrnet2d_coco_vs_fit3d \
    --coco-image image.png \
    --fit3d-tag barbell_row__s03__50591643 \
    --fit3d-frame 200 \
    --hrnet-cfg ../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    --hrnet-ckpt ../mmpose/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.pth

This will produce:
  debug_hrnet_coco.png
  debug_hrnet_fit3d_barbell_row__s03__50591643_f200.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from mmpose.apis.inferencers import Pose2DInferencer


def draw_keypoints(frame: np.ndarray, kpts: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """Draw simple dots for each keypoint."""
    out = frame.copy()
    for x, y in kpts:
        cv2.circle(out, (int(x), int(y)), 3, color, -1)
    return out


def run_on_image(inferencer: Pose2DInferencer, image_bgr: np.ndarray, out_path: Path) -> None:
    """Run HRNet2D on a BGR image array and save overlay."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred = next(inferencer(img_rgb, return_datasamples=True, show=False))
    ds = pred["predictions"][0]
    kpts = np.asarray(ds.pred_instances.keypoints[0], dtype=float)[..., :2]

    vis = draw_keypoints(image_bgr, kpts)
    cv2.imwrite(str(out_path), vis)
    print(f"[INFO] saved overlay -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize HRNet2D keypoints on COCO image and Fit3D frame."
    )
    parser.add_argument(
        "--coco-image",
        type=Path,
        required=True,
        help="Path to a COCO-style RGB image (e.g., image.png).",
    )
    parser.add_argument(
        "--fit3d-tag",
        type=str,
        required=True,
        help="Tag of Fit3D clip, e.g., barbell_row__s03__50591643 (used to locate video via hrnet2d_kpts/<tag>/kpts2d_hrnet.json).",
    )
    parser.add_argument(
        "--fit3d-frame",
        type=int,
        default=200,
        help="0-based frame index within the Fit3D clip to visualize.",
    )
    parser.add_argument(
        "--hrnet-cfg",
        type=str,
        required=True,
        help="HRNet-W32 2D config path (from mmpose).",
    )
    parser.add_argument(
        "--hrnet-ckpt",
        type=str,
        required=True,
        help="HRNet-W32 2D checkpoint path (.pth).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("."),
        help="Directory to save debug images.",
    )
    args = parser.parse_args()

    inferencer = Pose2DInferencer(
        args.hrnet_cfg,
        args.hrnet_ckpt,
        device="cpu",
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    # 1) COCO image
    coco_bgr = cv2.imread(str(args.coco_image))
    if coco_bgr is None:
        raise SystemExit(f"Failed to read image: {args.coco_image}")
    run_on_image(inferencer, coco_bgr, args.out_root / "debug_hrnet_coco.png")

    # 2) Fit3D frame: locate video via hrnet2d_kpts JSON, then run inferencer on raw frame
    kpts_json = Path("hrnet2d_kpts") / args.fit3d_tag / "kpts2d_hrnet.json"
    if not kpts_json.exists():
        raise SystemExit(f"kpts2d_hrnet.json not found: {kpts_json}")
    import json

    with kpts_json.open() as f:
        data = json.load(f)
    video_path = Path(data["video"])

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.fit3d_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Failed to read frame {args.fit3d_frame} from {video_path}")

    run_on_image(
        inferencer,
        frame,
        args.out_root
        / f"debug_hrnet_fit3d_{args.fit3d_tag}_f{args.fit3d_frame}.png",
    )


if __name__ == "__main__":
    main()
