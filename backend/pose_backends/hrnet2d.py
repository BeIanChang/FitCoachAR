"""HRNet-W32 2D pose backend using MMPose.

该 backend 仅用于离线实验：
  - 使用 MMPose 的 HRNet-W32 top-down 模型对每帧图像做 2D 关键点检测；
  - 在 COCO-17 关节点上计算肘 / 膝关节角度和若干距离特征；
  - 返回的字段与 offline_eval / offline_eval_gt 使用的键一致，便于
    和其它 backend（MediaPipe2D/3D, MoveNet3D 等）对齐。

需要环境变量指定 MMPose 配置与 checkpoint：

  export MMPOSE_2D_CONFIG=/path/to/td-hm_hrnet-w32_..._coco-256x192.py
  export MMPOSE_2D_CHECKPOINT=/path/to/td-hm_hrnet-w32_..._coco-256x192.pth
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import PoseBackend


class HRNet2DPoseBackend(PoseBackend):
    """2D pose backend built on MMPose HRNet-W32."""

    name = "hrnet2d"
    dimension_hint = "2D"

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # 延迟导入 mmpose，并确保可以从本仓库旁边的 mmpose 源码目录加载。
        try:
            import sys
            from mmengine.config import Config  # type: ignore

            # 若本地有克隆的 mmpose 仓库，则优先使用
            root = Path(__file__).resolve().parents[2]
            mmpose_repo = root.parent / "mmpose"
            if mmpose_repo.exists() and str(mmpose_repo) not in sys.path:
                sys.path.insert(0, str(mmpose_repo))

            from mmpose.apis.inference import (  # type: ignore
                inference_topdown,
                init_model,
            )
        except Exception as exc:  # pragma: no cover - 依赖缺失时提示
            raise RuntimeError(
                "HRNet2DPoseBackend requires mmpose >=1.0 and mmengine. "
                "Please install mmpose in the current environment."
            ) from exc

        self._init_model = init_model
        self._inference_topdown = inference_topdown

        config_path = os.getenv("MMPOSE_2D_CONFIG")
        checkpoint_path = os.getenv("MMPOSE_2D_CHECKPOINT")

        if not config_path or not os.path.exists(config_path):
            raise RuntimeError(
                "Set MMPOSE_2D_CONFIG to the HRNet-W32 config file path "
                "(e.g., td-hm_hrnet-w32_8xb64-210e_coco-256x192.py)."
            )
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise RuntimeError(
                "Set MMPOSE_2D_CHECKPOINT to the HRNet-W32 checkpoint "
                "file path (.pth)."
            )

        try:
            # 使用 CPU 即可，用于离线实验
            self.model = self._init_model(config_path, checkpoint_path, device="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize HRNet2D model with config {config_path}: {exc}"
            ) from exc

        self.logger.info(
            "Initialized HRNet2D backend (config=%s)", config_path
        )

    def handle_command(self, command_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 2D backend 不参与校准逻辑，忽略所有命令
        return None

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Compute angle at b (in degrees) formed by points a-b-c."""
        v1 = a - b
        v2 = c - b
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0.0 or v2_norm == 0.0:
            return float("nan")
        cos_theta = float(np.dot(v1, v2) / (v1_norm * v2_norm))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return float(np.degrees(np.arccos(cos_theta)))

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:  # type: ignore[override]
        """Run HRNet-W32 2D inference on a single frame and compute metrics."""
        # 推理 2D pose
        try:
            data_samples = self._inference_topdown(self.model, frame_bgr)
        except Exception:
            return None

        if not data_samples:
            return None

        # 只取第一个人的结果（Fit3D / 自录视频里基本只有一人）
        sample = data_samples[0]
        try:
            from mmpose.structures import PoseDataSample  # type: ignore

            if isinstance(sample, PoseDataSample):
                kpts = sample.pred_instances.keypoints[0]  # [J,2]
            else:
                # 兼容旧版本字典格式
                kpts = sample["pred_instances"]["keypoints"][0]
        except Exception:
            # 回退到通用访问方式
            try:
                kpts = sample.pred_instances.keypoints[0]  # type: ignore[attr-defined]
            except Exception:
                return None

        kpts = np.asarray(kpts, dtype=float)  # [J,2]
        if kpts.shape[0] < 17:
            return None

        # COCO-17 关键点索引
        nose = 0
        l_sh, r_sh = 5, 6
        l_el, r_el = 7, 8
        l_wr, r_wr = 9, 10
        l_hp, r_hp = 11, 12
        l_kn, r_kn = 13, 14
        l_an, r_an = 15, 16

        p_ls = kpts[l_sh]
        p_rs = kpts[r_sh]
        p_le = kpts[l_el]
        p_re = kpts[r_el]
        p_lw = kpts[l_wr]
        p_rw = kpts[r_wr]
        p_lh = kpts[l_hp]
        p_rh = kpts[r_hp]
        p_lk = kpts[l_kn]
        p_rk = kpts[r_kn]
        p_la = kpts[l_an]
        p_ra = kpts[r_an]

        # 角度（在图像平面上）
        right_elbow_angle = self._angle(p_rs, p_re, p_rw)
        left_elbow_angle = self._angle(p_ls, p_le, p_lw)
        right_knee_angle = self._angle(p_rh, p_rk, p_ra)
        left_knee_angle = self._angle(p_lh, p_lk, p_la)

        # 距离特征（与 compute_fit3d_gt_metrics 中的定义保持近似）
        shl = float(np.linalg.norm(p_ls - p_rs))
        rshl_rpalm = float(p_rs[1] - p_rw[1])
        rknee_rhip = float(p_rh[1] - p_rk[1])
        rhip_rfeet = float(p_ra[1] - p_rh[1])

        return {
            "right_elbow_angle": float(right_elbow_angle),
            "left_elbow_angle": float(left_elbow_angle),
            "right_knee_angle": float(right_knee_angle),
            "left_knee_angle": float(left_knee_angle),
            "metric_shl_dist": shl,
            "metric_rshl_rpalm": rshl_rpalm,
            "metric_rknee_rhip": rknee_rhip,
            "metric_rhip_rfeet": rhip_rfeet,
            "curl_counter": None,
            "squat_counter": None,
        }

