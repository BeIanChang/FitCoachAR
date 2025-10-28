"""MMPose PoseLifter backend (requires optional PyTorch + MMPose)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from .base import PoseBackend

try:
    from mmpose.apis import inference_pose_lifter_model, init_pose_model  # type: ignore
except ImportError:
    inference_pose_lifter_model = None  # type: ignore
    init_pose_model = None  # type: ignore


class MMPosePoseLifterBackend(PoseBackend):
    """Pose backend that lifts 2D joints to 3D using MMPose PoseLifter."""

    name = "mmpose_poselifter"
    dimension_hint = "3D"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if inference_pose_lifter_model is None or init_pose_model is None:
            raise RuntimeError(
                "MMPosePoseLifterBackend requires the mmpose package. "
                "Install mmpose and its dependencies, then set MMPOSE_CONFIG "
                "and MMPOSE_CHECKPOINT to the PoseLifter config/checkpoint."
            )

        config_path = os.getenv("MMPOSE_CONFIG")
        checkpoint_path = os.getenv("MMPOSE_CHECKPOINT")

        if not config_path or not os.path.exists(config_path):
            raise RuntimeError("Set MMPOSE_CONFIG to the PoseLifter config file path.")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise RuntimeError(
                "Set MMPOSE_CHECKPOINT to the PoseLifter checkpoint file path."
            )

        try:
            self.model = init_pose_model(config_path, checkpoint_path, device="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize PoseLifter model with config {config_path}: {exc}"
            ) from exc

        self.logger.info(
            "Initialized MMPose PoseLifter backend (config=%s)", config_path
        )

    def handle_command(self, command_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """PoseLifter backend currently does not support calibration commands."""
        return None

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run PoseLifter inference."""
        # PoseLifter expects 2D detections per person, usually obtained from
        # a preceding 2D detector. Integrating that end-to-end is non-trivial,
        # so we raise to indicate further work is needed.
        raise NotImplementedError(
            "Integrate a 2D detector (e.g., MMPose TopDown) and pass sequences "
            "of 2D joints into inference_pose_lifter_model. Map the 3D joints "
            "back to the FitCoachAR payload format."
        )
