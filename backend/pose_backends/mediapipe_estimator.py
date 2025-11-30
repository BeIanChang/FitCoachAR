"""
MediaPipe-based 2D pose estimator implementing the generic PoseEstimator interface.

Provides:
- process_frame: 2D landmarks (x, y, z=0, visibility) for a single BGR frame.
- process_video: 2D landmarks for each frame in a video file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from .base import PoseEstimator, LM


class MediaPipePoseEstimator(PoseEstimator):
    """Thin wrapper around MediaPipe Pose for 2D landmark extraction."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        model_complexity: int = 1,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _landmarks_to_dicts(self, landmarks) -> List[LM]:
        data = []
        for lm in landmarks:
            a = {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(0.0),
                    "visibility": float(getattr(lm, "visibility", 0.0)),
                }
            data.append(LM(a))
        return data

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[List[LM]]:
        """Return pose landmarks for a single BGR frame, or None if not detected."""
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        return self._landmarks_to_dicts(results.pose_landmarks.landmark)

    def process_video(
        self, video_path: str, max_frames: Optional[int] = None
    ) -> Optional[List[List[LM]]]:
        """
        Run pose estimation over a full video.

        Returns a list of per-frame landmark lists; frames with no detection yield None entries.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        outputs: List[Optional[List[Dict[str, Any]]]] = []
        try:
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and count >= max_frames:
                    break
            count += 1
            outputs.append(self.process_frame(frame))
        finally:
            cap.release()
        return outputs

    def landmark_dict(self) -> Dict[str, int]:
        """Name-to-index mapping following MediaPipe PoseLandmark enumeration."""
        pl = self.mp_pose.PoseLandmark
        return {name: getattr(pl, name).value for name in dir(pl) if name.isupper()}

    def close(self) -> None:
        self.pose.close()
