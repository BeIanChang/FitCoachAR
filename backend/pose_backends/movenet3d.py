"""MoveNet 3D backend wrapper (requires optional TensorFlow dependencies)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from .base import PoseBackend

try:
    import tensorflow as tf  # type: ignore
except ImportError:
    tf = None  # type: ignore


class MoveNet3DBackend(PoseBackend):
    """Pose backend powered by Google's MoveNet 3D (TensorFlow Lite)."""

    name = "movenet_3d"
    dimension_hint = "3D"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if tf is None:
            raise RuntimeError(
                "MoveNet3DBackend requires TensorFlow to be installed. "
                "Install tensorflow>=2.12 and set MOVENET_3D_MODEL to a "
                "MoveNet 3D TFLite file."
            )

        model_path = os.getenv("MOVENET_3D_MODEL")
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(
                "Set MOVENET_3D_MODEL to the path of the MoveNet 3D TFLite model."
            )

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MoveNet 3D model from {model_path}: {exc}"
            ) from exc

        input_details = self.interpreter.get_input_details()
        if not input_details:
            raise RuntimeError("MoveNet 3D model exposes no inputs.")
        self.input_index = input_details[0]["index"]
        self.input_shape = input_details[0]["shape"]

        output_details = self.interpreter.get_output_details()
        if not output_details:
            raise RuntimeError("MoveNet 3D model exposes no outputs.")
        self.output_index = output_details[0]["index"]

        self.logger.info("Initialized MoveNet 3D backend with model %s", model_path)

    def handle_command(self, command_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """MoveNet backend currently does not support calibration commands."""
        return None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize frame for MoveNet inference."""
        # Most MoveNet variants expect 256x256 RGB normalized to [-1,1]
        import cv2

        height, width = self.input_shape[1], self.input_shape[2]
        resized = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        return np.expand_dims(normalized, axis=0)

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run MoveNet 3D inference and format the payload."""
        input_tensor = self._preprocess(frame_bgr)
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)

        # Model-specific post-processing is required here. Without the exact
        # MoveNet 3D signature, we cannot decode joints reliably, so we flag it.
        raise NotImplementedError(
            "MoveNet 3D backend is placeholders only. Supply decoding logic "
            "for the specific model outputs and map them to the FitCoachAR payload."
        )
