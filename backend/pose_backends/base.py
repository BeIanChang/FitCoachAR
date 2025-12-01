"""Common interfaces for FitCoachAR pose-processing backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PoseBackendResult:
    """Standard payload returned by pose backends for each processed frame."""

    landmarks: List[Dict[str, float]]
    payload: Dict[str, Any] = field(default_factory=dict)


class PoseBackend(ABC):
    """Abstract base class for pose-processing implementations."""

    name: str = "base"
    dimension_hint: str = "2D"

    def handle_command(self, command_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch external commands (calibration, reset, etc.)."""
        return None

    @abstractmethod
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a decoded BGR frame and return data to send to clients."""

    def close(self) -> None:
        """Release resources."""
        return None


class LM:
    def __init__(self, d: Dict[str, Any]):
        self.x = d["x"]
        self.y = d["y"]
        self.z = d.get("z", 0.0)
        self.visibility = d.get("visibility", 0.0)


class PoseEstimator(ABC):
    @abstractmethod
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a decoded BGR frame and return landmarks."""
    @abstractmethod
    def process_video(self, video_path: str) -> Optional[List[Dict[str, Any]]]:
        """Process a video file and return landmarks per frame."""
    @abstractmethod
    def landmark_dict(self) -> Dict[str, int]:
        """Return a mapping from landmark name to index in the output array/order."""
