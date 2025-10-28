"""Pose backend registry for FitCoachAR.

Allows the WebSocket server to swap between different pose-processing
implementations (2D MediaPipe, future 3D, etc.) without touching the
transport layer.
"""

from typing import Dict, Type

from .base import PoseBackend
from .mediapipe2d import MediaPipe2DPoseBackend
from .mediapipe3d import MediaPipe3DBackend
from .mmpose_lifter import MMPosePoseLifterBackend
from .movenet3d import MoveNet3DBackend


BACKEND_REGISTRY: Dict[str, Type[PoseBackend]] = {
    MediaPipe2DPoseBackend.name: MediaPipe2DPoseBackend,
    MediaPipe3DBackend.name: MediaPipe3DBackend,
    MoveNet3DBackend.name: MoveNet3DBackend,
    MMPosePoseLifterBackend.name: MMPosePoseLifterBackend,
}


def get_available_backends():
    """Return the list of registered backend names."""
    return list(BACKEND_REGISTRY.keys())


def build_pose_backend(name: str) -> PoseBackend:
    """Instantiate a pose backend by registry name."""
    backend_cls = BACKEND_REGISTRY.get(name)
    if not backend_cls:
        raise ValueError(
            f"Unknown pose backend '{name}'. "
            f"Available options: {', '.join(get_available_backends())}"
        )
    return backend_cls()
