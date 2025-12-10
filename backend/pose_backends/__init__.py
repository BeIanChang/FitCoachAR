"""Pose backend registry for FitCoachAR with graceful dependency handling."""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Type

from .base import PoseBackend

logger = logging.getLogger(__name__)

BACKEND_REGISTRY: Dict[str, Type[PoseBackend]] = {}


def _register(module_name: str, class_name: str) -> None:
    """Attempt to import and register a backend class."""
    try:
        module = importlib.import_module(f".{module_name}", __name__)
        backend_cls: Type[PoseBackend] = getattr(module, class_name)
        BACKEND_REGISTRY[backend_cls.name] = backend_cls
    except Exception as exc:
        logger.warning("Skipping backend %s.%s: %s", module_name, class_name, exc)


_register("mediapipe2d", "MediaPipe2DPoseBackend")
_register("mediapipe3d", "MediaPipe3DBackend")
_register("movenet3d", "MoveNet3DBackend")
_register("mmpose_lifter", "MMPosePoseLifterBackend")
_register("hrnet2d", "HRNet2DPoseBackend")


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
