"""Standalone MoveNet inference microservice.

Run this inside the TensorFlow environment:

    python backend/services/movenet_service.py --model /path/to/movenet.tflite

The service exposes a single POST /infer endpoint that accepts a JSON body:
    {"frame": "<base64 JPEG>"}
and responds with:
    {"keypoints": [[y, x, score], ...], "score": float}
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request


try:
    import tensorflow as tf  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "tensorflow is required to run the MoveNet service. "
        "Install tensorflow-macos/tensorflow-metal inside this environment."
    ) from exc


def build_interpreter(model_path: str, fallback_size: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()[0]
    resize_shape = list(input_details["shape"])
    if len(resize_shape) == 4:
        if resize_shape[1] <= 1:
            resize_shape[1] = fallback_size
        if resize_shape[2] <= 1:
            resize_shape[2] = fallback_size
        if resize_shape != list(input_details["shape"]):
            interpreter.resize_tensor_input(input_details["index"], resize_shape)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details


def letterbox(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def preprocess(frame_bgr: np.ndarray, input_details, fallback_size: int) -> np.ndarray:
    height = int(input_details["shape"][1])
    width = int(input_details["shape"][2])
    if height <= 1 or width <= 1:
        height = width = fallback_size
    padded = letterbox(frame_bgr, height, width)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    dtype = input_details["dtype"]
    quant_params = input_details.get("quantization", (0.0, 0))
    scale, zero_point = quant_params if quant_params else (0.0, 0)

    if dtype == np.uint8:
        if scale and scale > 0:
            normalized = rgb.astype(np.float32) / 255.0
            quantized = normalized / scale + zero_point
            quantized = np.clip(np.round(quantized), 0, 255).astype(np.uint8)
            tensor = quantized
        else:
            tensor = rgb.astype(np.uint8)
    elif dtype == np.int32:
        tensor = rgb.astype(np.int32)
    else:
        tensor = (rgb.astype(np.float32) - 127.5) / 127.5

    return np.expand_dims(tensor, axis=0).astype(dtype)


def run_inference(
    interpreter: tf.lite.Interpreter,
    input_details,
    output_details,
    frame_bgr: np.ndarray,
    fallback_size: int,
) -> Tuple[np.ndarray, float]:
    input_tensor = preprocess(frame_bgr, input_details, fallback_size)
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])  # [1, 6, 56]
    detections = output[0]
    best_idx = np.argmax(detections[:, -1])
    best_score = detections[best_idx, -1]
    keypoints = detections[best_idx, : 17 * 3].reshape(17, 3)
    return keypoints, float(best_score)


class KalmanFilter2D:
    """Tiny Kalman filter for one joint (2D)."""

    def __init__(self, q: float = 1e-3, r: float = 5e-3):
        self.q = q
        self.r = r
        self.x: np.ndarray | None = None  # shape (2,)
        self.P: np.ndarray | None = None  # shape (2,2)

    def update(self, z: np.ndarray) -> np.ndarray:
        if self.x is None:
            self.x = z.copy()
            self.P = np.eye(2, dtype=np.float32)
            return z
        # predict: x = x, P = P + Q
        Q = np.eye(2, dtype=np.float32) * self.q
        R = np.eye(2, dtype=np.float32) * self.r
        self.P = self.P + Q
        # update
        K = self.P @ np.linalg.inv(self.P + R)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(2, dtype=np.float32) - K) @ self.P
        return self.x


def kalman_smooth(
    coords: np.ndarray,
    filters: Dict[int, KalmanFilter2D],
    base_q: float,
    base_r: float,
    conf: float,
    motion: float,
) -> np.ndarray:
    """Per-joint Kalman smoothing (2D state)."""
    out = coords.copy()
    q = base_q * (1.0 + 2.0 * motion)
    r = base_r * (1.0 + max(0.0, 1.0 - conf))
    for j in range(coords.shape[0]):
        f = filters.get(j)
        if f is None:
            f = KalmanFilter2D()
            filters[j] = f
        f.q = q
        f.r = r
        out[j] = f.update(coords[j])
    return out


def _yaw_compensate(coords: np.ndarray, ls: np.ndarray, rs: np.ndarray, lh: np.ndarray, rh: np.ndarray):
    """Rotate pelvis axis to horizontal, shoulder axis orthogonal-ish, then apply yaw scaling."""
    pelvis_vec = rh - lh
    shoulder_vec = rs - ls
    theta = np.arctan2(pelvis_vec[1], pelvis_vec[0])
    rot = np.array(
        [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]],
        dtype=np.float32,
    )
    coords = coords @ rot.T
    pelvis_vec = pelvis_vec @ rot.T
    shoulder_vec = shoulder_vec @ rot.T
    shoulder_w = float(np.linalg.norm(shoulder_vec))
    hip_w = float(np.linalg.norm(pelvis_vec))
    yaw_proxy = shoulder_w / hip_w if hip_w > 1e-6 else 1.0
    yaw_scale = float(np.clip(yaw_proxy, 0.7, 1.3))
    coords[:, 0] /= yaw_scale
    return coords, yaw_scale, shoulder_w, hip_w, rot


def _enforce_bone_lengths(coords: np.ndarray, bone_ema: Dict[Tuple[int, int], float], momentum: float):
    """Project limb bones toward symmetric lengths; keep a running EMA target per bone."""
    # COCO/MoveNet indices
    LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    limb_pairs = [
        ((LS, LE), (RS, RE)),  # upper arm
        ((LE, LW), (RE, RW)),  # lower arm
        ((LH, LK), (RH, RK)),  # thigh
        ((LK, LA), (RK, RA)),  # calf
    ]
    for (l_parent, l_child), (r_parent, r_child) in limb_pairs:
        lp, lc = coords[l_parent], coords[l_child]
        rp, rc = coords[r_parent], coords[r_child]
        l_vec = lc - lp
        r_vec = rc - rp
        l_len = float(np.linalg.norm(l_vec))
        r_len = float(np.linalg.norm(r_vec))
        if l_len < 1e-6 and r_len < 1e-6:
            continue
        raw_target = 0.5 * (l_len + r_len) if r_len > 1e-6 else l_len
        ema_key = (l_parent, l_child, r_parent, r_child)
        prev = bone_ema.get(ema_key, raw_target)
        target_len = momentum * prev + (1.0 - momentum) * raw_target
        bone_ema[ema_key] = target_len
        if l_len > 1e-6:
            coords[l_child] = lp + l_vec / l_len * target_len
        if r_len > 1e-6:
            coords[r_child] = rp + r_vec / r_len * target_len


def postprocess_keypoints(
    keypoints: np.ndarray,
    kalman_state: Dict[int, KalmanFilter2D],
    prev_raw: np.ndarray | None,
    bone_ema: Dict[Tuple[int, int], float],
    ema_momentum: float = 0.8,
    base_q: float = 1e-4,
    base_r: float = 5e-3,
) -> Tuple[np.ndarray, Dict[int, KalmanFilter2D], np.ndarray, Dict[Tuple[int, int], float]]:
    """
    Apply lightweight post-processing:
      - Kalman smoothing (confidence/motion adaptive)
      - body-centric canonicalization + yaw compensation
      - anatomical bone-length consistency (with per-bone EMA targets)
    Returns (processed_keypoints, new_kalman_state, prev_raw_coords, bone_ema)
    """
    coords = keypoints[:, :2].astype(np.float32)
    # motion estimate
    motion = float(np.nanmean(np.linalg.norm(coords - prev_raw, axis=1))) if prev_raw is not None else 0.0
    conf = float(np.nanmean(keypoints[:, 2]))
    # 1) Kalman smoothing
    coords = kalman_smooth(coords, kalman_state, base_q, base_r, conf, motion)

    # 2) body-centric canonicalization
    LS, RS, LH, RH = 5, 6, 11, 12
    torso_center = (coords[LS] + coords[RS] + coords[LH] + coords[RH]) / 4.0
    coords -= torso_center
    coords, yaw_scale, shoulder_w, hip_w, rot = _yaw_compensate(
        coords, coords[LS], coords[RS], coords[LH], coords[RH]
    )
    scale = max(1e-3, (shoulder_w + hip_w) * 0.5)
    coords /= scale

    # 3) anatomical consistency
    _enforce_bone_lengths(coords, bone_ema, ema_momentum)

    # 4) map back to normalized image space
    coords *= scale
    coords[:, 0] *= yaw_scale
    coords = coords @ rot  # rotate back
    coords += torso_center
    coords = np.clip(coords, 0.0, 1.0)

    processed = keypoints.copy()
    processed[:, :2] = coords
    return processed, kalman_state, coords, bone_ema


def create_app(model_path: str, fallback_size: int) -> Flask:
    app = Flask(__name__)
    interpreter, input_details, output_details = build_interpreter(model_path, fallback_size)
    kalman_state: Dict[int, KalmanFilter2D] = {}
    bone_ema: Dict[Tuple[int, int], float] = {}
    prev_raw: np.ndarray | None = None

    @app.route("/infer", methods=["POST"])
    def infer():
        start_time = time.time()
        payload = request.get_json(silent=True) or {}
        frame_b64 = payload.get("frame")
        if not frame_b64:
            return jsonify({"error": "frame field missing"}), 400

        if isinstance(frame_b64, str) and frame_b64.startswith("data:image"):
            frame_b64 = frame_b64.split(",", 1)[1]

        try:
            frame_bytes = base64.b64decode(frame_b64)
        except Exception:
            return jsonify({"error": "invalid base64 frame"}), 400

        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image data"}), 400

        keypoints, score = run_inference(
            interpreter, input_details, output_details, frame, fallback_size
        )
        nonlocal kalman_state
        nonlocal prev_raw, bone_ema
        processed, kalman_state, prev_raw, bone_ema = postprocess_keypoints(
            keypoints, kalman_state, prev_raw, bone_ema
        )
        app.logger.debug("Top keypoint sample: %s", keypoints[0])
        latency_ms = (time.time() - start_time) * 1000.0
        app.logger.debug("MoveNet inference latency %.2f ms, score %.3f", latency_ms, score)
        return jsonify(
            {
                "keypoints": processed.tolist(),
                "score": score,
                "latency_ms": latency_ms,
            }
        )

    @app.route("/reset", methods=["POST"])
    def reset():
        """Reset smoothing/bone stats to avoid cross-video contamination."""
        nonlocal kalman_state, bone_ema, prev_raw
        kalman_state = {}
        bone_ema = {}
        prev_raw = None
        return jsonify({"status": "reset"})

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="MoveNet inference service")
    parser.add_argument("--model", required=True, help="Path to MoveNet TFLite model file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--input-size",
        type=int,
        default=int(os.getenv("MOVENET_INPUT_SIZE", "256")),
        help="Fallback square input size when the model reports dynamic dims",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    app = create_app(args.model, args.input_size)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
