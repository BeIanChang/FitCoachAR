"""MoveNet backend delegating inference to an external service."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from calibration_store import store, new_record, CalibrationRecord
from .base import PoseBackend
from llm_feedback import LLMFeedbackGenerator


class MoveNet3DBackend(PoseBackend):
    """Pose backend that calls an external MoveNet inference service."""

    name = "movenet_3d"
    dimension_hint = "3D"
    BICEP_CANONICAL = {"extended": 160.0, "contracted": 30.0}
    SQUAT_CANONICAL = {"up": 160.0, "down": 50.0}

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.service_url = os.getenv("MOVENET_SERVICE_URL", "http://127.0.0.1:8502/infer")
        self.request_timeout = float(os.getenv("MOVENET_SERVICE_TIMEOUT", "2.0"))

        self.keypoint_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        self.index = {name: idx for idx, name in enumerate(self.keypoint_names)}

        self.current_mode = "common"
        self.pending_calibration: Optional[Dict[str, Any]] = None
        self.last_frame_bgr: Optional[np.ndarray] = None
        self._reset_state(reset_calibration=True)
        self.llm_feedback = LLMFeedbackGenerator(use_api=False)
        self._apply_active_calibration("bicep_curls")
        self._apply_active_calibration("squats")

    def _apply_active_calibration(self, exercise: str):
        record = store.get_active_record(exercise, self.current_mode)
        if not record:
            return
        self._apply_record(record)

    def _capture_snapshot(self) -> Optional[str]:
        if self.last_frame_bgr is None:
            return None
        frame = self.last_frame_bgr
        max_dim = 320
        h, w = frame.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return None
        return base64.b64encode(buffer).decode("utf-8")

    def _finalize_bicep_calibration(self, mode: str, critic: float) -> Dict[str, Any]:
        if not self.pending_calibration:
            raise RuntimeError("No pending calibration to finalize.")
        angles = {
            "extended": float(self.pending_calibration["extended_angle"]),
            "contracted": float(self.arm_contracted_angle),
        }
        canonical = self.BICEP_CANONICAL
        eta = {
            "extended": (angles["extended"] - canonical["extended"]) / canonical["extended"],
            "contracted": (angles["contracted"] - canonical["contracted"]) / canonical["contracted"],
        }
        record = new_record(
            exercise="bicep_curls",
            mode=mode,
            angles=angles,
            eta=eta,
            canonical=canonical,
            critic=critic,
            images={
                "extended": self.pending_calibration.get("extended_image"),
                "contracted": self._capture_snapshot(),
            },
        )
        store.add_record(record)
        self.pending_calibration = None
        self._apply_record(record)
        return record.to_dict()

    def _finalize_squat_calibration(self, mode: str, critic: float) -> Dict[str, Any]:
        if not self.pending_calibration:
            raise RuntimeError("No pending calibration to finalize.")
        angles = {
            "up": float(self.pending_calibration["up_angle"]),
            "down": float(self.squat_down_angle),
        }
        canonical = self.SQUAT_CANONICAL
        eta = {
            "up": (angles["up"] - canonical["up"]) / canonical["up"],
            "down": (angles["down"] - canonical["down"]) / canonical["down"],
        }
        record = new_record(
            exercise="squats",
            mode=mode,
            angles=angles,
            eta=eta,
            canonical=canonical,
            critic=critic,
            images={
                "up": self.pending_calibration.get("up_image"),
                "down": self._capture_snapshot(),
            },
        )
        store.add_record(record)
        self.pending_calibration = None
        self._apply_record(record)
        return record.to_dict()

    def _apply_record(self, record: Any):
        if isinstance(record, CalibrationRecord):
            record_dict = record.to_dict()
        else:
            record_dict = record
        exercise = record_dict.get("exercise")
        angles = record_dict.get("angles", {})
        if exercise == "bicep_curls":
            self.arm_extended_angle = angles.get("extended", getattr(self, "arm_extended_angle", 160))
            self.arm_contracted_angle = angles.get("contracted", getattr(self, "arm_contracted_angle", 30))
        elif exercise == "squats":
            self.squat_up_angle = angles.get("up", getattr(self, "squat_up_angle", 160))
            self.squat_down_angle = angles.get("down", getattr(self, "squat_down_angle", 50))

    def _apply_default_angles(self, exercise: str):
        if exercise == "bicep_curls":
            self.arm_extended_angle = self.BICEP_CANONICAL["extended"]
            self.arm_contracted_angle = self.BICEP_CANONICAL["contracted"]
        elif exercise == "squats":
            self.squat_up_angle = self.SQUAT_CANONICAL["up"]
            self.squat_down_angle = self.SQUAT_CANONICAL["down"]

    def _required_joints(self, exercise: Optional[str]) -> List[int]:
        if exercise == "bicep_curls":
            return [
                self.index["right_shoulder"],
                self.index["right_elbow"],
                self.index["right_wrist"],
            ]
        if exercise == "squats":
            return [
                self.index["right_hip"],
                self.index["right_knee"],
                self.index["right_ankle"],
            ]
        if exercise in ("push_up", "lateral_raise", "barbell_row"):
            return [
                self.index["right_shoulder"],
                self.index["right_elbow"],
                self.index["right_wrist"],
                self.index["right_hip"],
            ]
        return [self.index["right_shoulder"], self.index["right_elbow"]]

    def handle_command(self, command_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        command = command_data.get("command")
        exercise = command_data.get("exercise") or self.selected_exercise or "bicep_curls"
        self.selected_exercise = exercise

        if command == "select_exercise":
            self.pending_calibration = None
            summary = store.to_summary(exercise)
            return {
                "event": "exercise_selected",
                "exercise": exercise,
                "mode": self.current_mode,
                **summary,
            }

        if command == "set_mode":
            mode = command_data.get("mode", "common")
            if mode not in ("common", "calibration"):
                mode = "common"
            self.current_mode = mode
            if mode == "common":
                self.pending_calibration = None
            active = store.get_active_record(exercise, mode)
            if active:
                self._apply_record(active)
            return {
                "event": "mode_updated",
                "exercise": exercise,
                "mode": mode,
                "activeCalibration": active,
                "critics": store.get_critics(exercise),
            }

        if command == "set_critic":
            mode = command_data.get("mode", self.current_mode)
            value = float(command_data.get("value", 0.2))
            store.set_critic(exercise, mode, value)
            return {
                "event": "critic_updated",
                "exercise": exercise,
                "mode": mode,
                "critics": store.get_critics(exercise),
            }

        if command == "list_calibrations":
            summary = store.to_summary(exercise)
            return {
                "event": "calibration_list",
                "exercise": exercise,
                "mode": self.current_mode,
                **summary,
            }

        if command == "use_calibration":
            mode = command_data.get("mode", self.current_mode)
            record_id = command_data.get("record_id")
            if record_id:
                store.set_active_record(exercise, mode, record_id)
                record = store.get_active_record(exercise, mode)
                if record:
                    self._apply_record(record)
            else:
                store.set_active_record(exercise, mode, None)
                self._apply_default_angles(exercise)
                record = None
            return {
                "event": "calibration_applied",
                "exercise": exercise,
                "mode": mode,
                "activeCalibration": record,
            }

        if command == "delete_calibration":
            record_id = command_data.get("record_id")
            deleted_record = None
            if record_id:
                for entry in store.list_records(exercise):
                    if entry["id"] == record_id:
                        deleted_record = entry
                        break
                store.delete_record(exercise, record_id)
            if deleted_record and deleted_record.get("mode") == self.current_mode:
                active = store.get_active_record(exercise, self.current_mode)
                if active:
                    self._apply_record(active)
                else:
                    self._apply_default_angles(exercise)
            summary = store.to_summary(exercise)
            return {
                "event": "calibration_deleted",
                "exercise": exercise,
                "deleted_id": record_id,
                **summary,
            }

        if command == "reset":
            summary = {"total_reps": self.total_reps, "mistakes": self.mistake_counter}
            self._reset_state(reset_calibration=False)
            return {"summary": summary}

        if command == "calibrate_down":
            self.arm_extended_angle = self.last_right_elbow_angle
            self.pending_calibration = {
                "exercise": "bicep_curls",
                "mode": self.current_mode,
                "extended_angle": self.arm_extended_angle,
                "extended_image": self._capture_snapshot(),
            }
            return {
                "event": "calibration_stage",
                "stage": "extended",
                "exercise": "bicep_curls",
                "mode": self.current_mode,
                "angles": {"extended": self.arm_extended_angle},
            }

        if command == "calibrate_up":
            self.arm_contracted_angle = self.last_right_elbow_angle
            critic = store.get_critics("bicep_curls")[self.current_mode]
            try:
                record = self._finalize_bicep_calibration(self.current_mode, critic)
            except RuntimeError as exc:
                return {"event": "calibration_error", "message": str(exc)}
            return {
                "event": "calibration_complete",
                "exercise": "bicep_curls",
                "mode": self.current_mode,
                "record": record,
            }

        if command == "calibrate_squat_up":
            self.squat_up_angle = self.last_right_knee_angle
            self.pending_calibration = {
                "exercise": "squats",
                "mode": self.current_mode,
                "up_angle": self.squat_up_angle,
                "up_image": self._capture_snapshot(),
            }
            return {
                "event": "calibration_stage",
                "stage": "up",
                "exercise": "squats",
                "mode": self.current_mode,
                "angles": {"up": self.squat_up_angle},
            }

        if command == "calibrate_squat_down":
            self.squat_down_angle = self.last_right_knee_angle
            critic = store.get_critics("squats")[self.current_mode]
            try:
                record = self._finalize_squat_calibration(self.current_mode, critic)
            except RuntimeError as exc:
                return {"event": "calibration_error", "message": str(exc)}
            return {
                "event": "calibration_complete",
                "exercise": "squats",
                "mode": self.current_mode,
                "record": record,
            }

        return None

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        self.last_frame_bgr = frame_bgr.copy()
        try:
            detections = self._run_remote_inference(frame_bgr)
        except Exception as exc:
            self.logger.error("MoveNet service error: %s", exc)
            return {"landmarks": [], "backend": self.name}

        if detections is None:
            return {"landmarks": [], "backend": self.name}

        keypoints = detections["keypoints"]
        instance_score = detections.get("score", 0.0)
        if instance_score < -0.01:
            return {"landmarks": [], "backend": self.name}

        landmarks = self._build_landmarks(np.array(keypoints, dtype=np.float32))
        if not landmarks:
            return {"landmarks": [], "backend": self.name}

        return self._process_landmarks(landmarks)

    # ------------------------------------------------------------------ #
    # Remote inference
    # ------------------------------------------------------------------ #

    def _run_remote_inference(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        success, buffer = cv2.imencode(".jpg", frame_bgr)
        if not success:
            raise RuntimeError("Failed to encode frame for MoveNet service.")
        encoded = base64.b64encode(buffer).decode("utf-8")

        response = requests.post(
            self.service_url,
            json={"frame": encoded},
            timeout=self.request_timeout,
        )
        response.raise_for_status()

        data = response.json()
        keypoints = data.get("keypoints")
        if not keypoints:
            return None
        if len(keypoints) != 17:
            raise ValueError("MoveNet service returned unexpected number of keypoints.")
        return data

    # ------------------------------------------------------------------ #
    # Landmark & feedback logic (mirrors MediaPipe implementation)
    # ------------------------------------------------------------------ #

    def _build_landmarks(self, keypoints: np.ndarray):
        class Landmark:
            def __init__(self, x: float, y: float, z: float, visibility: float):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility

        landmarks: List[Landmark] = []
        for kp in keypoints:
            y, x, score = kp
            # Clamp to [0,1] to keep overlay sane
            x = float(np.clip(x, 0.0, 1.0))
            y = float(np.clip(y, 0.0, 1.0))
            score = float(np.clip(score, 0.0, 1.0))
            landmarks.append(Landmark(x, y, 0.0, score))
        return landmarks

    def _reset_state(self, reset_calibration: bool = False):
        self.curl_counter = 0
        self.curl_state = "DOWN"
        self.squat_counter = 0
        self.squat_state = "UP"
        self.feedback = ""
        self.feedback_landmarks: List[int] = []
        self.last_right_elbow_angle = 0.0
        self.last_right_knee_angle = 0.0
        self.last_plausible_right_elbow = 0.0
        self.last_plausible_left_elbow = 0.0
        self.last_plausible_right_knee = 0.0
        self.last_plausible_left_knee = 0.0
        self.elbow_baseline_x = 0.0
        self.total_reps = 0
        self.mistake_counter: Dict[str, int] = {}
        self.selected_exercise: Optional[str] = None

        if reset_calibration:
            self.arm_extended_angle = 160
            self.arm_contracted_angle = 30
            self.squat_up_angle = 160
            self.squat_down_angle = 50
        else:
            self.arm_extended_angle = getattr(self, "arm_extended_angle", 160)
            self.arm_contracted_angle = getattr(self, "arm_contracted_angle", 30)
            self.squat_up_angle = getattr(self, "squat_up_angle", 160)
            self.squat_down_angle = getattr(self, "squat_down_angle", 50)

    def _process_landmarks(self, landmarks):
        right_shoulder = self._point(landmarks, "right_shoulder")
        right_elbow = self._point(landmarks, "right_elbow")
        right_wrist = self._point(landmarks, "right_wrist")
        left_shoulder = self._point(landmarks, "left_shoulder")
        left_elbow = self._point(landmarks, "left_elbow")
        left_wrist = self._point(landmarks, "left_wrist")

        left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        if 0 < left_elbow_angle < 180:
            self.last_plausible_left_elbow = left_elbow_angle
        else:
            left_elbow_angle = self.last_plausible_left_elbow

        right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        if 0 < right_elbow_angle < 180:
            self.last_plausible_right_elbow = right_elbow_angle
        else:
            right_elbow_angle = self.last_plausible_right_elbow

        right_hip = self._point(landmarks, "right_hip")
        right_knee = self._point(landmarks, "right_knee")
        right_ankle = self._point(landmarks, "right_ankle")
        left_hip = self._point(landmarks, "left_hip")
        left_knee = self._point(landmarks, "left_knee")
        left_ankle = self._point(landmarks, "left_ankle")

        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        if 45 < left_knee_angle < 181:
            self.last_plausible_left_knee = left_knee_angle
        else:
            left_knee_angle = self.last_plausible_left_knee

        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        if 45 < right_knee_angle < 181:
            self.last_plausible_right_knee = right_knee_angle
        else:
            right_knee_angle = self.last_plausible_right_knee

        required_indices = self._required_joints(self.selected_exercise)
        essential_visible = all(
            landmarks[idx].visibility >= 0.01 for idx in required_indices
        )

        feedback = ""
        feedback_landmarks: List[int] = []

        if not essential_visible:
            feedback = "Adjust camera to show the active limb"
        else:
            right_elbow_x = landmarks[self.index["right_elbow"]].x
            self.last_right_elbow_angle = right_elbow_angle
            self.last_right_knee_angle = right_knee_angle

            exercise = self.selected_exercise or "bicep_curls"

            # Upper-body reps
            if exercise in ("bicep_curls", "push_up", "lateral_raise", "barbell_row"):
                if exercise == "lateral_raise":
                    shoulder_y = landmarks[self.index["right_shoulder"]].y
                    wrist_y = landmarks[self.index["right_wrist"]].y
                    in_up_position = wrist_y <= shoulder_y - 0.05
                elif exercise == "barbell_row":
                    in_up_position = right_elbow_angle < (self.arm_contracted_angle + 20)
                else:
                    in_up_position = right_elbow_angle < (self.arm_contracted_angle + 20)

                if self.curl_state == "DOWN":
                    self.elbow_baseline_x = right_elbow_x
                    if in_up_position:
                        self.curl_state = "UP"
                    feedback = ""
                elif self.curl_state == "UP":
                    in_down_position = right_elbow_angle > (self.arm_extended_angle - 20)
                    if in_down_position:
                        self.curl_state = "DOWN"
                        self.curl_counter += 1
                        self.total_reps += 1
                        feedback = ""
                    else:
                        stability_threshold = 0.05
                        if abs(right_elbow_x - self.elbow_baseline_x) > stability_threshold:
                            feedback = "Keep elbow stable!"
                            self.mistake_counter["elbow_stability"] = (
                                self.mistake_counter.get("elbow_stability", 0) + 1
                            )
                            feedback_landmarks = [
                                self.index["right_shoulder"],
                                self.index["right_elbow"],
                                self.index["right_wrist"],
                            ]
                        elif not in_up_position:
                            feedback = "Reach full range!"
                            self.mistake_counter["upper_rom"] = (
                                self.mistake_counter.get("upper_rom", 0) + 1
                            )
                            feedback_landmarks = [
                                self.index["right_shoulder"],
                                self.index["right_elbow"],
                                self.index["right_wrist"],
                            ]
                        else:
                            feedback = "Great curl!"

            # Squat reps
            if exercise == "squats":
                right_hip_y = landmarks[self.index["right_hip"]].y
                right_knee_y = landmarks[self.index["right_knee"]].y

                if self.squat_state == "UP":
                    if right_knee_angle < (self.squat_down_angle + 20):
                        self.squat_state = "DOWN"
                    if feedback not in ["Keep elbow stable!", "Curl higher!", "Great curl!", "Reach full range!"]:
                        feedback = ""
                elif self.squat_state == "DOWN":
                    if right_knee_angle > (self.squat_up_angle - 20):
                        self.squat_state = "UP"
                        self.squat_counter += 1
                        self.total_reps += 1
                        feedback = ""
                    else:
                        if right_hip_y > right_knee_y:
                            feedback = "Good depth!"
                        else:
                            feedback = "Go deeper!"
                            self.mistake_counter["squat_depth"] = (
                                self.mistake_counter.get("squat_depth", 0) + 1
                            )
                            feedback_landmarks = [
                                self.index["right_hip"],
                                self.index["right_knee"],
                                self.index["right_ankle"],
                            ]

        llm_message = ""
        if feedback and feedback not in ["Adjust camera to show full body"]:
            error_record = {
                "exercise": self.selected_exercise,
                "phase": self.curl_state
                if self.selected_exercise == "bicep_curls"
                else self.squat_state,
                "errors": [
                    {
                        "joint": "right_elbow"
                        if self.selected_exercise == "bicep_curls"
                        else "right_knee",
                        "deviation_deg": abs(right_elbow_angle - self.arm_contracted_angle)
                        if self.selected_exercise == "bicep_curls"
                        else abs(right_knee_angle - self.squat_down_angle),
                        "type": feedback.lower().replace(" ", "_"),
                    }
                ],
                "critic_level": 0.5,
                "user_style": "friendly",
            }
            llm_message = self.llm_feedback.generate_feedback(error_record)

        landmarks_payload = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }
            for lm in landmarks
        ]

        return {
            "landmarks": landmarks_payload,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "curl_counter": self.curl_counter,
            "squat_counter": self.squat_counter,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "feedback": feedback,
            "llm_feedback": llm_message,
            "feedback_landmarks": feedback_landmarks,
            "kinematic_features": {},
            "backend": self.name,
        }

    def _point(self, landmarks, name: str):
        idx = self.index[name]
        lm = landmarks[idx]
        return [lm.x, lm.y, lm.z]

    @staticmethod
    def _calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
