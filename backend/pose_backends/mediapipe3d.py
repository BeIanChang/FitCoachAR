"""MediaPipe 3D backend leveraging world landmark coordinates."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from base64 import b64encode
from calibration_store import CalibrationRecord, new_record, store
from filters import LandmarkSmoother
from kinematics import AngularFeatureExtractor
from llm_feedback import LLMFeedbackGenerator
from .base import PoseBackend


class MediaPipe3DBackend(PoseBackend):
    """Pose backend that operates on MediaPipe's 3D world landmarks."""

    name = "mediapipe_3d"
    dimension_hint = "3D"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            enable_segmentation=False,
        )
        self.world_smoother = LandmarkSmoother(window_length=5, polyorder=2)
        self.llm_feedback = LLMFeedbackGenerator(use_api=False)
        self.user_profile = {"upper_arm_length": None, "thigh_length": None}
        self.current_mode = "common"
        self.pending_calibration: Optional[Dict[str, Any]] = None
        self.calibration_session: Optional[Dict[str, Any]] = None
        self.last_frame_bgr: Optional[np.ndarray] = None
        self.reset_state(reset_calibration=True)
        self._apply_active_calibration("bicep_curls")
        self._apply_active_calibration("squats")

    def _required_joints(self, exercise: Optional[str]) -> List[int]:
        pose = self.mp_pose.PoseLandmark
        if exercise == "bicep_curls":
            return [
                pose.RIGHT_SHOULDER.value,
                pose.RIGHT_ELBOW.value,
                pose.RIGHT_WRIST.value,
            ]
        if exercise == "squats":
            return [
                pose.RIGHT_HIP.value,
                pose.RIGHT_KNEE.value,
                pose.RIGHT_ANKLE.value,
            ]
        return [pose.RIGHT_SHOULDER.value, pose.RIGHT_ELBOW.value]

    def reset_state(self, reset_calibration: bool = False):
        self.curl_counter = 0
        self.curl_state = "DOWN"
        self.squat_counter = 0
        self.squat_state = "UP"
        self.feedback = ""
        self.feedback_landmarks: List[int] = []
        self.selected_exercise: Optional[str] = None
        self.total_reps = 0
        self.mistake_counter: Dict[str, int] = {}
        self.last_right_elbow_angle = 0.0
        self.last_right_knee_angle = 0.0
        self.last_plausible_left_elbow = 0.0
        self.last_plausible_right_elbow = 0.0
        self.last_plausible_left_knee = 0.0
        self.last_plausible_right_knee = 0.0
        self.elbow_baseline_x = 0.0
        self.last_payload: Optional[Dict[str, Any]] = None
        self.world_landmarks = None
        self.visibility_landmarks = None

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
        self._apply_active_calibration("bicep_curls")
        self._apply_active_calibration("squats")
        self.calibration_session = None

    BICEP_CANONICAL = {"extended": 160.0, "contracted": 30.0}
    SQUAT_CANONICAL = {"up": 160.0, "down": 50.0}
    CALIBRATION_FREEZE_SECONDS = 2.0

    def _apply_active_calibration(self, exercise: str):
        record = store.get_active_record(exercise, self.current_mode)
        if record:
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
        success, buf = cv2.imencode(".jpg", frame)
        if not success:
            return None
        return b64encode(buf).decode("utf-8")

    def _apply_record(self, record: Any):
        if isinstance(record, CalibrationRecord):
            record_dict = record.to_dict()
        else:
            record_dict = record
        exercise = record_dict.get("exercise")
        angles = record_dict.get("angles", {})
        if exercise == "bicep_curls":
            self.arm_extended_angle = angles.get("extended", self.arm_extended_angle)
            self.arm_contracted_angle = angles.get("contracted", self.arm_contracted_angle)
        elif exercise == "squats":
            self.squat_up_angle = angles.get("up", self.squat_up_angle)
            self.squat_down_angle = angles.get("down", self.squat_down_angle)

    def _apply_default_angles(self, exercise: str):
        if exercise == "bicep_curls":
            self.arm_extended_angle = self.BICEP_CANONICAL["extended"]
            self.arm_contracted_angle = self.BICEP_CANONICAL["contracted"]
        elif exercise == "squats":
            self.squat_up_angle = self.SQUAT_CANONICAL["up"]
            self.squat_down_angle = self.SQUAT_CANONICAL["down"]

    def _finalize_bicep_calibration(self, mode: str, critic: float) -> Dict[str, Any]:
        if not self.pending_calibration:
            raise RuntimeError("No pending calibration to finalize.")
        angles = {
            "extended": float(self.pending_calibration["extended_angle"]),
            "contracted": float(self.last_right_elbow_angle),
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
            "down": float(self.last_right_knee_angle),
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

    def handle_command(self, command_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        command = command_data.get("command")
        exercise = command_data.get("exercise") or self.selected_exercise or "bicep_curls"
        self.selected_exercise = exercise

        if command == "select_exercise":
            self.pending_calibration = None
            return {
                "event": "exercise_selected",
                "exercise": exercise,
                "mode": self.current_mode,
                **store.to_summary(exercise),
            }

        if command == "set_mode":
            mode = command_data.get("mode", "common")
            if mode not in ("common", "calibration"):
                mode = "common"
            self.current_mode = mode
            if mode == "common":
                self.pending_calibration = None
                self.calibration_session = None
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
            mode = command_data.get("mode", "common")
            value = float(command_data.get("value", 0.2))
            store.set_critic(exercise, mode, value)
            return {
                "event": "critic_updated",
                "exercise": exercise,
                "mode": mode,
                "critics": store.get_critics(exercise),
            }

        if command == "list_calibrations":
            return {
                "event": "calibration_list",
                "exercise": exercise,
                "mode": self.current_mode,
                **store.to_summary(exercise),
            }

        if command == "use_calibration":
            mode = command_data.get("mode", "common")
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

        if command == "start_auto_calibration":
            self.calibration_session = {
                "exercise": exercise,
                "mode": self.current_mode,
                "min_angle": None,
                "max_angle": None,
                "min_image": None,
                "max_image": None,
                "frozen": False,
                "last_update": time.time(),
            }
            return {
                "event": "calibration_started",
                "exercise": exercise,
                "mode": self.current_mode,
            }

        if command == "finalize_auto_calibration":
            session = self.calibration_session
            if not session or session.get("exercise") != exercise:
                return {
                    "event": "calibration_error",
                    "message": "No active calibration session. Start a new calibration first.",
                }
            min_angle = session.get("min_angle")
            max_angle = session.get("max_angle")
            if min_angle is None or max_angle is None:
                return {
                    "event": "calibration_error",
                    "message": "Not enough movement detected. Perform a few full reps, then finish.",
                }

            if exercise == "bicep_curls":
                angles = {"extended": float(max_angle), "contracted": float(min_angle)}
                canonical = self.BICEP_CANONICAL
                eta = {
                    "extended": (angles["extended"] - canonical["extended"]) / canonical["extended"],
                    "contracted": (angles["contracted"] - canonical["contracted"]) / canonical["contracted"],
                }
                images = {
                    "extended": session.get("max_image"),
                    "contracted": session.get("min_image"),
                }
            else:
                angles = {"up": float(max_angle), "down": float(min_angle)}
                canonical = self.SQUAT_CANONICAL
                eta = {
                    "up": (angles["up"] - canonical["up"]) / canonical["up"],
                    "down": (angles["down"] - canonical["down"]) / canonical["down"],
                }
                images = {
                    "up": session.get("max_image"),
                    "down": session.get("min_image"),
                }

            critic = store.get_critics(exercise)[self.current_mode]
            record = new_record(
                exercise=exercise,
                mode=self.current_mode,
                angles=angles,
                eta=eta,
                canonical=canonical,
                critic=critic,
                images=images,
            )
            store.add_record(record)
            self.calibration_session = None
            self._apply_record(record)
            return {
                "event": "calibration_complete",
                "exercise": exercise,
                "mode": self.current_mode,
                "record": record.to_dict(),
            }

        if command == "cancel_calibration":
            self.calibration_session = None
            self.pending_calibration = None
            self.current_mode = "common"
            self.reset_state(reset_calibration=False)
            return {
                "event": "calibration_cancelled",
                "exercise": exercise,
            }

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
            record = self._finalize_bicep_calibration(self.current_mode, critic)
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
            record = self._finalize_squat_calibration(self.current_mode, critic)
            return {
                "event": "calibration_complete",
                "exercise": "squats",
                "mode": self.current_mode,
                "record": record,
            }

        if command == "reset":
            summary = {"total_reps": self.total_reps, "mistakes": self.mistake_counter}
            self.reset_state(reset_calibration=False)
            return {"summary": summary}

        return None

    def _world_coordinates(self, landmarks):
        coords = []
        for lm in landmarks:
            coords.append([lm.x, lm.y, lm.z])
        return coords

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        self.last_frame_bgr = frame_bgr.copy()
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        if not results.pose_world_landmarks or not results.pose_landmarks:
            self.last_payload = {"landmarks": []}
            return self.last_payload

        world_landmarks_raw = results.pose_world_landmarks.landmark
        smoothed_world = self.world_smoother.smooth(world_landmarks_raw)
        if not smoothed_world:
            return None

        image_landmarks = results.pose_landmarks.landmark
        self.world_landmarks = self._world_coordinates(smoothed_world)
        self.visibility_landmarks = image_landmarks

        mp_pose = self.mp_pose
        world_points = self.world_landmarks

        def world_point(idx):
            return world_points[idx]

        left_elbow_angle = AngularFeatureExtractor.compute_joint_angle(
            world_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            world_point(mp_pose.PoseLandmark.LEFT_ELBOW.value),
            world_point(mp_pose.PoseLandmark.LEFT_WRIST.value),
        )
        if 0 < left_elbow_angle < 180:
            self.last_plausible_left_elbow = left_elbow_angle
        else:
            left_elbow_angle = self.last_plausible_left_elbow

        right_elbow_angle = AngularFeatureExtractor.compute_joint_angle(
            world_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            world_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            world_point(mp_pose.PoseLandmark.RIGHT_WRIST.value),
        )
        if 0 < right_elbow_angle < 180:
            self.last_plausible_right_elbow = right_elbow_angle
        else:
            right_elbow_angle = self.last_plausible_right_elbow

        left_knee_angle = AngularFeatureExtractor.compute_joint_angle(
            world_point(mp_pose.PoseLandmark.LEFT_HIP.value),
            world_point(mp_pose.PoseLandmark.LEFT_KNEE.value),
            world_point(mp_pose.PoseLandmark.LEFT_ANKLE.value),
        )
        if 45 < left_knee_angle < 181:
            self.last_plausible_left_knee = left_knee_angle
        else:
            left_knee_angle = self.last_plausible_left_knee

        right_knee_angle = AngularFeatureExtractor.compute_joint_angle(
            world_point(mp_pose.PoseLandmark.RIGHT_HIP.value),
            world_point(mp_pose.PoseLandmark.RIGHT_KNEE.value),
            world_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
        )
        if 45 < right_knee_angle < 181:
            self.last_plausible_right_knee = right_knee_angle
        else:
            right_knee_angle = self.last_plausible_right_knee

        feedback = ""
        feedback_landmarks: List[int] = []

        def visible(idx):
            return (
                self.visibility_landmarks[idx].visibility
                if self.visibility_landmarks
                else 1.0
            )

        required_indices = self._required_joints(self.selected_exercise)
        essential_visible = all(visible(idx) >= 0.1 for idx in required_indices)

        if not essential_visible:
            feedback = "Adjust camera to show the active limb"
        else:
            right_elbow_position_2d = image_landmarks[
                mp_pose.PoseLandmark.RIGHT_ELBOW.value
            ]
            self.last_right_elbow_angle = right_elbow_angle
            self.last_right_knee_angle = right_knee_angle

            if self.curl_state == "DOWN":
                self.elbow_baseline_x = right_elbow_position_2d.x
                if right_elbow_angle < (self.arm_contracted_angle + 20):
                    self.curl_state = "UP"
                feedback = ""
            elif self.curl_state == "UP":
                if right_elbow_angle > (self.arm_extended_angle - 20):
                    self.curl_state = "DOWN"
                    print("Registered curl rep")
                    self.curl_counter += 1
                    self.total_reps += 1
                    feedback = ""
                else:
                    stability_threshold = (
                        self.user_profile["upper_arm_length"] * 0.15
                        if self.user_profile.get("upper_arm_length")
                        else 0.05
                    )
                    if abs(right_elbow_position_2d.x - self.elbow_baseline_x) > stability_threshold:
                        feedback = "Keep elbow stable!"
                        self.mistake_counter["elbow_stability"] = (
                            self.mistake_counter.get("elbow_stability", 0) + 1
                        )
                        feedback_landmarks = [
                            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                            mp_pose.PoseLandmark.RIGHT_WRIST.value,
                        ]
                    elif right_elbow_angle > (self.arm_contracted_angle + 20):
                        feedback = "Curl higher!"
                        self.mistake_counter["curl_depth"] = (
                            self.mistake_counter.get("curl_depth", 0) + 1
                        )
                        feedback_landmarks = [
                            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                            mp_pose.PoseLandmark.RIGHT_WRIST.value,
                        ]
                    else:
                        feedback = "Great curl!"

            right_hip_y = image_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            right_knee_y = image_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

            if self.squat_state == "UP":
                if right_knee_angle < (self.squat_down_angle + 20):
                    self.squat_state = "DOWN"
                if feedback not in ["Keep elbow stable!", "Curl higher!", "Great curl!"]:
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
                            mp_pose.PoseLandmark.RIGHT_HIP.value,
                            mp_pose.PoseLandmark.RIGHT_KNEE.value,
                            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                        ]

        if self.calibration_session and self.calibration_session.get("exercise") == (
            self.selected_exercise or "bicep_curls"
        ):
            if self.selected_exercise == "bicep_curls":
                self._update_calibration_extremes(right_elbow_angle)
            elif self.selected_exercise == "squats":
                self._update_calibration_extremes(right_knee_angle)

        angular_features = AngularFeatureExtractor.extract_all_features(smoothed_world)

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

        landmarks_payload = []
        for i, coord in enumerate(world_points):
            image_lm = image_landmarks[i]
            landmarks_payload.append(
                {
                    "x": image_lm.x,
                    "y": image_lm.y,
                    "z": coord[2],
                    "visibility": image_lm.visibility,
                }
            )

        data_to_send = {
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
            "kinematic_features": angular_features,
            "backend": self.name,
        }

        if self.calibration_session:
            data_to_send["calibration_progress"] = {
                "exercise": self.calibration_session.get("exercise"),
                "mode": self.calibration_session.get("mode"),
                "min_angle": self.calibration_session.get("min_angle"),
                "max_angle": self.calibration_session.get("max_angle"),
                "frozen": self.calibration_session.get("frozen", False),
            }

        self.last_payload = data_to_send
        return data_to_send

    def close(self) -> None:
        self.holistic.close()

    def _update_calibration_extremes(self, angle: float):
        """Track min/max angles and snapshots during auto calibration."""
        if angle <= 0:
            return
        session = self.calibration_session
        if session is None:
            return
        if session.get("frozen"):
            return

        now = time.time()
        if (
            session.get("min_angle") is not None
            and session.get("max_angle") is not None
            and session.get("last_update")
            and now - session["last_update"] > self.CALIBRATION_FREEZE_SECONDS
        ):
            session["frozen"] = True
            self.calibration_session = session
            return

        updated = False
        if session.get("min_angle") is None or angle < session["min_angle"]:
            session["min_angle"] = angle
            session["min_image"] = self._capture_snapshot()
            updated = True
        if session.get("max_angle") is None or angle > session["max_angle"]:
            session["max_angle"] = angle
            session["max_image"] = self._capture_snapshot()
            updated = True

        if updated:
            session["last_update"] = now
            self.calibration_session = session
