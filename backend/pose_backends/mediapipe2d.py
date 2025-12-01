"""MediaPipe-based 2.5D pose backend."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from filters import KalmanLandmarkSmoother
from kinematics import AngularFeatureExtractor
from llm_feedback import LLMFeedbackGenerator
from calibration_store import store, new_record, CalibrationRecord
from session import WorkoutSession
from .base import PoseBackend


def calculate_angle(a, b, c):
    """Calculate the angle at point b formed by points a-b-c."""
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


class MediaPipe2DPoseBackend(PoseBackend):
    """Default pose backend built on MediaPipe Holistic (2.5D)."""

    name = "mediapipe_2d"
    dimension_hint = "2.5D"

    BICEP_CANONICAL = {"extended": 160.0, "contracted": 30.0}
    SQUAT_CANONICAL = {"up": 160.0, "down": 50.0}
    CALIBRATION_FREEZE_SECONDS = 2.0

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.landmark_smoother = KalmanLandmarkSmoother()
        self.llm_feedback = LLMFeedbackGenerator(use_api=False)
        self.user_profile = {
            "upper_arm_length": None,
            "thigh_length": None,
        }
        self.current_mode = "common"
        self.pending_calibration: Optional[Dict[str, Any]] = None
        self.calibration_session: Optional[Dict[str, Any]] = None
        self.last_frame_bgr: Optional[np.ndarray] = None
        self.reset_state(reset_calibration=True)
        self._apply_active_calibration("bicep_curls")
        self._apply_active_calibration("squats")
        
        # Session management
        self.active_session: Optional[WorkoutSession] = None

    def _apply_active_calibration(self, exercise: str):
        record = store.get_active_record(exercise, self.current_mode)
        if not record:
            return
        angles = record.get("angles", {})
        if exercise == "bicep_curls":
            self.arm_extended_angle = angles.get("extended", self.arm_extended_angle)
            self.arm_contracted_angle = angles.get("contracted", self.arm_contracted_angle)
        elif exercise == "squats":
            self.squat_up_angle = angles.get("up", self.squat_up_angle)
            self.squat_down_angle = angles.get("down", self.squat_down_angle)

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

    def _required_joints(self, exercise: Optional[str]) -> List[int]:
        mp_pose = self.mp_pose
        if exercise == "bicep_curls":
            return [
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                mp_pose.PoseLandmark.RIGHT_WRIST.value,
            ]
        if exercise == "squats":
            return [
                mp_pose.PoseLandmark.RIGHT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_KNEE.value,
                mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            ]
        if exercise in ("push_up", "lateral_raise", "barbell_row"):
            return [
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                mp_pose.PoseLandmark.RIGHT_WRIST.value,
                mp_pose.PoseLandmark.RIGHT_HIP.value,
            ]
        return [
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        ]

    def reset_state(self, reset_calibration: bool = False):
        self.curl_counter = 0
        self.curl_state = "DOWN"
        self.squat_counter = 0
        self.squat_state = "UP"
        self.feedback = ""
        self.llm_feedback_message = ""
        self.feedback_landmarks: List[int] = []
        self.last_processed_data: Optional[Dict[str, Any]] = None
        self.selected_exercise: Optional[str] = None

        self.last_plausible_left_elbow = 0
        self.last_plausible_right_elbow = 0
        self.last_plausible_left_knee = 0
        self.last_plausible_right_knee = 0
        self.last_right_elbow_angle = 0
        self.last_right_knee_angle = 0
        self.elbow_baseline_x = 0

        self.landmarks = None
        self.total_reps = 0
        self.rep_timestamps: List[float] = []
        self.mistake_counter: Dict[str, int] = {}

        if reset_calibration:
            self.arm_extended_angle = 160
            self.arm_contracted_angle = 30
            self.squat_up_angle = 160
            self.squat_down_angle = 50
        else:
            # Preserve personalized calibration between workouts.
            self.arm_extended_angle = getattr(self, "arm_extended_angle", 160)
            self.arm_contracted_angle = getattr(self, "arm_contracted_angle", 30)
            self.squat_up_angle = getattr(self, "squat_up_angle", 160)
            self.squat_down_angle = getattr(self, "squat_down_angle", 50)
        # Cancel any in-flight auto calibration when state resets.
        self.calibration_session = None

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
            summary = {
                "total_reps": self.total_reps,
                "mistakes": self.mistake_counter,
            }
            self.reset_state(reset_calibration=False)
            return {"summary": summary}

        if command == "start_auto_calibration":
            # Begin a streaming calibration session that captures angle extremes.
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
            if self.landmarks:
                shoulder = np.array(
                    [
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                    ]
                )
                elbow = np.array(
                    [
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                    ]
                )
                self.user_profile["upper_arm_length"] = np.linalg.norm(shoulder - elbow)
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
            if self.landmarks:
                hip = np.array(
                    [
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                    ]
                )
                knee = np.array(
                    [
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                    ]
                )
                self.user_profile["thigh_length"] = np.linalg.norm(hip - knee)
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

        # ==================== SESSION COMMANDS ====================
        
        if command == "start_session":
            # Start a new workout session
            # Payload: {"command": "start_session", "sets": [
            #   {"exercise": "squats", "reps": 10},
            #   {"exercise": "bicep_curls", "reps": 12}
            # ], "name": "My Workout"}
            sets_config = command_data.get("sets", [])
            session_name = command_data.get("name", "Workout")
            
            if not sets_config:
                return {
                    "event": "session_error",
                    "message": "No sets provided. Include 'sets' array with exercise and reps."
                }
            
            # Validate exercises
            valid_exercises = {"bicep_curls", "squats"}
            for s in sets_config:
                if s.get("exercise") not in valid_exercises:
                    return {
                        "event": "session_error",
                        "message": f"Invalid exercise: {s.get('exercise')}. Valid: {valid_exercises}"
                    }
            
            self.active_session = WorkoutSession.from_config(sets_config, name=session_name)
            self.active_session.start()
            
            # Set the current exercise and reset counters
            if self.active_session.current_exercise:
                self.selected_exercise = self.active_session.current_exercise
            self.reset_state(reset_calibration=False)
            self._apply_active_calibration(self.selected_exercise)
            
            return {
                "event": "session_started",
                "session": self.active_session.to_dict(),
                "progress": self.active_session.get_progress(),
            }
        
        if command == "next_set":
            # Advance to the next set in the session
            if not self.active_session:
                return {
                    "event": "session_error",
                    "message": "No active session. Start a session first."
                }
            
            next_set = self.active_session.advance_to_next_set()
            
            if next_set:
                # Switch to the new exercise
                self.selected_exercise = next_set.exercise
                self.reset_state(reset_calibration=False)
                self._apply_active_calibration(self.selected_exercise)
                
                return {
                    "event": "set_started",
                    "session": self.active_session.to_dict(),
                    "progress": self.active_session.get_progress(),
                }
            else:
                # Session complete
                return {
                    "event": "session_complete",
                    "session": self.active_session.to_dict(),
                    "progress": self.active_session.get_progress(),
                }
        
        if command == "skip_set":
            # Skip current set and move to next
            if not self.active_session:
                return {
                    "event": "session_error",
                    "message": "No active session."
                }
            
            next_set = self.active_session.skip_current_set()
            
            if next_set:
                self.selected_exercise = next_set.exercise
                self.reset_state(reset_calibration=False)
                self._apply_active_calibration(self.selected_exercise)
                
                return {
                    "event": "set_skipped",
                    "session": self.active_session.to_dict(),
                    "progress": self.active_session.get_progress(),
                }
            else:
                return {
                    "event": "session_complete",
                    "session": self.active_session.to_dict(),
                    "progress": self.active_session.get_progress(),
                }
        
        if command == "get_session_progress":
            # Get current session status
            if not self.active_session:
                return {
                    "event": "session_progress",
                    "active": False,
                    "progress": None,
                }
            
            return {
                "event": "session_progress",
                "active": True,
                "progress": self.active_session.get_progress(),
            }
        
        if command == "end_session":
            # End the current session early
            if not self.active_session:
                return {
                    "event": "session_error",
                    "message": "No active session to end."
                }
            
            self.active_session.finish()
            summary = self.active_session.to_dict()
            self.active_session = None
            self.reset_state(reset_calibration=False)
            
            return {
                "event": "session_ended",
                "session": summary,
            }

        return None

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        self.last_frame_bgr = frame_bgr.copy()
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        if not results.pose_landmarks:
            self.last_processed_data = {"landmarks": []}
            return self.last_processed_data

        smoothed_landmarks = self.landmark_smoother.smooth(
            results.pose_landmarks.landmark
        )
        if not smoothed_landmarks:
            return None

        self.landmarks = smoothed_landmarks

        mp_pose = self.mp_pose
        landmarks = smoothed_landmarks

        # Extract coordinates for angle calculation
        shoulder_left = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
        ]
        elbow_left = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
        ]
        wrist_left = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
        ]

        shoulder_right = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        ]
        elbow_right = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
        ]
        wrist_right = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
        ]

        left_elbow_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
        if 0 < left_elbow_angle < 180:
            self.last_plausible_left_elbow = left_elbow_angle
        else:
            left_elbow_angle = self.last_plausible_left_elbow

        right_elbow_angle = calculate_angle(shoulder_right, elbow_right, wrist_right)
        if 0 < right_elbow_angle < 180:
            self.last_plausible_right_elbow = right_elbow_angle
        else:
            right_elbow_angle = self.last_plausible_right_elbow

        hip_left = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        knee_left = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        ankle_left = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        left_knee_angle = calculate_angle(hip_left, knee_left, ankle_left)
        if 45 < left_knee_angle < 181:
            self.last_plausible_left_knee = left_knee_angle
        else:
            left_knee_angle = self.last_plausible_left_knee

        hip_right = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
        knee_right = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
        ankle_right = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]
        right_knee_angle = calculate_angle(hip_right, knee_right, ankle_right)
        if 45 < right_knee_angle < 181:
            self.last_plausible_right_knee = right_knee_angle
        else:
            right_knee_angle = self.last_plausible_right_knee

        # Require only essential joints for current exercise
        required_indices = self._required_joints(self.selected_exercise)
        essential_visible = all(
            landmarks[idx].visibility >= 0.1 for idx in required_indices
        )

        feedback = ""
        feedback_landmarks: List[int] = []

        if not essential_visible:
            feedback = "Adjust camera to show the active limb"
        else:
            right_elbow_x = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x
            self.last_right_elbow_angle = right_elbow_angle
            self.last_right_knee_angle = right_knee_angle

            exercise = self.selected_exercise or "bicep_curls"

            # Upper-body reps (all mapped to curl_counter)
            if exercise in ("bicep_curls", "push_up", "lateral_raise", "barbell_row"):
                if exercise == "lateral_raise":
                    # Use shoulder abduction: compare wrist vs shoulder height
                    shoulder_y = shoulder_right[1]
                    wrist_y = wrist_right[1]
                    # DOWN: hand clearly below shoulder; UP: hand at or above shoulder
                    in_up_position = wrist_y <= shoulder_y - 0.05
                elif exercise == "barbell_row":
                    # Use elbow flexion + horizontal displacement
                    # Treat as curl-like: flexed when elbow angle small
                    in_up_position = right_elbow_angle < (self.arm_contracted_angle + 20)
                else:  # bicep_curls or push_up
                    in_up_position = right_elbow_angle < (self.arm_contracted_angle + 20)

                if self.curl_state == "DOWN":
                    self.elbow_baseline_x = right_elbow_x
                    if in_up_position:
                        self.curl_state = "UP"
                    feedback = ""
                elif self.curl_state == "UP":
                    # Consider "extended" when angle back near extended baseline
                    in_down_position = right_elbow_angle > (self.arm_extended_angle - 20)
                    if in_down_position:
                        self.curl_state = "DOWN"
                        self.curl_counter += 1
                        print("rep counted")
                        self.total_reps += 1
                        self.rep_timestamps.append(time.time())  # Record timestamp for tempo
                        # Track rep in active session
                        if self.active_session and self.active_session.current_set:
                            self.active_session.record_rep()
                        feedback = ""
                    else:
                        stability_threshold = (
                            self.user_profile["upper_arm_length"] * 0.15
                            if self.user_profile.get("upper_arm_length")
                            else 0.05
                        )
                        if abs(right_elbow_x - self.elbow_baseline_x) > stability_threshold:
                            feedback = "Keep elbow stable!"
                            self.mistake_counter["elbow_stability"] = (
                                self.mistake_counter.get("elbow_stability", 0) + 1
                            )
                            feedback_landmarks = [
                                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                                mp_pose.PoseLandmark.RIGHT_WRIST.value,
                            ]
                        elif not in_up_position:
                            feedback = "Reach full range!"
                            self.mistake_counter["upper_rom"] = (
                                self.mistake_counter.get("upper_rom", 0) + 1
                            )
                            feedback_landmarks = [
                                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                                mp_pose.PoseLandmark.RIGHT_WRIST.value,
                            ]
                        else:
                            feedback = "Good rep!"

            # Squat reps (mapped to squat_counter)
            if exercise == "squats":
                right_hip_y = hip_right[1]
                right_knee_y = knee_right[1]

                if self.squat_state == "UP":
                    if right_knee_angle < (self.squat_down_angle + 20):
                        self.squat_state = "DOWN"
                    if feedback not in ["Keep elbow stable!", "Curl higher!", "Great curl!", "Good rep!"]:
                        feedback = ""
                elif self.squat_state == "DOWN":
                    if right_knee_angle > (self.squat_up_angle - 20):
                        self.squat_state = "UP"
                        self.squat_counter += 1
                        self.total_reps += 1
                        self.rep_timestamps.append(time.time())  # Record timestamp for tempo
                        # Track rep in active session
                        if self.active_session and self.active_session.current_set:
                            self.active_session.record_rep()
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

        # Distance-based metrics (for analysis / signal-based counting)
        shl_dx = shoulder_left[0] - shoulder_right[0]
        shl_dy = shoulder_left[1] - shoulder_right[1]
        shl_dist = float((shl_dx ** 2 + shl_dy ** 2) ** 0.5)

        rshl_rpalm_dist = float(shoulder_right[1] - wrist_right[1])
        rshl_rhip_dist = float(hip_right[1] - shoulder_right[1])
        rpalm_rhip_dist = float(wrist_right[1] - hip_right[1])
        rknee_rhip_dist = float(knee_right[1] - hip_right[1])
        rknee_rfeet_dist = float(ankle_right[1] - knee_right[1])
        rhip_rfeet_dist = float(ankle_right[1] - hip_right[1])

        # Update auto-calibration session with new extremes.
        if self.calibration_session and self.calibration_session.get("exercise") == (
            self.selected_exercise or "bicep_curls"
        ):
            if self.selected_exercise == "bicep_curls":
                self._update_calibration_extremes(right_elbow_angle)
            elif self.selected_exercise == "squats":
                self._update_calibration_extremes(right_knee_angle)

        angular_features = AngularFeatureExtractor.extract_all_features(landmarks)

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

        landmarks_data = [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            for lm in landmarks
        ]

        # Build arrow feedback for visual coaching
        arrow_feedback = []
        if feedback and feedback not in ["", "Good rep!", "Great curl!", "Good depth!", "Adjust camera to show full body"]:
            if self.selected_exercise == "bicep_curls":
                if "elbow" in feedback.lower():
                    # Elbow drifting - arrow pointing inward to body
                    arrow_feedback.append({
                        "joint_idx": 14,  # Right elbow
                        "type": "elbow_stability",
                        "direction": "left",  # Point toward body
                        "color": "#facc15"  # Yellow
                    })
                elif "range" in feedback.lower() or "higher" in feedback.lower():
                    # Need to curl higher - arrow pointing up
                    arrow_feedback.append({
                        "joint_idx": 16,  # Right wrist
                        "type": "curl_higher",
                        "direction": "up",
                        "color": "#facc15"
                    })
            elif self.selected_exercise == "squats":
                if "deeper" in feedback.lower():
                    # Need to squat deeper - arrow pointing down on hip
                    arrow_feedback.append({
                        "joint_idx": 24,  # Right hip
                        "type": "squat_deeper",
                        "direction": "down",
                        "color": "#facc15"
                    })

        data_to_send = {
            "landmarks": landmarks_data,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "curl_counter": self.curl_counter,
            "squat_counter": self.squat_counter,
            "rep_timestamps": self.rep_timestamps,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            # Distance metrics (2D, image-space)
            "metric_shl_dist": shl_dist,
            "metric_rshl_rpalm": rshl_rpalm_dist,
            "metric_rshl_rhip": rshl_rhip_dist,
            "metric_rpalm_rhip": rpalm_rhip_dist,
            "metric_rknee_rhip": rknee_rhip_dist,
            "metric_rknee_rfeet": rknee_rfeet_dist,
            "metric_rhip_rfeet": rhip_rfeet_dist,
            "feedback": feedback,
            "llm_feedback": llm_message,
            "feedback_landmarks": feedback_landmarks,
            "arrow_feedback": arrow_feedback,  # New: structured arrow data
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

        # Include session progress if active
        if self.active_session:
            data_to_send["session_progress"] = self.active_session.get_progress()

        self.last_processed_data = data_to_send
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

        updated = False
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
