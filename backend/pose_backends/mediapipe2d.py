"""MediaPipe-based 2.5D pose backend."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from filters import KalmanLandmarkSmoother
from kinematics import AngularFeatureExtractor
from llm_feedback import LLMFeedbackGenerator
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
        self.reset_state(reset_calibration=True)

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

    def handle_command(self, command_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        command = command_data.get("command")
        exercise = command_data.get("exercise")
        if exercise:
            self.selected_exercise = exercise

        if command == "select_exercise":
            return None

        if command == "calibrate_down":
            self.arm_extended_angle = self.last_right_elbow_angle
            if self.landmarks:
                shoulder = np.array(
                    [
                        self.landmarks[
                            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                        ].x,
                        self.landmarks[
                            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                        ].y,
                    ]
                )
                elbow = np.array(
                    [
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                    ]
                )
                self.user_profile["upper_arm_length"] = np.linalg.norm(shoulder - elbow)
            self.logger.info(
                "Calibrated DOWN angle to: %.2f and Arm Length: %.4f",
                self.arm_extended_angle,
                self.user_profile.get("upper_arm_length") or 0,
            )
        elif command == "calibrate_up":
            self.arm_contracted_angle = self.last_right_elbow_angle
            self.logger.info("Calibrated UP angle to: %.2f", self.arm_contracted_angle)
        elif command == "calibrate_squat_down":
            self.squat_down_angle = self.last_right_knee_angle
            self.logger.info(
                "Calibrated SQUAT DOWN angle to: %.2f", self.squat_down_angle
            )
        elif command == "calibrate_squat_up":
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
            self.logger.info(
                "Calibrated SQUAT UP angle to: %.2f and Thigh Length: %.4f",
                self.squat_up_angle,
                self.user_profile.get("thigh_length") or 0,
            )
        elif command == "reset":
            summary = {
                "total_reps": self.total_reps,
                "mistakes": self.mistake_counter,
            }
            self.reset_state(reset_calibration=False)
            return {"summary": summary}

        return None

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
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

        # Landmark visibility check
        required_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        if self.selected_exercise == "bicep_curls":
            required_landmarks.extend(
                [
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                ]
            )
        elif self.selected_exercise == "squats":
            required_landmarks.extend(
                [
                    mp_pose.PoseLandmark.LEFT_KNEE,
                    mp_pose.PoseLandmark.RIGHT_KNEE,
                    mp_pose.PoseLandmark.LEFT_ANKLE,
                    mp_pose.PoseLandmark.RIGHT_ANKLE,
                ]
            )

        is_occluded = any(
            landmarks[lm.value].visibility < 0.7 for lm in required_landmarks
        )

        feedback = ""
        feedback_landmarks: List[int] = []

        if is_occluded:
            feedback = "Adjust camera to show full body"
        else:
            right_elbow_x = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x
            self.last_right_elbow_angle = right_elbow_angle
            self.last_right_knee_angle = right_knee_angle

            if self.curl_state == "DOWN":
                self.elbow_baseline_x = right_elbow_x
                if right_elbow_angle < (self.arm_contracted_angle + 20):
                    self.curl_state = "UP"
                feedback = ""
            elif self.curl_state == "UP":
                if right_elbow_angle > (self.arm_extended_angle - 20):
                    self.curl_state = "DOWN"
                    self.curl_counter += 1
                    self.total_reps += 1
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

            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

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

        data_to_send = {
            "landmarks": landmarks_data,
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

        self.last_processed_data = data_to_send
        return data_to_send

    def close(self) -> None:
        self.holistic.close()
