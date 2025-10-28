"""MediaPipe 3D backend leveraging world landmark coordinates."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            enable_segmentation=False,
        )
        self.world_smoother = LandmarkSmoother(window_length=5, polyorder=2)
        self.llm_feedback = LLMFeedbackGenerator(use_api=False)
        self.user_profile = {"upper_arm_length": None, "thigh_length": None}
        self.reset_state(reset_calibration=True)

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

    def handle_command(self, command_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        command = command_data.get("command")
        exercise = command_data.get("exercise")
        if exercise:
            self.selected_exercise = exercise

        if command == "select_exercise":
            return None

        if command == "calibrate_down":
            self.arm_extended_angle = self.last_right_elbow_angle
            if self.world_landmarks is not None:
                shoulder = np.array(
                    self.world_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                )
                elbow = np.array(
                    self.world_landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                )
                self.user_profile["upper_arm_length"] = np.linalg.norm(shoulder - elbow)
            self.logger.info(
                "Calibrated DOWN angle to: %.2f (arm length %.4f)",
                self.arm_extended_angle,
                self.user_profile.get("upper_arm_length") or 0.0,
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
            if self.world_landmarks is not None:
                hip = np.array(
                    self.world_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                )
                knee = np.array(
                    self.world_landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
                )
                self.user_profile["thigh_length"] = np.linalg.norm(hip - knee)
            self.logger.info(
                "Calibrated SQUAT UP angle to: %.2f (thigh length %.4f)",
                self.squat_up_angle,
                self.user_profile.get("thigh_length") or 0.0,
            )
        elif command == "reset":
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

        is_occluded = any(visible(lm.value) < 0.7 for lm in required_landmarks)

        if is_occluded:
            feedback = "Adjust camera to show full body"
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

        self.last_payload = data_to_send
        return data_to_send

    def close(self) -> None:
        self.holistic.close()
