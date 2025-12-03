"""
Simplified pose processor using calibration_v2 + rep_counter_v2 + form_checker.
Implements the 'Orchestrator' pattern to separate Counting from Form Checking.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np

# Import the core logic components
from calibration_v2 import (
    CalibrationParams,
    calibrate_from_landmarks,
    CalibrationStore,
)
from rep_counter_v2 import NormalizedRepCounter
from form_checker import FormChecker
from kinematics import KinematicFeatureExtractor

from filters import KalmanLandmarkSmoother
from .base import PoseBackend, PoseEstimator


class PoseProcessor(PoseBackend):
    name = "pose_processor_2d"
    dimension_hint = "2D"

    def __init__(self, estimator: PoseEstimator):
        self.estimator = estimator
        self.joint_map = self.estimator.landmark_dict()
        self.landmark_smoother = KalmanLandmarkSmoother()

        # The Geometry Engine
        self.feature_extractor = KinematicFeatureExtractor()

        self.selected_exercise: str = "bicep_curls"
        self.current_mode: str = "common"
        self.critic_value: float = 0.5  # Default sensitivity (0.0 = Loose, 1.0 = Strict)

        # Calibration State
        self.calibration_session: Optional[Dict[str, Any]] = None
        self.calibration_buffer: List = []
        self.calibration_timestamps: List[float] = []
        self.calibration_progress: Optional[Dict[str, Any]] = None

        self.param_store = CalibrationStore()

        # The Logic Engines (Separated)
        self.counters: Dict[str, NormalizedRepCounter] = {}
        self.checkers: Dict[str, FormChecker] = {}

    def handle_command(self, command_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        command = command_data.get("command")
        print(command)
        exercise = command_data.get("exercise") or self.selected_exercise or "bicep_curls"
        self.selected_exercise = exercise

        if command == "select_exercise":
            records = self.param_store.get_all_records(exercise)
            active = self.param_store.get_active_params(exercise)

            # Load stored critic preference
            ex_params = self.param_store.get_exercise_params(exercise)
            self.critic_value = float(ex_params.get("critic", 0.5))

            if active and exercise not in self.counters:
                # Initialize both engines with the active calibration
                self.counters[exercise] = NormalizedRepCounter(active)
                self.checkers[exercise] = FormChecker(active)

            return {
                "event": "exercise_selected",
                "exercise": exercise,
                "mode": self.current_mode,
                "records": [r.to_dict() for r in records],
                "active": {"common": active.to_dict() if active else None},
                "critics": {"common": self.critic_value},
            }

        if command == "save_calibrations":
            raw_params = command_data.get("calibration_params")
            if not raw_params:
                return {"event": "calibration_error", "message": "Missing params"}

            params = CalibrationParams.from_dict(raw_params)
            critic = float(command_data.get("critic", 0.5))
            self.critic_value = critic

            # Save critic preference
            self.param_store.set_exercise_params(exercise, critic=critic, deviation=0.2)

            # Save record
            rec_id = self.param_store.add_calibration_record(exercise, params, make_active=True)

            # Update Engines
            self.counters[exercise] = NormalizedRepCounter(params)
            self.checkers[exercise] = FormChecker(params)

            return {
                "event": "calibration_saved",
                "exercise": exercise,
                "record": {"id": rec_id, "critic": critic},
                "active": {"common": rec_id}
            }

        if command == "use_workout":
            print(command_data)
            record_id = command_data.get("record_id")
            if not record_id:
                return {"event": "error", "message": "ID required"}

            # If frontend used `use_workout`, treat as applying for the 'common' (workout) slot
            if command == "use_workout":
                mode = "common"
            else:
                mode = command_data.get("mode", "calibration")

            self.param_store.set_active_record(exercise, record_id)
            params = self.param_store.get_active_params(exercise)

            if params:
                self.counters[exercise] = NormalizedRepCounter(params)
                self.checkers[exercise] = FormChecker(params)

            return {"event": "calibration_applied", "mode": mode, "activeCalibration": {"id": record_id}}

        if command == "set_critic":
            self.critic_value = float(command_data.get("value", 0.5))
            self.param_store.set_exercise_params(exercise, critic=self.critic_value, deviation=0.2)
            return {"event": "critic_updated", "value": self.critic_value}

        if command == "reset":
            if exercise in self.counters:
                self.counters[exercise].reset()
            return {"summary": {"total_reps": 0}}

        if command == "start_auto_calibration":
            self.calibration_session = {"exercise": exercise, "end_time": time.time() + 15.0}
            self.calibration_buffer = []
            self.calibration_timestamps = []
            self.calibration_progress = None
            return {"event": "calibration_started", "exercise": exercise}

        if command == "finalize_auto_calibration":
            if not self.calibration_session:
                return {"event": "error", "message": "No session"}

            landmarks_array = np.stack(self.calibration_buffer, axis=0)
            try:
                # Run the geometry analysis
                calib_result = calibrate_from_landmarks(
                    landmarks_array,
                    self.joint_map,
                    exercise=exercise,
                    smoothing_window=5,
                    fps=30
                )
            except Exception as e:
                return {"event": "calibration_error", "message": str(e)}

            params = calib_result.params
            rec_id = self.param_store.add_calibration_record(exercise, params, make_active=True)

            # Init Engines immediately
            self.counters[exercise] = NormalizedRepCounter(params)
            self.checkers[exercise] = FormChecker(params)

            self.calibration_session = None
            self.calibration_buffer = []

            return {
                "event": "calibration_complete",
                "exercise": exercise,
                "record": {"id": rec_id, "calibration_params": params.to_dict()}
            }

        if command == "cancel_calibration":
            self.calibration_session = None
            self.calibration_buffer = []
            return {"event": "calibration_cancelled"}

        if command == "set_mode":
            self.current_mode = command_data.get("mode", "common")
            return {"event": "mode_updated", "mode": self.current_mode}

        return None

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        # 1. Pose Estimation
        raw_landmarks = self.estimator.process_frame(frame_bgr)
        if not raw_landmarks: return {"landmarks": []}

        smoothed_objs = self.landmark_smoother.smooth(raw_landmarks)
        if not smoothed_objs: return {"landmarks": []}

        smoothed_landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in smoothed_objs]

        # 2. Kinematic Analysis (Extract Metrics)
        # Returns { "primary": float, "form": { "elbow_drift": float, ... } }
        metrics = self.feature_extractor.extract_metrics(smoothed_landmarks, self.joint_map, self.selected_exercise)

        payload = {
            "landmarks": smoothed_landmarks,
            "exercise": self.selected_exercise,
            "feedback": "",
            "rep_count": 0,
            # Include exercise-specific counters expected by the frontend
            "rep_phase": "BOTTOM"
        }

        # 3. Logic Pipeline (Common Mode)
        if not self.calibration_session and self.selected_exercise in self.counters:

            # A. Counting Engine (Forgiving)
            # Tracks cycles even if form is bad
            counter = self.counters[self.selected_exercise]
            rep_state = counter.update(metrics["primary"], time.time())

            payload["rep_count"] = rep_state.rep_count
            payload["rep_phase"] = rep_state.phase

            # B. Form Engine (Strict / Adaptive)
            # Rates quality based on 'critic' and 'capabilities'
            if self.selected_exercise in self.checkers:
                checker = self.checkers[self.selected_exercise]

                # Combine primary + secondary metrics for the checker
                all_metrics = metrics["form"].copy()
                all_metrics["primary"] = metrics["primary"]

                # Generate feedback strings
                feedback_msgs = checker.check(all_metrics, rep_state.progress, self.critic_value)

                if feedback_msgs:
                    payload["feedback"] = " ".join(feedback_msgs)

        # 4. Calibration Pipeline
        if self.calibration_session:
            self.calibration_buffer.append(smoothed_landmarks)
            self.calibration_timestamps.append(time.time())

            end_time = self.calibration_session.get("end_time")
            now = time.time()
            frozen = end_time is not None and now >= end_time

            self.calibration_progress = {
                "exercise": self.selected_exercise,
                "seconds_remaining": max(0.0, end_time - now) if end_time else 0,
                "frozen": frozen,
            }
            payload["calibration_progress"] = self.calibration_progress
        print(payload['rep_count'], payload['rep_phase'])
        return payload

    def close(self) -> None:
        if hasattr(self.estimator, 'close'):
            self.estimator.close()