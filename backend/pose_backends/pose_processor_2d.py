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
from session import WorkoutSession, FormSnapshot

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
        
        # Session management
        self.active_session: Optional[WorkoutSession] = None
        
        # Rep tracking state
        self.total_reps = 0
        self.last_rep_count: int = 0  # Track previous rep count to detect new reps
        self.post_rep_command: Optional[str] = None
        self.post_rep_command_frames: int = 0
        self.all_form_snapshots: List[Dict] = []


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

            mode = "common"
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
                # Run the updated geometry analysis
                calib_result = calibrate_from_landmarks(
                    landmarks_array,
                    self.joint_map,
                    exercise=exercise,
                    smoothing_window=5,
                    fps=30
                )
            except Exception as e:
                print(e)
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

        # ==================== SESSION COMMANDS ====================
        
        if command == "start_session":
            sets_config = command_data.get("sets", [])
            session_name = command_data.get("name", "Workout")
            
            if not sets_config:
                return {"event": "session_error", "message": "No sets provided."}
            
            self.active_session = WorkoutSession.from_config(sets_config, name=session_name)
            self.active_session.start()
            
            # Set the current exercise and reset counter
            if self.active_session.current_exercise:
                self.selected_exercise = self.active_session.current_exercise
                if self.selected_exercise in self.counters:
                    self.counters[self.selected_exercise].reset()
            
            return {
                "event": "session_started",
                "session": self.active_session.to_dict(),
                "progress": self.active_session.get_progress(),
            }
        
        if command == "next_set":
            if not self.active_session:
                return {"event": "session_error", "message": "No active session."}
            
            next_set = self.active_session.advance_to_next_set()
            if next_set:
                self.selected_exercise = next_set.exercise
                if self.selected_exercise in self.counters:
                    self.counters[self.selected_exercise].reset()
                return {
                    "event": "set_started",
                    "session": self.active_session.to_dict(),
                    "progress": self.active_session.get_progress(),
                }
            else:
                return {
                    "event": "session_complete",
                    "session": self.active_session.to_dict(),
                    "progress": self.active_session.get_progress(),
                }

        if command == "end_session":
            if not self.active_session:
                return {"event": "session_error", "message": "No active session."}
            self.active_session.finish()
            return {"event": "session_ended", "session": self.active_session.to_dict()}

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
            "rep_phase": "BOTTOM"
        }

        # 3. Logic Pipeline (Common Mode - Rep Counting & Form Checking)
        if not self.calibration_session and self.selected_exercise in self.counters:
            
            # A. Orchestrate the Counter
            counter = self.counters[self.selected_exercise]
            
            # NOTE: We now pass the FORM metrics to the counter's update method.
            # The counter calculates counting based on 'primary' but saves 'form' snapshots upon rep completion.
            rep_state = counter.update(metrics["primary"], time.time(), metrics["form"])
            
            payload["rep_count"] = rep_state.rep_count
            payload["rep_phase"] = rep_state.phase

            # B. Session Tracking
            if rep_state.rep_count > self.last_rep_count:
                self.total_reps += 1
                if self.active_session and self.active_session.current_set:
                    self.active_session.current_set.add_rep()
            self.last_rep_count = rep_state.rep_count

            # C. Form Engine (Strict / Adaptive)
            if self.selected_exercise in self.checkers:
                checker = self.checkers[self.selected_exercise]

                # Combine primary + secondary metrics for live feedback
                all_metrics = metrics["form"].copy()
                all_metrics["primary"] = metrics["primary"]

                # CRITICAL: Retrieve the 'validated' form metrics stored by the counter 
                # at the moment of the last successful rep completion.
                validated_form_metrics = rep_state.form_at_rep_complete.copy()
                
                # The Checker uses 'all_metrics' for real-time guidance
                # and 'validated_form_metrics' to confirm if the last rep met normative constraints.
                feedback_msgs = ""

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