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
from kinematics import KinematicFeatureExtractor
from session import WorkoutSession, FormSnapshot
from coaches import FormAnalyzer
from coaches.realtime_form_coach import get_realtime_form_coach

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
        
        # Session management
        self.active_session: Optional[WorkoutSession] = None
        
        # Form analysis
        self.form_analyzer = FormAnalyzer()
        self.realtime_form_coach = get_realtime_form_coach()
        
        # Rep tracking state
        self.total_reps = 0
        self.rep_timestamps: List[float] = []
        self.last_form_snapshot: Optional[FormSnapshot] = None
        self.all_form_snapshots: List[Dict] = []
        self.post_rep_command: Optional[str] = None
        self.post_rep_command_frames: int = 0
        self.last_rep_count: int = 0  # Track previous rep count to detect new reps
        
        # Initialize rep buffer
        self._init_rep_buffer()

    def _init_rep_buffer(self):
        """Initialize or clear the buffer for collecting rep data."""
        self.rep_buffer = {
            "frames": [],
            "start_time": None,
            "in_rep": False,
            "min_angle": 180,
            "max_angle": 0,
            "critical_frame": None,
        }
    
    def _buffer_frame(self, frame_data: Dict[str, Any]):
        """Add a frame's data to the rep buffer."""
        self.rep_buffer["frames"].append(frame_data)
        
        # Track critical frame (min angle = bottom of movement)
        angle = frame_data.get("primary_angle", 180)
        if angle < self.rep_buffer["min_angle"]:
            self.rep_buffer["min_angle"] = angle
            self.rep_buffer["critical_frame"] = frame_data
    
    def _analyze_rep_buffer(self, exercise: str) -> Optional[FormSnapshot]:
        """
        Analyze the buffered rep data and create a FormSnapshot.
        Returns None if insufficient data.
        """
        frames = self.rep_buffer["frames"]
        if len(frames) < 3:
            return None
        
        critical = self.rep_buffer["critical_frame"] or frames[len(frames) // 2]
        start_time = self.rep_buffer["start_time"] or time.time()
        
        # Calculate static primitives from critical frame
        static_primitives = {}
        
        if exercise == "squats":
            depth_angle = critical.get("knee_angle", 90)
            depth_cat = self.form_analyzer.categorize_primitive("squats", "squat_depth", depth_angle)
            static_primitives["squat_depth"] = {"value": round(depth_angle, 1), "category": depth_cat}
            
            torso_angle = critical.get("torso_angle", 0)
            torso_cat = self.form_analyzer.categorize_primitive("squats", "torso_angle", torso_angle)
            static_primitives["torso_angle"] = {"value": round(torso_angle, 1), "category": torso_cat}
            
        elif exercise == "bicep_curls":
            flexion_angle = critical.get("elbow_angle", 90)
            flexion_cat = self.form_analyzer.categorize_primitive("bicep_curls", "peak_flexion", flexion_angle)
            static_primitives["peak_flexion"] = {"value": round(flexion_angle, 1), "category": flexion_cat}
            
            elbow_drift = critical.get("elbow_drift", 0)
            drift_cat = self.form_analyzer.categorize_primitive("bicep_curls", "elbow_position", abs(elbow_drift))
            static_primitives["elbow_position"] = {"value": round(abs(elbow_drift), 3), "category": drift_cat}
        
        # Calculate dynamic primitives from frame sequence
        dynamic_primitives = {}
        
        if len(frames) >= 2:
            min_idx = 0
            for i, f in enumerate(frames):
                if f.get("primary_angle", 180) == self.rep_buffer["min_angle"]:
                    min_idx = i
                    break
            
            descent_frames = frames[:min_idx + 1] if min_idx > 0 else frames[:1]
            ascent_frames = frames[min_idx:] if min_idx < len(frames) - 1 else frames[-1:]
            
            if exercise == "squats":
                if len(descent_frames) >= 2:
                    descent_time = descent_frames[-1].get("timestamp", 0) - descent_frames[0].get("timestamp", 0)
                    if descent_time > 0:
                        descent_speed = abs(descent_frames[-1].get("hip_y", 0) - descent_frames[0].get("hip_y", 0)) / descent_time
                        descent_cat = self.form_analyzer.categorize_primitive("squats", "descent_speed", descent_speed)
                        dynamic_primitives["descent_speed"] = {"value": round(descent_speed, 3), "category": descent_cat}
                
                if len(ascent_frames) >= 2:
                    ascent_time = ascent_frames[-1].get("timestamp", 0) - ascent_frames[0].get("timestamp", 0)
                    if ascent_time > 0:
                        ascent_speed = abs(ascent_frames[-1].get("hip_y", 0) - ascent_frames[0].get("hip_y", 0)) / ascent_time
                        ascent_cat = self.form_analyzer.categorize_primitive("squats", "ascent_speed", ascent_speed)
                        dynamic_primitives["ascent_speed"] = {"value": round(ascent_speed, 3), "category": ascent_cat}
                
                knee_positions = [f.get("knee_x", 0) for f in frames]
                if knee_positions:
                    knee_deviation = max(knee_positions) - min(knee_positions)
                    stability_cat = self.form_analyzer.categorize_primitive("squats", "knee_stability", knee_deviation)
                    dynamic_primitives["knee_stability"] = {"value": round(knee_deviation, 3), "category": stability_cat}
            
            elif exercise == "bicep_curls":
                if len(descent_frames) >= 2:
                    lift_time = descent_frames[-1].get("timestamp", 0) - descent_frames[0].get("timestamp", 0)
                    if lift_time > 0:
                        angle_change = abs(descent_frames[-1].get("elbow_angle", 0) - descent_frames[0].get("elbow_angle", 0))
                        lift_speed = angle_change / lift_time
                        lift_cat = self.form_analyzer.categorize_primitive("bicep_curls", "lift_speed", lift_speed)
                        dynamic_primitives["lift_speed"] = {"value": round(lift_speed, 1), "category": lift_cat}
                
                shoulder_positions = [f.get("shoulder_z", 0) for f in frames]
                if shoulder_positions:
                    swing = max(shoulder_positions) - min(shoulder_positions)
                    swing_cat = self.form_analyzer.categorize_primitive("bicep_curls", "swing_momentum", swing)
                    dynamic_primitives["swing_momentum"] = {"value": round(swing, 3), "category": swing_cat}
        
        # Determine form states
        all_primitives = {}
        for name, data in static_primitives.items():
            all_primitives[name] = data.get("category", "unknown")
        for name, data in dynamic_primitives.items():
            all_primitives[name] = data.get("category", "unknown")
        
        form_states = self.form_analyzer.determine_form_states(exercise, all_primitives)
        
        # Get rep count from counter
        rep_number = self.counters[exercise].state.rep_count if exercise in self.counters else self.total_reps
        
        snapshot = FormSnapshot(
            rep_number=rep_number,
            timestamp=start_time,
            static_primitives=static_primitives,
            dynamic_primitives=dynamic_primitives,
            form_states=form_states
        )
        
        return snapshot

    def _handle_rep_completion(self, exercise: str):
        """Handle all logic that follows a successful rep count."""
        self.total_reps += 1
        self.rep_timestamps.append(time.time())

        if self.rep_buffer["in_rep"]:
            snapshot = self._analyze_rep_buffer(exercise)
            if snapshot:
                print(f"FormSnapshot: {snapshot.form_states} | Static: {list(snapshot.static_primitives.keys())} | Dynamic: {list(snapshot.dynamic_primitives.keys())}")
                self.last_form_snapshot = snapshot
                self.all_form_snapshots.append(snapshot.to_dict())
                
                if self.active_session and self.active_session.current_set:
                    self.active_session.current_set.add_form_snapshot(snapshot)

                command = self.realtime_form_coach.get_post_rep_command(
                    exercise, snapshot.form_states
                )
                if command:
                    self.post_rep_command = command
                    self.post_rep_command_frames = 90
            
            self._init_rep_buffer()

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
                if self.selected_exercise in self.counters:
                    self.counters[self.selected_exercise].reset()
                
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
                if self.selected_exercise in self.counters:
                    self.counters[self.selected_exercise].reset()
                
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
            print(summary)
            return {
                "event": "session_ended",
                "session": summary,
            }

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
            rep_state = counter.update(metrics["primary"], time.time(), metrics["form"])

            payload["rep_count"] = rep_state.rep_count
            payload["rep_phase"] = rep_state.phase

            # B. Rep buffer management for form analysis
            # Build frame data for buffering
            frame_data = {
                "timestamp": time.time(),
                "primary_angle": metrics["primary"],
                "elbow_angle": metrics["form"].get("elbow_angle", metrics["primary"]) if self.selected_exercise == "bicep_curls" else 0,
                "knee_angle": metrics["form"].get("knee_angle", metrics["primary"]) if self.selected_exercise == "squats" else 0,
                "torso_angle": metrics["form"].get("torso_angle", 0),
                "elbow_drift": metrics["form"].get("elbow_drift", 0),
                "hip_y": metrics["form"].get("hip_y", 0),
                "knee_x": metrics["form"].get("knee_x", 0),
                "shoulder_z": metrics["form"].get("shoulder_z", 0),
            }
            
            # Start buffering when entering the movement phase
            if rep_state.phase in ["DOWN_PHASE", "UP_PHASE"]:
                if not self.rep_buffer["in_rep"]:
                    self.rep_buffer["in_rep"] = True
                    self.rep_buffer["start_time"] = time.time()
                self._buffer_frame(frame_data)
            
            # Detect rep completion (rep count increased)
            if rep_state.rep_count > self.last_rep_count:
                self._handle_rep_completion(self.selected_exercise)
                
                # Track rep in active session
                if self.active_session and self.active_session.current_set:
                    self.active_session.current_set.add_rep()
            
            self.last_rep_count = rep_state.rep_count


            # D. Post-rep command display (countdown frames)
            if self.post_rep_command_frames > 0:
                payload["post_rep_command"] = self.post_rep_command
                self.post_rep_command_frames -= 1
                if self.post_rep_command_frames == 0:
                    self.post_rep_command = None
            
            # E. Include form snapshots in payload
            if self.all_form_snapshots:
                payload["form_snapshots"] = self.all_form_snapshots

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