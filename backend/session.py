"""
Workout Session Management for FitCoachAR

Supports multi-exercise, multi-set workout sessions.
Example session: 3x10 squats, 2x12 bicep curls, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import time
import uuid


@dataclass
class FormSnapshot:
    """
    A snapshot of form primitives captured for a single repetition.
    Contains both the raw metric values and their categorized names.
    """
    rep_number: int
    timestamp: float
    # Static primitives: {"squat_depth": {"value": 85, "category": "deep"}, ...}
    static_primitives: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Dynamic primitives: {"descent_speed": {"value": 0.4, "category": "controlled"}, ...}
    dynamic_primitives: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Form states detected for this rep: ["GOOD_REP"] or ["SHALLOW_DEPTH", "KNEE_COLLAPSE"]
    form_states: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rep_number": self.rep_number,
            "timestamp": self.timestamp,
            "static_primitives": self.static_primitives,
            "dynamic_primitives": self.dynamic_primitives,
            "form_states": self.form_states,
        }


@dataclass
class ExerciseSet:
    """A single set of an exercise within a workout session."""
    
    exercise: str           # "bicep_curls" or "squats"
    target_reps: int        # Goal reps for this set
    completed_reps: int = 0
    mistakes: Dict[str, int] = field(default_factory=dict)
    form_snapshots: List[FormSnapshot] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if target reps have been reached."""
        return self.completed_reps >= self.target_reps
    
    @property
    def remaining_reps(self) -> int:
        """Reps left to complete this set."""
        return max(0, self.target_reps - self.completed_reps)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of this set in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return None
    
    def add_rep(self) -> bool:
        """
        Record a completed rep.
        Returns True if this rep completed the set.
        """
        self.completed_reps += 1
        return self.is_complete
    
    def add_mistake(self, mistake_type: str):
        """Record a form mistake."""
        self.mistakes[mistake_type] = self.mistakes.get(mistake_type, 0) + 1

    def add_form_snapshot(self, snapshot: FormSnapshot):
        """Record a form snapshot for a completed rep."""
        self.form_snapshots.append(snapshot)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exercise": self.exercise,
            "target_reps": self.target_reps,
            "completed_reps": self.completed_reps,
            "remaining_reps": self.remaining_reps,
            "is_complete": self.is_complete,
            "mistakes": self.mistakes,
            "form_snapshots": [s.to_dict() for s in self.form_snapshots],
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class WorkoutSession:
    """
    A complete workout session containing multiple exercise sets.
    
    Usage:
        session = WorkoutSession.from_config([
            {"exercise": "squats", "reps": 10},
            {"exercise": "bicep_curls", "reps": 12},
            {"exercise": "squats", "reps": 10},
        ])
        
        # Start first set
        session.start()
        
        # Record reps
        set_complete = session.record_rep()
        
        # Move to next set
        session.advance_to_next_set()
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Workout"
    sets: List[ExerciseSet] = field(default_factory=list)
    current_set_index: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    
    @classmethod
    def from_config(cls, sets_config: List[Dict[str, Any]], name: str = "Workout") -> "WorkoutSession":
        """
        Create a session from a configuration list.
        
        Args:
            sets_config: List of dicts with "exercise" and "reps" keys
            name: Optional session name
            
        Example:
            WorkoutSession.from_config([
                {"exercise": "squats", "reps": 10},
                {"exercise": "bicep_curls", "reps": 12},
            ])
        """
        sets = [
            ExerciseSet(
                exercise=s.get("exercise", "bicep_curls"),
                target_reps=s.get("reps", 10)
            )
            for s in sets_config
        ]
        return cls(sets=sets, name=name)
    
    @property
    def current_set(self) -> Optional[ExerciseSet]:
        """Get the currently active set."""
        if 0 <= self.current_set_index < len(self.sets):
            return self.sets[self.current_set_index]
        return None
    
    @property
    def current_exercise(self) -> Optional[str]:
        """Get the exercise type for the current set."""
        if self.current_set:
            return self.current_set.exercise
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if all sets have been completed."""
        return self.current_set_index >= len(self.sets)
    
    @property
    def is_active(self) -> bool:
        """Check if session is in progress."""
        return self.started_at is not None and self.finished_at is None
    
    @property
    def total_sets(self) -> int:
        return len(self.sets)
    
    @property
    def completed_sets(self) -> int:
        """Number of fully completed sets."""
        return sum(1 for s in self.sets if s.is_complete)
    
    @property
    def total_reps_completed(self) -> int:
        """Total reps across all sets."""
        return sum(s.completed_reps for s in self.sets)
    
    @property
    def total_reps_target(self) -> int:
        """Total target reps across all sets."""
        return sum(s.target_reps for s in self.sets)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Total session duration."""
        if self.started_at:
            end = self.finished_at or time.time()
            return end - self.started_at
        return None
    
    @property
    def all_mistakes(self) -> Dict[str, int]:
        """Aggregate mistakes from all sets."""
        combined = {}
        for s in self.sets:
            for k, v in s.mistakes.items():
                combined[k] = combined.get(k, 0) + v
        return combined
    
    def start(self):
        """Start the session and the first set."""
        self.started_at = time.time()
        if self.current_set:
            self.current_set.start_time = time.time()
    
    def record_rep(self) -> Dict[str, Any]:
        """
        Record a rep for the current set.
        
        Returns:
            Dict with "set_complete" and "session_complete" flags
        """
        result = {
            "set_complete": False,
            "session_complete": False,
            "current_set_index": self.current_set_index,
            "reps_in_set": 0,
            "target_reps": 0,
        }
        
        if self.current_set:
            set_complete = self.current_set.add_rep()
            result["set_complete"] = set_complete
            result["reps_in_set"] = self.current_set.completed_reps
            result["target_reps"] = self.current_set.target_reps
            
        return result
    
    def record_mistake(self, mistake_type: str):
        """Record a mistake for the current set."""
        if self.current_set:
            self.current_set.add_mistake(mistake_type)
    
    def advance_to_next_set(self) -> Optional[ExerciseSet]:
        """
        Move to the next set in the session.
        
        Returns:
            The next ExerciseSet, or None if session is complete
        """
        # End current set
        if self.current_set:
            self.current_set.end_time = time.time()
        
        # Move to next
        self.current_set_index += 1
        
        # Start next set if available
        if self.current_set:
            self.current_set.start_time = time.time()
            return self.current_set
        else:
            # Session complete
            self.finished_at = time.time()
            return None
    
    def skip_current_set(self) -> Optional[ExerciseSet]:
        """Skip the current set and move to next."""
        return self.advance_to_next_set()
    
    def finish(self):
        """End the session early."""
        if self.current_set and not self.current_set.end_time:
            self.current_set.end_time = time.time()
        self.finished_at = time.time()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current session progress for UI display."""
        return {
            "session_id": self.id,
            "session_name": self.name,
            "is_active": self.is_active,
            "is_complete": self.is_complete,
            "current_set_index": self.current_set_index,
            "total_sets": self.total_sets,
            "completed_sets": self.completed_sets,
            "current_exercise": self.current_exercise,
            "current_set": self.current_set.to_dict() if self.current_set else None,
            "total_reps_completed": self.total_reps_completed,
            "total_reps_target": self.total_reps_target,
            "duration_seconds": self.duration_seconds,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Full session data for saving/summary."""
        return {
            "id": self.id,
            "name": self.name,
            "sets": [s.to_dict() for s in self.sets],
            "current_set_index": self.current_set_index,
            "is_complete": self.is_complete,
            "total_sets": self.total_sets,
            "completed_sets": self.completed_sets,
            "total_reps_completed": self.total_reps_completed,
            "total_reps_target": self.total_reps_target,
            "all_mistakes": self.all_mistakes,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
        }


# Convenience function for quick session creation
def create_simple_session(exercise: str, sets: int, reps: int) -> WorkoutSession:
    """
    Create a simple session with repeated sets of the same exercise.
    
    Example:
        session = create_simple_session("squats", sets=3, reps=10)
        # Creates: 3 sets of 10 squats
    """
    config = [{"exercise": exercise, "reps": reps} for _ in range(sets)]
    return WorkoutSession.from_config(config, name=f"{sets}x{reps} {exercise}")


def create_circuit_session(exercises: List[str], reps: int, rounds: int = 1) -> WorkoutSession:
    """
    Create a circuit session alternating between exercises.
    
    Example:
        session = create_circuit_session(["squats", "bicep_curls"], reps=10, rounds=3)
        # Creates: squats(10) -> curls(10) -> squats(10) -> curls(10) -> squats(10) -> curls(10)
    """
    config = []
    for _ in range(rounds):
        for ex in exercises:
            config.append({"exercise": ex, "reps": reps})
    return WorkoutSession.from_config(config, name=f"{rounds}x Circuit")
