
"""
Realtime Form Coach - Aligned with Form Primitives and States

Provides realtime coaching feedback based on the form analysis system:
1. Post-Rep Feedback - Concise coaching command after a bad rep
2. Current Rep Guidance - Arrow indicators on skeleton overlay

Uses the same form states as the post-workout analysis:
- BODY_SWING, ELBOW_DRIFT, INCOMPLETE_RANGE, etc.
"""

from typing import Dict, List, Optional, Any


# Mapping from form states to concise coaching commands
# These are shown AFTER a bad rep is completed
POST_REP_COMMANDS = {
    "bicep_curls": {
        "BODY_SWING": [
            "No swinging! Use your bicep only.",
            "Keep your body still!",
            "Don't use momentum!",
        ],
        "ELBOW_DRIFT": [
            "Pin your elbow to your side!",
            "Elbow stays fixed!",
            "Don't move your elbow!",
        ],
        "INCOMPLETE_RANGE": [
            "Curl higher!",
            "Full range of motion!",
            "Bring it all the way up!",
        ],
        "UNCONTROLLED_LOWER": [
            "Lower slowly!",
            "Control the descent!",
            "Don't drop the weight!",
        ],
        "WRIST_STRAIN": [
            "Keep wrist straight!",
            "Neutral wrist position!",
            "Don't bend your wrist!",
        ],
        "NO_SQUEEZE": [
            "Squeeze at the top!",
            "Pause and contract!",
            "Hold at peak!",
        ],
        "GOOD_REP": [
            "Good rep!",
            "Nice form!",
            "Keep it up!",
        ],
    },
    "squats": {
        "SHALLOW_SQUAT": [
            "Go deeper!",
            "Drop lower!",
            "Get below parallel!",
        ],
        "KNEE_CAVE": [
            "Knees out!",
            "Push knees over toes!",
            "Don't let knees cave!",
        ],
        "FORWARD_LEAN": [
            "Chest up!",
            "Stay upright!",
            "Don't lean forward!",
        ],
        "HIP_ASYMMETRY": [
            "Keep hips level!",
            "Balance both sides!",
            "Don't shift to one side!",
        ],
        "GOOD_REP": [
            "Good depth!",
            "Great squat!",
            "Perfect form!",
        ],
    },
}

# Arrow configurations for guiding the CURRENT rep
# Maps form states to arrow indicators on specific joints
GUIDANCE_ARROWS = {
    "bicep_curls": {
        "BODY_SWING": {
            "joint": "right_shoulder",  # 12
            "joint_idx": 12,
            "direction": "none",  # No arrow, just highlight
            "message": "Keep still",
            "color": "#ef4444",  # Red
        },
        "ELBOW_DRIFT": {
            "joint": "right_elbow",  # 14
            "joint_idx": 14,
            "direction": "left",  # Point toward body
            "message": "Pin elbow",
            "color": "#f59e0b",  # Orange
        },
        "INCOMPLETE_RANGE": {
            "joint": "right_wrist",  # 16
            "joint_idx": 16,
            "direction": "up",
            "message": "Curl higher",
            "color": "#f59e0b",
        },
        "UNCONTROLLED_LOWER": {
            "joint": "right_wrist",
            "joint_idx": 16,
            "direction": "down",
            "message": "Slow down",
            "color": "#f59e0b",
        },
    },
    "squats": {
        "SHALLOW_SQUAT": {
            "joint": "right_hip",  # 24
            "joint_idx": 24,
            "direction": "down",
            "message": "Go deeper",
            "color": "#f59e0b",
        },
        "KNEE_CAVE": {
            "joint": "right_knee",  # 26
            "joint_idx": 26,
            "direction": "right",  # Push outward
            "message": "Knees out",
            "color": "#ef4444",
        },
        "FORWARD_LEAN": {
            "joint": "right_shoulder",  # 12
            "joint_idx": 12,
            "direction": "up",
            "message": "Chest up",
            "color": "#f59e0b",
        },
        "HIP_ASYMMETRY": {
            "joint": "left_hip",  # 23
            "joint_idx": 23,
            "direction": "none",
            "message": "Balance hips",
            "color": "#ef4444",
        },
    },
}

# Joint indices for highlighting (MediaPipe pose landmarks)
JOINT_INDICES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class RealtimeFormCoach:
    """
    Realtime coaching aligned with form primitives and states.
    
    Usage:
        coach = RealtimeFormCoach()
        
        # After a rep is completed with form issues:
        command = coach.get_post_rep_command("bicep_curls", ["BODY_SWING", "ELBOW_DRIFT"])
        
        # Get guidance arrow for current rep based on last issue:
        arrow = coach.get_guidance_arrow("bicep_curls", "ELBOW_DRIFT", landmarks)
    """
    
    def __init__(self):
        self.command_index = {}  # Track which command variant to use (for variety)
        self.last_issue = None  # Track last detected issue for guidance
    
    def get_post_rep_command(
        self, 
        exercise: str, 
        form_states: List[str]
    ) -> Optional[str]:
        """
        Get a concise coaching command after a rep is completed.
        
        Args:
            exercise: "bicep_curls" or "squats"
            form_states: List of detected form states for the completed rep
        
        Returns:
            Short coaching command string, or None if rep was good
        """
        if not form_states:
            return None
        
        # If it's a good rep, occasionally give positive feedback
        if "GOOD_REP" in form_states and len(form_states) == 1:
            return self._get_command(exercise, "GOOD_REP")
        
        # Get the most important issue (first non-GOOD_REP state)
        for state in form_states:
            if state != "GOOD_REP":
                self.last_issue = state  # Remember for guidance
                return self._get_command(exercise, state)
        
        return None
    
    def _get_command(self, exercise: str, state: str) -> Optional[str]:
        """Get a command with rotation to avoid repetition."""
        commands = POST_REP_COMMANDS.get(exercise, {}).get(state, [])
        if not commands:
            return None
        
        key = (exercise, state)
        idx = self.command_index.get(key, 0)
        command = commands[idx % len(commands)]
        self.command_index[key] = idx + 1
        
        return command
    
    def get_guidance_arrow(
        self,
        exercise: str,
        form_state: Optional[str],
        landmarks: List[Dict[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get arrow data for guiding the current rep.
        
        Args:
            exercise: "bicep_curls" or "squats"
            form_state: The issue to guide on (usually last detected issue)
            landmarks: Current pose landmarks
        
        Returns:
            Arrow data dict with joint position, direction, message, color
        """
        if not form_state or not landmarks:
            return None
        
        arrow_config = GUIDANCE_ARROWS.get(exercise, {}).get(form_state)
        if not arrow_config:
            return None
        
        joint_idx = arrow_config["joint_idx"]
        if joint_idx >= len(landmarks):
            return None
        
        landmark = landmarks[joint_idx]
        
        return {
            "joint_idx": joint_idx,
            "x": landmark.get("x", 0),
            "y": landmark.get("y", 0),
            "direction": arrow_config["direction"],
            "message": arrow_config["message"],
            "color": arrow_config["color"],
        }
    
    def get_highlight_joints(
        self,
        exercise: str,
        form_states: List[str]
    ) -> List[int]:
        """
        Get joint indices to highlight red for current issues.
        
        Args:
            exercise: "bicep_curls" or "squats"
            form_states: Current detected form states
        
        Returns:
            List of joint indices to highlight
        """
        joints = set()
        
        for state in form_states:
            if state == "GOOD_REP":
                continue
            arrow_config = GUIDANCE_ARROWS.get(exercise, {}).get(state)
            if arrow_config:
                joints.add(arrow_config["joint_idx"])
        
        return list(joints)
    
    def reset(self):
        """Reset state for new workout."""
        self.command_index = {}
        self.last_issue = None


# Singleton instance
_coach_instance = None

def get_realtime_form_coach() -> RealtimeFormCoach:
    """Get the singleton RealtimeFormCoach instance."""
    global _coach_instance
    if _coach_instance is None:
        _coach_instance = RealtimeFormCoach()
    return _coach_instance