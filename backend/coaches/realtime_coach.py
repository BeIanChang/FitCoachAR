"""
Realtime Coach - Fast, Rule-Based Feedback During Workout

Provides three types of realtime feedback:
1. Text Tips - Short coaching messages (e.g., "Keep elbow stable!")
2. Arrow Feedback - Directional arrows on joints for AR overlay
3. Joint Highlighting - Which joints to highlight red/green

All feedback is generated without LLM calls for <100ms latency.
"""

from typing import Dict, List, Optional, Any


class RealtimeCoach:
    """
    Fast, rule-based coaching feedback for realtime workout guidance.
    
    Usage:
        coach = RealtimeCoach()
        
        # During workout loop:
        tip = coach.get_text_feedback("bicep_curls", metrics)
        arrows = coach.get_arrow_feedback("bicep_curls", tip, landmarks)
        error_joints = coach.get_error_joints("bicep_curls", tip)
    """
    
    # Template-based tips with variants to avoid repetition
    TEXT_TEMPLATES = {
        # Bicep Curls
        ("bicep_curls", "elbow_stability"): [
            "Keep elbow stable!",
            "Lock your elbow at your side!",
            "Don't let your elbow drift!",
        ],
        ("bicep_curls", "curl_higher"): [
            "Curl higher!",
            "Bring the weight up more!",
            "Full range of motion!",
        ],
        ("bicep_curls", "perfect"): [
            "Good rep!",
            "Great curl!",
            "Perfect form!",
        ],
        
        # Squats
        ("squats", "go_deeper"): [
            "Go deeper!",
            "Drop those hips lower!",
            "Get below parallel!",
        ],
        ("squats", "knees_out"): [
            "Push knees out!",
            "Don't let knees cave in!",
            "Knees track over toes!",
        ],
        ("squats", "chest_up"): [
            "Keep chest up!",
            "Don't lean forward!",
            "Upright torso!",
        ],
        ("squats", "perfect"): [
            "Good depth!",
            "Great squat!",
            "Perfect form!",
        ],
        
        # General
        ("general", "camera"): [
            "Adjust camera to show full body",
            "Move back so I can see you",
        ],
    }
    
    # Arrow configurations for each feedback type
    ARROW_CONFIG = {
        # Bicep Curls
        "elbow_stability": {
            "joint_idx": 14,  # Right elbow (MediaPipe)
            "direction": "left",  # Point toward body
            "color": "#facc15",  # Yellow
        },
        "curl_higher": {
            "joint_idx": 16,  # Right wrist
            "direction": "up",
            "color": "#facc15",
        },
        
        # Squats
        "go_deeper": {
            "joint_idx": 24,  # Right hip
            "direction": "down",
            "color": "#facc15",
        },
        "knees_out": {
            "joint_idx": 26,  # Right knee
            "direction": "right",  # Push outward
            "color": "#facc15",
        },
        "chest_up": {
            "joint_idx": 12,  # Right shoulder
            "direction": "up",
            "color": "#facc15",
        },
    }
    
    # Joint indices to highlight for each feedback type
    ERROR_JOINTS = {
        "elbow_stability": [11, 13, 14],  # Shoulder, upper arm, elbow
        "curl_higher": [14, 16],  # Elbow, wrist
        "go_deeper": [23, 24, 25, 26],  # Hips and knees
        "knees_out": [25, 26],  # Knees
        "chest_up": [11, 12],  # Shoulders
    }
    
    def __init__(self):
        self.template_indices: Dict[tuple, int] = {}
        self.last_feedback_type: Optional[str] = None
        self.feedback_cooldown: Dict[str, int] = {}  # Prevent spam
    
    def get_text_feedback(
        self,
        exercise: str,
        error_type: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get a text tip for the current exercise state.
        
        Args:
            exercise: "bicep_curls" or "squats"
            error_type: Type of error detected (e.g., "elbow_stability", "go_deeper")
            metrics: Optional dict with current angle/form metrics
            
        Returns:
            Short coaching tip string
        """
        if error_type is None:
            # No error - return positive feedback
            key = (exercise, "perfect")
        else:
            key = (exercise, error_type)
        
        # Fallback to general if specific template not found
        if key not in self.TEXT_TEMPLATES:
            key = (exercise, "perfect")
            if key not in self.TEXT_TEMPLATES:
                return ""
        
        return self._get_template_variant(key)
    
    def get_arrow_feedback(
        self,
        exercise: str,
        feedback_text: str,
        landmarks: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate arrow data for AR overlay based on feedback.
        
        Args:
            exercise: Current exercise
            feedback_text: The text feedback being shown
            landmarks: Optional landmark data for visibility check
            
        Returns:
            List of arrow dicts with joint_idx, direction, color, type
        """
        arrows = []
        
        # Skip positive feedback - no arrows needed
        if not feedback_text or feedback_text in ["", "Good rep!", "Great curl!", "Good depth!", "Perfect form!", "Great squat!"]:
            return arrows
        
        # Detect error type from feedback text
        error_type = self._detect_error_type(exercise, feedback_text)
        
        if error_type and error_type in self.ARROW_CONFIG:
            config = self.ARROW_CONFIG[error_type]
            arrows.append({
                "joint_idx": config["joint_idx"],
                "type": error_type,
                "direction": config["direction"],
                "color": config["color"],
            })
        
        return arrows
    
    def get_error_joints(
        self,
        exercise: str,
        feedback_text: str
    ) -> List[int]:
        """
        Get joint indices that should be highlighted as errors.
        
        Args:
            exercise: Current exercise
            feedback_text: The text feedback being shown
            
        Returns:
            List of MediaPipe joint indices to highlight red
        """
        if not feedback_text:
            return []
        
        error_type = self._detect_error_type(exercise, feedback_text)
        
        if error_type and error_type in self.ERROR_JOINTS:
            return self.ERROR_JOINTS[error_type]
        
        return []
    
    def analyze_form(
        self,
        exercise: str,
        current_angle: float,
        target_angle: float,
        elbow_drift: float = 0.0,
        stability_threshold: float = 0.05
    ) -> Optional[str]:
        """
        Analyze form metrics and return error type if any.
        
        Args:
            exercise: Current exercise
            current_angle: Current joint angle
            target_angle: Target/calibrated angle
            elbow_drift: Horizontal elbow movement (for curls)
            stability_threshold: Max allowed drift
            
        Returns:
            Error type string or None if form is good
        """
        if exercise == "bicep_curls":
            # Check elbow stability
            if elbow_drift > stability_threshold:
                return "elbow_stability"
            # Check curl depth (angle should be small at top)
            # This would need phase detection to work properly
            
        elif exercise == "squats":
            # Check squat depth
            if current_angle > target_angle + 20:  # Not deep enough
                return "go_deeper"
        
        return None
    
    def _get_template_variant(self, key: tuple) -> str:
        """Cycle through template variants to avoid repetition."""
        templates = self.TEXT_TEMPLATES.get(key, [])
        if not templates:
            return ""
        
        if key not in self.template_indices:
            self.template_indices[key] = 0
        
        idx = self.template_indices[key]
        tip = templates[idx]
        
        # Cycle to next variant
        self.template_indices[key] = (idx + 1) % len(templates)
        
        return tip
    
    def _detect_error_type(self, exercise: str, feedback_text: str) -> Optional[str]:
        """Detect error type from feedback text."""
        text_lower = feedback_text.lower()
        
        if exercise == "bicep_curls":
            if "elbow" in text_lower or "stable" in text_lower or "drift" in text_lower:
                return "elbow_stability"
            if "higher" in text_lower or "range" in text_lower or "curl" in text_lower:
                return "curl_higher"
                
        elif exercise == "squats":
            if "deeper" in text_lower or "lower" in text_lower or "drop" in text_lower:
                return "go_deeper"
            if "knee" in text_lower and ("out" in text_lower or "cave" in text_lower):
                return "knees_out"
            if "chest" in text_lower or "lean" in text_lower or "upright" in text_lower:
                return "chest_up"
        
        return None
    
    def reset(self):
        """Reset template cycling and cooldowns."""
        self.template_indices.clear()
        self.feedback_cooldown.clear()
        self.last_feedback_type = None
