"""
Realtime Coach - Fast, Rule-Based Feedback During Workout

Provides three types of realtime feedback:
1. Text Tips - Short coaching messages (e.g., "Keep elbow stable!")
2. Arrow Feedback - Directional arrows on joints for AR overlay
3. Joint Highlighting - Which joints to highlight red/green

All feedback is generated without LLM calls for <100ms latency.
"""

from typing import Dict, List, Optional, Any


from .form_analyzer import FormAnalyzer

# Mapping from form states to concise coaching commands (Moved from RealtimeFormCoach)
POST_REP_COMMANDS = {
    "bicep_curls": {
        "BODY_SWING": ["No swinging!", "Keep body still!", "No momentum!"],
        "ELBOW_DRIFT": ["Pin elbow!", "Elbow fixed!", "Don't move elbow!"],
        "INCOMPLETE_RANGE": ["Curl higher!", "Full range!", "All the way up!"],
        "UNCONTROLLED_LOWER": ["Lower slowly!", "Control descent!", "Don't drop it!"],
        "WRIST_STRAIN": ["Straight wrist!", "Neutral wrist!", "Don't bend wrist!"],
        "NO_SQUEEZE": ["Squeeze top!", "Pause at top!", "Hold peak!"],
        "GOOD_REP": ["Good rep!", "Nice form!", "Keep it up!"],
    },
    "squats": {
        "SHALLOW_SQUAT": ["Go deeper!", "Drop lower!", "Below parallel!"],
        "KNEE_CAVE": ["Knees out!", "Push knees out!", "No knee cave!"],
        "FORWARD_LEAN": ["Chest up!", "Stay upright!", "Don't lean!"],
        "HIP_ASYMMETRY": ["Level hips!", "Balance sides!", "Even push!"],
        "GOOD_REP": ["Good depth!", "Great squat!", "Perfect form!"],
    },
}

class RealtimeCoach:
    def get_post_rep_command(self, exercise: str, form_states: List[str]) -> Optional[str]:
        """Get a concise coaching command after a rep completion."""
        if not form_states:
            return None
        
        # If good rep, occasionally give positive
        if "GOOD_REP" in form_states and len(form_states) == 1:
            return self._get_command(exercise, "GOOD_REP")
            
        # Priority to errors
        for state in form_states:
            if state != "GOOD_REP":
                return self._get_command(exercise, state)
        return None

    def _get_command(self, exercise: str, state: str) -> Optional[str]:
        """Get command variant."""
        commands = POST_REP_COMMANDS.get(exercise, {}).get(state, [])
        if not commands:
            return None
        key = (exercise, state)
        idx = self.command_index.get(key, 0)
        cmd = commands[idx % len(commands)]
        self.command_index[key] = idx + 1
        return cmd

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
        
        # Stability / Hysteresis State
        self.sticky_error: Optional[str] = None
        self.sticky_timer: int = 0
        self.STICKY_DURATION: int = 15  # Frames to hold feedback (~0.5s at 30fps)
        
        # Unified Logic Engine
        self.form_analyzer = FormAnalyzer()
        
        # Post-Rep Command State
        self.command_index: Dict[tuple, int] = {}
    
    def get_text_feedback(
        self,
        exercise: str,
        error_type: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get a text tip for the current exercise state.
        Caches the result to prevent text flickering/cycling.
        """
        if error_type is None:
            # No error - return positive feedback
            key = (exercise, "perfect")
        else:
            key = (exercise, error_type)
        
        # Stable Feedback Logic:
        # Only cycle the text template if the error type has changed.
        # This prevents the text from changing every single frame (30fps cycling).
        if error_type != self.last_feedback_type:
            self.last_feedback_type = error_type
            
            # Generate new variant
            if key not in self.TEXT_TEMPLATES:
                key = (exercise, "perfect")
                if key not in self.TEXT_TEMPLATES:
                    self.current_tip = ""
                else:
                    self.current_tip = self._get_template_variant(key)
            else:
                self.current_tip = self._get_template_variant(key)
        
        # Return the cached tip (or empty if first run and logic failed, but init handles that)
        if not hasattr(self, "current_tip"):
             self.current_tip = ""
             
        return self.current_tip
    
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
        stability_threshold: float = 0.05  # Kept for signature compatibility but ignored
    ) -> Optional[str]:
        """
        Analyze form metrics using unified FormCode logic.
        Includes hysteresis to prevent feedback flickering.
        
        Args:
            exercise: Current exercise
            current_angle: Current joint angle
            target_angle: Target/calibrated angle
            elbow_drift: Horizontal elbow movement (for curls)
            stability_threshold: Ignored (uses FormCode config)
            
        Returns:
            Error type string or None if form is good
        """
        instant_error = None
        
        if exercise == "bicep_curls":
            # Priority 1: Check elbow stability (Safety/Mechanics)
            # Unified Logic: Use 'elbow_position' FormCode
            drift_category = self.form_analyzer.categorize_form_code(
                "bicep_curls", "elbow_position", elbow_drift
            )
            
            if drift_category in ["slight_drift", "major_drift"]:
                instant_error = "elbow_stability"
                
            # Priority 2: Check curl depth (Range of Motion)
            # Only checked if stability is good.
            
        elif exercise == "squats":
            # Priority 1: Check squat depth (Primary metric)
            # Unified Logic: Use 'squat_depth' FormCode
            depth_category = self.form_analyzer.categorize_form_code(
                "squats", "squat_depth", current_angle
            )
            
            # Note: For squats, 'shallow' is bad, 'deep'/'parallel' are good.
            if depth_category == "shallow":
                instant_error = "go_deeper"
        
        # --- Hysteresis Logic ---
        if instant_error:
            # New error detected: Update sticky state and refresh timer
            self.sticky_error = instant_error
            self.sticky_timer = self.STICKY_DURATION
            return instant_error
        else:
            # No immediate error: Check if we should hold the previous error
            if self.sticky_timer > 0:
                self.sticky_timer -= 1
                return self.sticky_error
            else:
                self.sticky_error = None
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
