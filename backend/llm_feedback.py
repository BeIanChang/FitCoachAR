"""
LLM-Driven Natural Language Feedback for FitCoachAR
Based on proposal Section 4.5

Uses prompt engineering with lightweight LLM to generate
natural, coach-like feedback from quantitative error data.
"""

import json
from typing import Dict, List


class LLMFeedbackGenerator:
    """
    Generates natural language coaching feedback using LLM.
    
    For real deployment, this would call GPT-4-mini or an on-device model.
    For this implementation, we use template-based generation with
    cached responses for common scenarios (to minimize latency).
    """
    
    # Template-based feedback cache for <100ms latency
    FEEDBACK_TEMPLATES = {
        # Bicep Curls
        ('bicep_curls', 'elbow_stability', 'high'): [
            "Keep your elbow stable—lock it at your side!",
            "Try to keep your elbow in one spot throughout the curl.",
            "Focus on keeping your upper arm still."
        ],
        ('bicep_curls', 'curl_depth', 'moderate'): [
            "Nice pace! Try curling a bit higher to finish each rep.",
            "Almost there—bring the weight up just a bit more.",
            "Good form, but aim for a fuller range of motion."
        ],
        ('bicep_curls', 'perfect', None): [
            "Excellent form! Keep it up!",
            "Great curl! Smooth and controlled.",
            "Perfect technique—well done!"
        ],
        
        # Squats
        ('squats', 'squat_depth', 'high'): [
            "Go deeper! Aim to get your hips below your knees.",
            "Drop those hips lower for a complete squat.",
            "You're almost there—just need a bit more depth."
        ],
        ('squats', 'knee_alignment', 'moderate'): [
            "Keep your knees tracking over your toes.",
            "Watch those knees—don't let them cave inward.",
            "Good depth, but focus on knee position."
        ],
        ('squats', 'perfect', None): [
            "Perfect squat! Great depth and form!",
            "Excellent control! Keep it up!",
            "That's how it's done! Beautiful form!"
        ],
        
        # General encouragement
        ('general', 'keep_going', None): [
            "You're doing great! Stay focused!",
            "Nice work! Keep that energy up!",
            "Strong effort! Maintain that form!"
        ]
    }
    
    def __init__(self, use_api=False, api_key=None):
        """
        Initialize feedback generator.
        
        Args:
            use_api: If True, use actual LLM API (requires api_key)
            api_key: API key for LLM service (e.g., OpenAI)
        """
        self.use_api = use_api
        self.api_key = api_key
        self.call_count = 0
        self.template_index = {}
    
    def generate_feedback(self, error_record: Dict) -> str:
        """
        Generate natural language feedback from error analysis.
        
        Args:
            error_record: Dictionary with structure:
                {
                    "exercise": "push_up" | "squat" | "bicep_curls",
                    "phase": "descent" | "bottom" | "ascent",
                    "errors": [
                        {
                            "joint": "right_elbow",
                            "deviation_deg": 12,
                            "type": "too_low" | "too_high" | "unstable"
                        }
                    ],
                    "critic_level": 0.4,  # 0=strict, 1=relaxed
                    "user_style": "friendly" | "professional" | "motivational"
                }
        
        Returns:
            Natural language feedback string
        """
        if self.use_api and self.api_key:
            return self._generate_with_api(error_record)
        else:
            return self._generate_from_template(error_record)
    
    def _generate_from_template(self, error_record: Dict) -> str:
        """
        Fast template-based generation for <100ms latency.
        """
        exercise = error_record.get('exercise', 'general')
        errors = error_record.get('errors', [])
        
        if not errors:
            # No errors - provide encouragement
            key = (exercise, 'perfect', None)
            return self._get_template_variant(key)
        
        # Get primary error
        primary_error = errors[0]
        error_type = self._categorize_error(primary_error)
        severity = self._compute_severity(primary_error['deviation_deg'])
        
        key = (exercise, error_type, severity)
        
        # Fallback to general feedback if specific template not found
        if key not in self.FEEDBACK_TEMPLATES:
            key = (exercise, 'perfect', None)
        
        return self._get_template_variant(key)
    
    def _get_template_variant(self, key):
        """Cycle through template variants to avoid repetition."""
        if key not in self.FEEDBACK_TEMPLATES:
            return "Keep going! You're doing well!"
        
        templates = self.FEEDBACK_TEMPLATES[key]
        if key not in self.template_index:
            self.template_index[key] = 0
        
        idx = self.template_index[key]
        feedback = templates[idx]
        
        # Cycle to next variant
        self.template_index[key] = (idx + 1) % len(templates)
        
        return feedback
    
    def _categorize_error(self, error: Dict) -> str:
        """Map error type to feedback category."""
        error_type = error.get('type', 'unknown')
        joint = error.get('joint', '')
        
        if 'elbow' in joint.lower() and 'unstable' in error_type:
            return 'elbow_stability'
        elif 'elbow' in joint.lower():
            return 'curl_depth'
        elif 'knee' in joint.lower() or 'hip' in joint.lower():
            return 'squat_depth'
        else:
            return 'keep_going'
    
    def _compute_severity(self, deviation_deg: float) -> str:
        """Categorize deviation severity."""
        if deviation_deg < 10:
            return 'low'
        elif deviation_deg < 20:
            return 'moderate'
        else:
            return 'high'
    
    def _generate_with_api(self, error_record: Dict) -> str:
        """
        Generate feedback using actual LLM API.
        
        This would make a real API call to GPT-4-mini or similar.
        Left as placeholder for future implementation.
        """
        # System prompt
        system_prompt = """You are a concise fitness coach.
Use short, motivating sentences and refer to joint names naturally.
Avoid jargon or negative wording. Keep responses under 15 words."""
        
        # User prompt with structured data
        user_prompt = f"""Given this exercise analysis, provide brief coaching feedback:
{json.dumps(error_record, indent=2)}

Provide 1-2 short sentences of actionable, positive feedback."""
        
        # TODO: Implement actual API call
        # For now, fall back to template
        return self._generate_from_template(error_record)
    
    def generate_session_summary(self, session_data: Dict) -> str:
        """
        Generate comprehensive post-session summary.
        
        This can be more detailed since latency is not critical.
        
        Args:
            session_data: {
                "total_reps": 15,
                "success_rate": 0.8,
                "mistakes": {"elbow_stability": 3, "curl_depth": 2},
                "avg_tempo": 2.5,
                "exercise": "bicep_curls"
            }
        """
        total_reps = session_data.get('total_reps', 0)
        success_rate = session_data.get('success_rate', 0.0)
        mistakes = session_data.get('mistakes', {})
        exercise = session_data.get('exercise', 'your workout')
        
        summary_parts = []
        
        # Opening
        summary_parts.append(f"Great session! You completed {total_reps} reps")
        
        # Success rate
        if success_rate >= 0.9:
            summary_parts.append("with excellent form throughout.")
        elif success_rate >= 0.7:
            summary_parts.append(f"with {int(success_rate * 100)}% good form.")
        else:
            summary_parts.append(f"Keep practicing—{int(success_rate * 100)}% were solid.")
        
        # Main feedback on mistakes
        if mistakes:
            most_common = max(mistakes.items(), key=lambda x: x[1])
            mistake_name = most_common[0].replace('_', ' ')
            count = most_common[1]
            summary_parts.append(f"\nI noticed {mistake_name} came up {count} times.")
            summary_parts.append("Let's focus on that next session.")
        
        # Encouragement
        summary_parts.append("Your consistency is improving—keep it up! 💪")
        
        return " ".join(summary_parts)
