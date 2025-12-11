"""
FitCoachAR Coaching System

Four components:
1. RealtimeCoach - Fast, rule-based feedback during workout (no LLM)
2. LLMCoach - AI-powered session summaries and Q&A (post-workout)
3. FormAnalyzer - Analyzes workout form using Form Codes and Super Form Codes
4. FormChecker - Real-time form feedback based on calibration params
"""

from .realtime_coach import RealtimeCoach
from .llm_coach import LLMCoach
from .form_analyzer import FormAnalyzer, FormChecker

__all__ = ["RealtimeCoach", "LLMCoach", "FormAnalyzer", "FormChecker"]
