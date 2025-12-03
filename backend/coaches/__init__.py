"""
FitCoachAR Coaching System

Three components:
1. RealtimeCoach - Fast, rule-based feedback during workout (no LLM)
2. LLMCoach - AI-powered session summaries and Q&A (post-workout)
3. FormAnalyzer - Analyzes workout form using primitives and states
"""

from .realtime_coach import RealtimeCoach
from .llm_coach import LLMCoach
from .form_analyzer import FormAnalyzer

__all__ = ["RealtimeCoach", "LLMCoach", "FormAnalyzer"]
