"""
LLM Coach - AI-Powered Post-Workout Coaching

Provides intelligent feedback after workout sessions:
1. Session Summary - Comprehensive review of workout performance
2. Q&A - Answer user questions about their workout

Uses Cerebras API with Llama 3.3 70B model.
"""

import json
import os
from typing import Dict, Any, Optional

try:
    from cerebras.cloud.sdk import Cerebras
except Exception:
    Cerebras = None


class LLMCoach:
    """
    AI-powered coaching for post-workout feedback and Q&A.
    
    Usage:
        coach = LLMCoach(api_key="your-key")
        
        # After workout:
        summary = coach.generate_session_summary(session_data)
        
        # User asks a question:
        answer = coach.answer_question(session_data, "How was my form?")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM Coach.
        
        Args:
            api_key: Cerebras API key. Falls back to CEREBRAS_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        self.client = None
        
        if self.api_key and Cerebras is not None:
            try:
                self.client = Cerebras(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Cerebras client: {e}")
                self.client = None
    
    @property
    def is_available(self) -> bool:
        """Check if LLM is available for use."""
        return self.client is not None
    
    def generate_session_summary(self, session_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive post-workout summary.
        
        Args:
            session_data: Dict containing:
                - total_reps: int
                - success_rate: float (0.0 to 1.0)
                - mistakes: Dict[str, int] (error type -> count)
                - avg_tempo: float (seconds per rep)
                - exercise: str
                - session_details: Optional dict with full session info
                
        Returns:
            Natural language summary string
        """
        if not self.is_available:
            return self._generate_fallback_summary(session_data)
        
        # Check if we have detailed form analysis
        form_analysis = session_data.get('form_analysis')
        
        if form_analysis:
            system_prompt = """You are a friendly and encouraging fitness coach summarizing a workout session.
You have detailed form analysis data including a form score, good reps count, and specific issues detected.

Provide a concise summary (3-4 sentences) that:
1. Opens with positive acknowledgment of their effort
2. Mentions total reps and the FORM SCORE (use the score from form_analysis, not success_rate)
3. If there are issues in top_issues, mention the most common one as a constructive tip
4. Ends with motivation for their next session

IMPORTANT: Use the form_analysis.score as the accuracy percentage, NOT success_rate.
Keep the tone supportive and actionable. Format as a single paragraph."""

            # Build a cleaner data summary for the LLM
            top_issues_text = ""
            if form_analysis.get('top_issues'):
                issues = [f"{issue.get('super_form_code', issue.get('state', 'unknown')).replace('_', ' ')}: {issue.get('description', '')}" 
                         for issue in form_analysis['top_issues'][:2]]
                top_issues_text = "\n".join(issues)
            
            user_prompt = f"""Workout: {session_data.get('exercise', 'workout')}
Total Reps: {form_analysis.get('total_reps', session_data.get('total_reps', 0))}
Good Reps: {form_analysis.get('good_reps', 0)}
Form Score: {form_analysis.get('score', 0)}%

Top Issues Detected:
{top_issues_text if top_issues_text else "None - great form!"}

Please provide a summary that aligns with this form analysis."""
        else:
            system_prompt = """You are a friendly and encouraging fitness coach summarizing a workout session.
Provide a concise summary (3-4 sentences) that:
1. Opens with positive acknowledgment of their effort
2. Mentions total reps and form accuracy percentage
3. Points out the most common mistake as a constructive tip for next time
4. Ends with motivation for their next session

Keep the tone supportive and actionable. Format as a single paragraph."""

            user_prompt = f"""Here is the workout data for a {session_data.get('exercise', 'workout')} session:
{json.dumps(session_data, indent=2)}

Please provide a summary."""

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b",
                max_completion_tokens=200,
                temperature=0.3,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM summary error: {e}")
            return self._generate_fallback_summary(session_data)
    
    def answer_question(self, session_data: Dict[str, Any], question: str) -> str:
        """
        Answer a user's question about their workout.
        
        Args:
            session_data: Workout session data
            question: User's question
            
        Returns:
            Natural language answer
        """
        if not self.is_available:
            return "I can only answer questions when connected to the AI coach. Please check the API configuration."
        
        system_prompt = """You are a knowledgeable and conversational fitness coach. A user has a question after their workout.

Guidelines:
- Answer the question directly and concisely
- If about workout performance, use the provided data to give supportive, actionable advice
- If a greeting or off-topic, respond briefly and friendly
- Keep responses under 3 sentences when possible
- Be encouraging but honest"""

        user_prompt = f"""My workout data:
{json.dumps(session_data, indent=2)}

My question: {question}"""

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b",
                max_completion_tokens=150,
                temperature=0.2,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Q&A error: {e}")
            return "I'm having trouble connecting right now. Please try again in a moment."
    
    def get_improvement_tips(self, session_data: Dict[str, Any]) -> str:
        """
        Get specific improvement tips based on workout mistakes.
        
        Args:
            session_data: Workout session data with mistakes
            
        Returns:
            Actionable improvement tips
        """
        if not self.is_available:
            return self._generate_fallback_tips(session_data)
        
        mistakes = session_data.get("mistakes", {})
        if not mistakes:
            return "Great job! Your form was solid throughout. Keep up the consistency!"
        
        system_prompt = """You are a fitness coach providing specific improvement tips.
Based on the workout mistakes, give 2-3 actionable tips to improve.
Be specific about body positioning and movement cues.
Keep it concise and encouraging."""

        user_prompt = f"""Exercise: {session_data.get('exercise', 'workout')}
Mistakes during session: {json.dumps(mistakes)}

What should I focus on to improve?"""

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b",
                max_completion_tokens=150,
                temperature=0.3,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM tips error: {e}")
            return self._generate_fallback_tips(session_data)
    
    def _generate_fallback_summary(self, session_data: Dict[str, Any]) -> str:
        """Generate summary without LLM (template-based fallback)."""
        total_reps = session_data.get('total_reps', 0)
        success_rate = session_data.get('success_rate', 0.0)
        mistakes = session_data.get('mistakes', {})
        exercise = session_data.get('exercise', 'your workout')
        
        parts = []
        
        # Opening
        parts.append(f"Great session! You completed {total_reps} reps of {exercise}")
        
        # Success rate
        if success_rate >= 0.9:
            parts.append("with excellent form throughout.")
        elif success_rate >= 0.7:
            parts.append(f"with {int(success_rate * 100)}% good form.")
        else:
            parts.append(f"— keep practicing, {int(success_rate * 100)}% were solid.")
        
        # Main feedback
        if mistakes:
            most_common = max(mistakes.items(), key=lambda x: x[1])
            mistake_name = most_common[0].replace('_', ' ')
            count = most_common[1]
            parts.append(f"Focus area: {mistake_name} came up {count} times.")
        
        # Encouragement
        parts.append("Keep up the great work! 💪")
        
        return " ".join(parts)
    
    def _generate_fallback_tips(self, session_data: Dict[str, Any]) -> str:
        """Generate tips without LLM (template-based fallback)."""
        mistakes = session_data.get("mistakes", {})
        exercise = session_data.get("exercise", "")
        
        tips = []
        
        if "elbow_stability" in mistakes:
            tips.append("Keep your elbow pinned to your side throughout the curl.")
        if "go_deeper" in mistakes or "squat_depth" in mistakes:
            tips.append("Work on hip mobility to achieve better squat depth.")
        if "knees_out" in mistakes or "knee_valgus" in mistakes:
            tips.append("Focus on pushing your knees out over your toes.")
        if "chest_up" in mistakes or "torso_lean" in mistakes:
            tips.append("Engage your core and keep your chest proud.")
        
        if not tips:
            tips.append("Keep practicing with focus on controlled movements.")
        
        return " ".join(tips)
