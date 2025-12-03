"""
Form Analyzer - Analyzes workout form using primitives and states.

This module takes the raw form data collected during a workout and:
1. Categorizes each primitive value using the thresholds in form_primitives_config.py
2. Determines which FormStates apply to each rep using form_states_config.py
3. Aggregates the results into a comprehensive workout summary.
"""

from typing import Dict, Any, List, Optional
from .form_primitives_config import PRIMITIVES_CONFIG
from .form_states_config import FORM_STATES_CONFIG


class FormAnalyzer:
    """
    Analyzes workout form based on collected primitive data.
    
    Usage:
        analyzer = FormAnalyzer()
        
        # Categorize a raw value
        category = analyzer.categorize_primitive("squats", "squat_depth", 85)
        # Returns: "deep"
        
        # Analyze a full session
        summary = analyzer.analyze_session(workout_session)
    """
    
    def __init__(self):
        self.primitives_config = PRIMITIVES_CONFIG
        self.states_config = FORM_STATES_CONFIG
    
    def categorize_primitive(
        self, 
        exercise: str, 
        primitive_name: str, 
        value: float
    ) -> Optional[str]:
        """
        Categorize a raw primitive value into its category name.
        
        Args:
            exercise: "squats" or "bicep_curls"
            primitive_name: e.g., "squat_depth", "descent_speed"
            value: The raw measured value
            
        Returns:
            Category name (e.g., "deep", "controlled") or None if not found
        """
        exercise_config = self.primitives_config.get(exercise, {})
        
        # Check both static and dynamic primitives
        primitive_config = None
        for ptype in ["static", "dynamic"]:
            if primitive_name in exercise_config.get(ptype, {}):
                primitive_config = exercise_config[ptype][primitive_name]
                break
        
        if not primitive_config:
            return None
        
        categories = primitive_config.get("categories", [])
        
        for cat in categories:
            v_min = cat.get("v_min")
            v_max = cat.get("v_max")
            
            # Check if value falls within this category's range
            min_ok = (v_min is None) or (value >= v_min)
            max_ok = (v_max is None) or (value < v_max)
            
            if min_ok and max_ok:
                return cat["name"]
        
        return None
    
    def determine_form_states(
        self, 
        exercise: str, 
        primitives: Dict[str, str]
    ) -> List[str]:
        """
        Determine which FormStates apply given a set of categorized primitives.
        
        Args:
            exercise: "squats" or "bicep_curls"
            primitives: Dict mapping primitive names to their categories
                        e.g., {"squat_depth": "deep", "knee_stability": "stable"}
        
        Returns:
            List of FormState names that apply, e.g., ["GOOD_REP"] or ["SHALLOW_DEPTH", "KNEE_COLLAPSE"]
        """
        states_config = self.states_config.get(exercise, {})
        active_states = []
        
        for state_name, state_def in states_config.items():
            rules = state_def.get("rules", [])
            state_applies = True
            
            for rule in rules:
                primitive_name = rule.get("primitive")
                must_be = rule.get("must_be", [])
                must_not_be = rule.get("must_not_be", [])
                
                actual_category = primitives.get(primitive_name)
                
                # Check must_be condition
                if must_be and actual_category not in must_be:
                    state_applies = False
                    break
                
                # Check must_not_be condition
                if must_not_be and actual_category in must_not_be:
                    state_applies = False
                    break
            
            if state_applies:
                active_states.append(state_name)
        
        return active_states
    
    def analyze_session(self, session) -> Dict[str, Any]:
        """
        Analyze a complete workout session.
        
        Args:
            session: A WorkoutSession object with form_snapshots in each ExerciseSet
        
        Returns:
            A comprehensive analysis dictionary
        """
        analysis = {
            "session_id": session.id,
            "session_name": session.name,
            "total_reps": 0,
            "total_good_reps": 0,
            "exercises": {},
            "overall_score": 0.0,
        }
        
        all_states_count = {}
        
        for exercise_set in session.sets:
            exercise = exercise_set.exercise
            
            if exercise not in analysis["exercises"]:
                analysis["exercises"][exercise] = {
                    "total_reps": 0,
                    "good_reps": 0,
                    "form_states_count": {},
                    "primitive_averages": {},
                }
            
            ex_analysis = analysis["exercises"][exercise]
            
            for snapshot in exercise_set.form_snapshots:
                analysis["total_reps"] += 1
                ex_analysis["total_reps"] += 1
                
                # Count form states
                for state in snapshot.form_states:
                    ex_analysis["form_states_count"][state] = \
                        ex_analysis["form_states_count"].get(state, 0) + 1
                    all_states_count[state] = all_states_count.get(state, 0) + 1
                
                # Track if this was a good rep
                if "GOOD_REP" in snapshot.form_states:
                    analysis["total_good_reps"] += 1
                    ex_analysis["good_reps"] += 1
                
                # Accumulate primitive values for averaging
                all_primitives = {
                    **snapshot.static_primitives,
                    **snapshot.dynamic_primitives
                }
                for prim_name, prim_data in all_primitives.items():
                    if prim_name not in ex_analysis["primitive_averages"]:
                        ex_analysis["primitive_averages"][prim_name] = {
                            "sum": 0,
                            "count": 0,
                            "categories": {}
                        }
                    
                    prim_avg = ex_analysis["primitive_averages"][prim_name]
                    prim_avg["sum"] += prim_data.get("value", 0)
                    prim_avg["count"] += 1
                    
                    cat = prim_data.get("category", "unknown")
                    prim_avg["categories"][cat] = prim_avg["categories"].get(cat, 0) + 1
        
        # Calculate averages and finalize
        for exercise, ex_data in analysis["exercises"].items():
            for prim_name, prim_data in ex_data["primitive_averages"].items():
                if prim_data["count"] > 0:
                    prim_data["average"] = prim_data["sum"] / prim_data["count"]
                del prim_data["sum"]
                del prim_data["count"]
            
            # Calculate exercise score
            if ex_data["total_reps"] > 0:
                ex_data["score"] = round(
                    (ex_data["good_reps"] / ex_data["total_reps"]) * 100, 1
                )
            else:
                ex_data["score"] = 0.0
        
        # Calculate overall score
        if analysis["total_reps"] > 0:
            analysis["overall_score"] = round(
                (analysis["total_good_reps"] / analysis["total_reps"]) * 100, 1
            )
        
        # Add top issues
        analysis["top_issues"] = self._get_top_issues(all_states_count)
        
        return analysis
    
    def _get_top_issues(
        self, 
        states_count: Dict[str, int], 
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get the top N most frequent form issues.
        """
        # Filter out GOOD_REP as it's not an issue
        issues = {k: v for k, v in states_count.items() if k != "GOOD_REP"}
        
        # Sort by count descending
        sorted_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)
        
        top_issues = []
        for state_name, count in sorted_issues[:top_n]:
            # Find description from config
            description = ""
            for exercise_states in self.states_config.values():
                if state_name in exercise_states:
                    description = exercise_states[state_name].get("description", "")
                    break
            
            top_issues.append({
                "state": state_name,
                "count": count,
                "description": description
            })
        
        return top_issues
