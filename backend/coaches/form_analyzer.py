"""
Form Analyzer - Analyzes workout form using FormCodes and Super FormCodes.

This module takes the raw form data collected during a workout and:
1. Categorizes each FormCode value using the thresholds in form_codes_config.py
2. Determines which Super FormCodes apply to each rep using super_form_codes_config.py
3. Aggregates the results into a comprehensive workout summary.

Also includes FormChecker for real-time form feedback during reps.
"""

from typing import Dict, Any, List, Optional

# Import CalibrationParams for FormChecker (optional dependency)
try:
    from calibration_v2 import CalibrationParams
except ImportError:
    CalibrationParams = None  # type: ignore

from .form_codes_config import FORM_CODES_CONFIG
from .super_form_codes_config import SUPER_FORM_CODES_CONFIG


class FormChecker:
    """
    Adaptive Form Checker for real-time feedback.
    Compares live metrics against User Capabilities (Personal Baseline)
    and Form Constraints (Stability).
    """
    
    def __init__(self, params):
        """
        Args:
            params: CalibrationParams object with capabilities and form_constraints
        """
        self.params = params

    def check(self, current_metrics: Dict[str, float], rep_progress: float, critic: float = 0.5) -> List[str]:
        """
        Evaluate form quality in real-time.

        Args:
            current_metrics: Dict of primary and secondary metrics.
            rep_progress: 0.0 (Bottom) to 1.0 (Top). Used to ignore warnings in wrong phases.
            critic: 0.0 (Loose) to 1.0 (Strict).
            
        Returns:
            List of feedback strings for the user.
        """
        feedback = []
        user_caps = self.params.capabilities
        constraints = self.params.form_constraints

        # Tolerance Multiplier:
        # Critic 0.0 -> Multiplier 2.0 (Forgiving)
        # Critic 0.5 -> Multiplier 1.4
        # Critic 1.0 -> Multiplier 0.8 (Strict)
        tolerance_multiplier = 2.0 - (1.2 * critic)

        # 1. CHECK RANGE OF MOTION (Primary Metric)
        # Only check this near the "turnaround" point (Bottom phase, progress < 0.3)
        if "primary" in current_metrics and "rom_min" in user_caps:
            val = current_metrics["primary"]
            target = user_caps["rom_min"]

            # Allow gap based on critic
            allowed_gap = 15.0 * tolerance_mp

            if rep_progress < 0.3 and val > (target + allowed_gap):
                feedback.append("Go deeper!")

        # 2. CHECK FORM STABILITY (Secondary Metrics)
        # Check these continuously
        for metric, value in current_metrics.items():
            if metric == "primary":
                continue

            if metric in constraints:
                c = constraints[metric]
                user_limit = c["max"]  # Worst value seen during calibration

                # Strict limit
                limit = user_limit * tolerance_multiplier

                if value > limit:
                    msg = self._get_feedback_string(metric)
                    if msg:
                        feedback.append(msg)

            # Special Case: Knee Valgus (Ratio must be HIGH, not LOW)
            if metric == "knee_valgus_index":
                if value < 0.8:
                    feedback.append("Push knees out")

            # Special Case: Symmetry (Hip Tilt)
            if metric == "hip_symmetry" and "hip_symmetry_max" in self.params.symmetry_profile:
                # Check if current asymmetry is significantly worse than user's baseline
                baseline = self.params.symmetry_profile["hip_symmetry_max"]
                if value > (baseline * 2.0):  # Allow 2x baseline before complaining
                    feedback.append("Even out your hips")

        return list(set(feedback))

    def _get_feedback_string(self, metric: str) -> str:
        """Map technical metric names to UI strings."""
        mapping = {
            "elbow_stability": "Keep elbow tight",
            "torso_swing": "Don't swing body",
            "torso_lean": "Keep chest up",
            "body_linearity": "Don't sag hips",
            "knee_valgus_index": "Push knees out",
            "hip_symmetry": "Even out hips"
        }
        return mapping.get(metric, "")


class FormAnalyzer:
    """
    Analyzes workout form based on collected Form Code data.
    
    Usage:
        analyzer = FormAnalyzer()
        
        # Categorize a raw Form Code value
        category = analyzer.categorize_form_code("squats", "squat_depth", 85)
        # Returns: "deep"
        
        # Determine Super Form Codes from categorized Form Codes
        super_codes = analyzer.determine_super_form_codes("squats", form_codes_dict)
        
        # Analyze a full session
        summary = analyzer.analyze_session(workout_session)
    """
    
    def __init__(self):
        self.form_codes_config = FORM_CODES_CONFIG
        self.super_form_codes_config = SUPER_FORM_CODES_CONFIG
    
    def categorize_form_code(
        self, 
        exercise: str, 
        form_code_name: str, 
        value: float
    ) -> Optional[str]:
        """
        Categorize a raw Form Code value into its category name.
        
        Args:
            exercise: "squats" or "bicep_curls"
            form_code_name: e.g., "squat_depth", "descent_speed"
            value: The raw measured value
            
        Returns:
            Category name (e.g., "deep", "controlled") or None if not found
        """
        exercise_config = self.form_codes_config.get(exercise, {})
        
        # Check both static and dynamic form codes
        form_code_config = None
        for ptype in ["static", "dynamic"]:
            if form_code_name in exercise_config.get(ptype, {}):
                form_code_config = exercise_config[ptype][form_code_name]
                break
        
        if not form_code_config:
            return None
        
        categories = form_code_config.get("categories", [])
        
        for cat in categories:
            v_min = cat.get("v_min")
            v_max = cat.get("v_max")
            
            # Check if value falls within this category's range
            min_ok = (v_min is None) or (value >= v_min)
            max_ok = (v_max is None) or (value < v_max)
            
            if min_ok and max_ok:
                return cat["name"]
        
        return None
    
    def determine_super_form_codes(
        self, 
        exercise: str, 
        form_codes: Dict[str, str]
    ) -> List[str]:
        """
        Determine which Super Form Codes apply given a set of categorized Form Codes.
        
        Args:
            exercise: "squats" or "bicep_curls"
            form_codes: Dict mapping Form Code names to their categories
                        e.g., {"squat_depth": "deep", "knee_stability": "stable"}
        
        Returns:
            List of Super Form Code names that apply, e.g., ["GOOD_REP"] or ["SHALLOW_DEPTH", "KNEE_COLLAPSE"]
        """
        super_codes_config = self.super_form_codes_config.get(exercise, {})
        active_super_codes = []
        
        for super_code_name, super_code_def in super_codes_config.items():
            rules = super_code_def.get("rules", [])
            code_applies = True
            
            for rule in rules:
                form_code_name = rule.get("form_code")
                must_be = rule.get("must_be", [])
                must_not_be = rule.get("must_not_be", [])
                
                actual_category = form_codes.get(form_code_name)
                
                # Check must_be condition
                if must_be and actual_category not in must_be:
                    code_applies = False
                    break
                
                # Check must_not_be condition
                if must_not_be and actual_category in must_not_be:
                    code_applies = False
                    break
            
            if code_applies:
                active_super_codes.append(super_code_name)
        
        return active_super_codes
    
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
        
        all_super_codes_count = {}
        
        for exercise_set in session.sets:
            exercise = exercise_set.exercise
            
            if exercise not in analysis["exercises"]:
                analysis["exercises"][exercise] = {
                    "total_reps": 0,
                    "good_reps": 0,
                    "super_form_codes_count": {},
                    "form_code_averages": {},
                }
            
            ex_analysis = analysis["exercises"][exercise]
            
            for snapshot in exercise_set.form_snapshots:
                analysis["total_reps"] += 1
                ex_analysis["total_reps"] += 1
                
                # Count super form codes
                for super_code in snapshot.form_states:
                    ex_analysis["super_form_codes_count"][super_code] = \
                        ex_analysis["super_form_codes_count"].get(super_code, 0) + 1
                    all_super_codes_count[super_code] = all_super_codes_count.get(super_code, 0) + 1
                
                # Track if this was a good rep
                if "GOOD_REP" in snapshot.form_states:
                    analysis["total_good_reps"] += 1
                    ex_analysis["good_reps"] += 1
                
                # Accumulate form code values for averaging
                all_form_codes = {
                    **snapshot.static_primitives,
                    **snapshot.dynamic_primitives
                }
                for code_name, code_data in all_form_codes.items():
                    if code_name not in ex_analysis["form_code_averages"]:
                        ex_analysis["form_code_averages"][code_name] = {
                            "sum": 0,
                            "count": 0,
                            "categories": {}
                        }
                    
                    code_avg = ex_analysis["form_code_averages"][code_name]
                    code_avg["sum"] += code_data.get("value", 0)
                    code_avg["count"] += 1
                    
                    cat = code_data.get("category", "unknown")
                    code_avg["categories"][cat] = code_avg["categories"].get(cat, 0) + 1
        
        # Calculate averages and finalize
        for exercise, ex_data in analysis["exercises"].items():
            for code_name, code_data in ex_data["form_code_averages"].items():
                if code_data["count"] > 0:
                    code_data["average"] = code_data["sum"] / code_data["count"]
                del code_data["sum"]
                del code_data["count"]
            
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
        analysis["top_issues"] = self._get_top_issues(all_super_codes_count)
        
        return analysis
    
    def _get_top_issues(
        self, 
        super_codes_count: Dict[str, int], 
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get the top N most frequent form issues (Super Form Codes).
        """
        # Filter out GOOD_REP as it's not an issue
        issues = {k: v for k, v in super_codes_count.items() if k != "GOOD_REP"}
        
        # Sort by count descending
        sorted_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)
        
        top_issues = []
        for super_code_name, count in sorted_issues[:top_n]:
            # Find description from config
            description = ""
            for exercise_codes in self.super_form_codes_config.values():
                if super_code_name in exercise_codes:
                    description = exercise_codes[super_code_name].get("description", "")
                    break
            
            top_issues.append({
                "super_form_code": super_code_name,
                "count": count,
                "description": description
            })
        
        return top_issues

    # Backward compatibility aliases
    def categorize_primitive(self, exercise: str, primitive_name: str, value: float) -> Optional[str]:
        """Alias for categorize_form_code (backward compatibility)."""
        return self.categorize_form_code(exercise, primitive_name, value)
    
    def determine_form_states(self, exercise: str, primitives: Dict[str, str]) -> List[str]:
        """Alias for determine_super_form_codes (backward compatibility)."""
        return self.determine_super_form_codes(exercise, primitives)
