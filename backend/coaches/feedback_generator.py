"""
Feedback Generator - Creates detailed, human-readable sentences from Form Codes and Super Form Codes.

This module takes the Form Code categories and Super Form Codes and generates
rich, descriptive feedback for each repetition.
"""

from typing import Dict, List, Any, Optional


# Form Code category descriptions - maps each category to a human-readable description
FORM_CODE_DESCRIPTIONS = {
    "bicep_curls": {
        "static": {
            "peak_flexion": {
                "full_contraction": "You achieved a full muscle contraction at the top of the curl.",
                "partial_contraction": "Your curl reached a partial contraction - try to bring the weight closer to your shoulder.",
                "incomplete_rep": "The curl was incomplete - you didn't lift the weight high enough to engage the bicep fully."
            },
            "wrist_angle": {
                "neutral": "Your wrist maintained a neutral, safe position throughout.",
                "flexed_or_extended": "Your wrist was bent during the curl, which can cause strain - keep it straight."
            },
            "elbow_position": {
                "anchored": "Your elbow stayed stable by your side - excellent form!",
                "slight_drift": "Your elbow drifted slightly from your body during the curl.",
                "major_drift": "Your elbow moved significantly forward or backward - try to keep it pinned to your side."
            }
        },
        "dynamic": {
            "lift_speed": {
                "slow": "The lifting phase was slow and controlled.",
                "controlled": "The lifting speed was well-controlled.",
                "explosive": "The lift was explosive - this can reduce muscle engagement if too fast."
            },
            "lower_speed": {
                "controlled": "You lowered the weight with good control (eccentric phase).",
                "fast_drop": "You dropped the weight too quickly - slow down the lowering phase for better gains."
            },
            "swing_momentum": {
                "no_swing": "No body swing detected - you used pure bicep strength.",
                "slight_swing": "There was slight body movement during the curl.",
                "body_swing": "Significant body swing detected - you're using momentum instead of muscle."
            },
            "path_arc": {
                "clean_arc": "The weight followed a clean, circular arc path.",
                "wandering_arc": "The weight path was inconsistent - focus on a smooth arc motion."
            },
            "pause_at_top": {
                "squeeze": "Good squeeze at the top - this maximizes muscle activation!",
                "touch_and_go": "No pause at the top - try holding for a moment to increase tension."
            }
        }
    },
    "squats": {
        "static": {
            "squat_depth": {
                "full_depth": "You achieved full squat depth - great mobility!",
                "parallel": "You reached parallel depth - good standard form.",
                "partial_squat": "The squat was shallow - try to go deeper for full muscle engagement."
            },
            "knee_alignment": {
                "aligned": "Your knees tracked well over your toes.",
                "slight_valgus": "Your knees caved inward slightly - focus on pushing them out.",
                "major_valgus": "Significant knee cave detected - this can cause injury. Push your knees outward."
            },
            "torso_angle": {
                "upright": "Your torso stayed upright - excellent posture!",
                "slight_lean": "There was a slight forward lean in your torso.",
                "excessive_lean": "Excessive forward lean detected - keep your chest up and core tight."
            },
            "hip_shift": {
                "centered": "Your hips stayed centered throughout the movement.",
                "slight_shift": "Your hips shifted slightly to one side.",
                "major_shift": "Significant hip shift detected - this indicates a muscle imbalance."
            }
        },
        "dynamic": {
            "descent_speed": {
                "controlled": "The descent was controlled and steady.",
                "fast_drop": "You dropped too quickly - control the descent for safety."
            },
            "ascent_speed": {
                "powerful": "Strong, powerful drive out of the bottom!",
                "slow_grind": "The ascent was slow - this could indicate the weight is challenging.",
                "sticking_point": "You had a sticking point during the ascent - work on that range."
            },
            "knee_stability": {
                "stable": "Your knees were stable throughout the movement.",
                "wobble": "Some knee wobble detected - strengthen your stabilizer muscles."
            },
            "balance": {
                "centered": "Your weight distribution was well-centered.",
                "forward_shift": "Your weight shifted forward onto your toes.",
                "backward_shift": "Your weight shifted back onto your heels."
            }
        }
    }
}

# Super Form Code summary sentences - used when a specific Super Form Code is detected
SUPER_FORM_CODE_SUMMARIES = {
    "bicep_curls": {
        "GOOD_REP": "This was a well-executed rep with proper form!",
        "INCOMPLETE_RANGE": "Focus on completing the full range of motion.",
        "BODY_SWING": "Reduce body momentum - isolate the bicep muscle.",
        "ELBOW_DRIFT": "Keep your elbow stationary by your side.",
        "UNCONTROLLED_LOWER": "Control the negative (lowering) phase.",
        "WRIST_STRAIN": "Maintain a neutral wrist position.",
        "NO_SQUEEZE": "Add a brief pause at the top of the curl."
    },
    "squats": {
        "GOOD_REP": "This was a well-executed rep with proper form!",
        "SHALLOW_SQUAT": "Work on achieving greater depth.",
        "KNEE_CAVE": "Focus on pushing your knees outward.",
        "FORWARD_LEAN": "Keep your chest up and core engaged.",
        "HIP_ASYMMETRY": "Address the hip shift - check for muscle imbalances."
    }
}


def generate_rep_feedback(
    exercise: str,
    static_form_codes: Dict[str, Any],
    dynamic_form_codes: Dict[str, Any],
    super_form_codes: List[str]
) -> Dict[str, Any]:
    """
    Generate detailed feedback for a single repetition.
    
    Args:
        exercise: The exercise type (e.g., "bicep_curls", "squats")
        static_form_codes: Dict mapping Form Code names to {"value": x, "category": "name"}
        dynamic_form_codes: Dict mapping Form Code names to {"value": x, "category": "name"}
        super_form_codes: List of detected Super Form Codes for this rep
    
    Returns:
        Dict containing:
            - is_good: bool indicating if this was a good rep
            - summary: A brief summary sentence
            - details: List of detailed feedback sentences
            - highlights: Key points to focus on (from Super Form Codes)
    """
    exercise_descriptions = FORM_CODE_DESCRIPTIONS.get(exercise, {})
    exercise_summaries = SUPER_FORM_CODE_SUMMARIES.get(exercise, {})
    
    is_good = "GOOD_REP" in super_form_codes
    details = []
    highlights = []
    
    # Generate details from static form codes
    static_desc = exercise_descriptions.get("static", {})
    for code_name, code_data in static_form_codes.items():
        # Handle both formats: {"value": x, "category": "name"} or just "category_name"
        if isinstance(code_data, dict):
            category = code_data.get("category", "")
        else:
            category = code_data
        
        if code_name in static_desc and category in static_desc[code_name]:
            details.append(static_desc[code_name][category])
    
    # Generate details from dynamic form codes
    dynamic_desc = exercise_descriptions.get("dynamic", {})
    for code_name, code_data in dynamic_form_codes.items():
        # Handle both formats: {"value": x, "category": "name"} or just "category_name"
        if isinstance(code_data, dict):
            category = code_data.get("category", "")
        else:
            category = code_data
        
        if code_name in dynamic_desc and category in dynamic_desc[code_name]:
            details.append(dynamic_desc[code_name][category])
    
    # Generate highlights from super form codes (excluding GOOD_REP)
    for super_code in super_form_codes:
        if super_code != "GOOD_REP" and super_code in exercise_summaries:
            highlights.append(exercise_summaries[super_code])
    
    # Create summary
    if is_good:
        summary = exercise_summaries.get("GOOD_REP", "Good rep!")
    elif highlights:
        summary = highlights[0]  # Use the first issue as the summary
    else:
        summary = "This rep needs improvement."
    
    return {
        "is_good": is_good,
        "summary": summary,
        "details": details,
        "highlights": highlights
    }


def generate_workout_feedback(
    exercise: str,
    snapshots: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate detailed feedback for all reps in a workout.
    
    Args:
        exercise: The exercise type
        snapshots: List of FormSnapshot dicts
    
    Returns:
        List of feedback dicts, one per rep
    """
    feedback_list = []
    
    for snapshot in snapshots:
        static_codes = snapshot.get("static_primitives", {})
        dynamic_codes = snapshot.get("dynamic_primitives", {})
        super_codes = snapshot.get("form_states", [])
        
        feedback = generate_rep_feedback(
            exercise=exercise,
            static_form_codes=static_codes,
            dynamic_form_codes=dynamic_codes,
            super_form_codes=super_codes
        )
        feedback_list.append(feedback)
    
    return feedback_list
