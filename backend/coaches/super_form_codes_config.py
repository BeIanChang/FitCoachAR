"""
Super Form Codes Configuration

Super Form Codes are composite states derived from Form Codes.
They represent high-level form assessments like "GOOD_REP", "BODY_SWING", etc.
Each Super Form Code has rules that check Form Code categories.
"""

SUPER_FORM_CODES_CONFIG = {
    "squats": {
        "GOOD_REP": {
            "description": "A well-executed squat with good depth and form.",
            "rules": [
                {"form_code": "squat_depth", "must_be": ["deep"]},
                {"form_code": "knee_stability", "must_not_be": ["unstable", "slight_wobble"]},
                {"form_code": "torso_angle", "must_not_be": ["bent_over", "leaning"]}
            ]
        },
        "SHALLOW_DEPTH": {
            "description": "The user did not go deep enough.",
            "rules": [
                {"form_code": "squat_depth", "must_be": ["shallow", "parallel"]}
            ]
        },
        "KNEE_COLLAPSE": {
            "description": "Knees caved inward during the movement.",
            "rules": [
                {"form_code": "knee_stability", "must_be": ["unstable", "slight_wobble"]}
            ]
        },
        "CHEST_FALL": {
            "description": "The user leaned too far forward.",
            "rules": [
                {"form_code": "torso_angle", "must_be": ["bent_over", "leaning"]}
            ]
        },
        "RUSHED_DESCENT": {
            "description": "The user dropped into the squat too quickly, losing control.",
            "rules": [
                {"form_code": "descent_speed", "must_be": ["dive"]}
            ]
        },
        "UNSTABLE_MOVEMENT": {
            "description": "The movement was shaky or jerky, indicating a loss of balance.",
            "rules": [
                {"form_code": "movement_smoothness", "must_be": ["unstable", "shaky"]}
            ]
        },
        "HIP_ASYMMETRY": {
            "description": "The user's hips were not level, indicating a shift to one side.",
            "rules": [
                {"form_code": "hip_shift", "must_be": ["major_shift", "slight_shift"]}
            ]
        }
    },
    "bicep_curls": {
        "GOOD_REP": {
            "description": "A well-executed curl with full contraction and no excessive swing.",
            "rules": [
                {"form_code": "peak_flexion", "must_be": ["full_contraction", "partial_contraction"]},
                {"form_code": "swing_momentum", "must_not_be": ["body_swing"]}
            ]
        },
        "INCOMPLETE_RANGE": {
            "description": "The user did not lift the weight high enough.",
            "rules": [
                {"form_code": "peak_flexion", "must_be": ["incomplete_rep", "partial_contraction"]}
            ]
        },
        "BODY_SWING": {
            "description": "The user used their back and shoulders to lift the weight.",
            "rules": [
                {"form_code": "swing_momentum", "must_be": ["body_swing"]}
            ]
        },
        "ELBOW_DRIFT": {
            "description": "The elbow moved forward or backward during the curl.",
            "rules": [
                {"form_code": "elbow_position", "must_be": ["major_drift", "slight_drift"]}
            ]
        },
        "UNCONTROLLED_LOWER": {
            "description": "The user dropped the weight instead of lowering it with control.",
            "rules": [
                {"form_code": "lower_speed", "must_be": ["fast_drop"]}
            ]
        },
        "WRIST_STRAIN": {
            "description": "The wrist was bent at an improper angle, risking injury.",
            "rules": [
                {"form_code": "wrist_angle", "must_be": ["flexed_or_extended"]}
            ]
        },
        "NO_SQUEEZE": {
            "description": "The user did not pause at the top to contract the muscle.",
            "rules": [
                {"form_code": "pause_at_top", "must_be": ["touch_and_go"]}
            ]
        }
    }
}
