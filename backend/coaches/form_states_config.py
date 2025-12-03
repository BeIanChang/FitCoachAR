FORM_STATES_CONFIG = {
    "squats": {
        "GOOD_REP": {
            "description": "A well-executed squat with good depth and form.",
            "rules": [
                {"primitive": "squat_depth", "must_be": ["deep"]},
                {"primitive": "knee_stability", "must_not_be": ["unstable", "slight_wobble"]},
                {"primitive": "torso_angle", "must_not_be": ["bent_over", "leaning"]}
            ]
        },
        "SHALLOW_DEPTH": {
            "description": "The user did not go deep enough.",
            "rules": [
                {"primitive": "squat_depth", "must_be": ["shallow", "parallel"]}
            ]
        },
        "KNEE_COLLAPSE": {
            "description": "Knees caved inward during the movement.",
            "rules": [
                {"primitive": "knee_stability", "must_be": ["unstable", "slight_wobble"]}
            ]
        },
        "CHEST_FALL": {
            "description": "The user leaned too far forward.",
            "rules": [
                {"primitive": "torso_angle", "must_be": ["bent_over", "leaning"]}
            ]
        },
        "RUSHED_DESCENT": {
            "description": "The user dropped into the squat too quickly, losing control.",
            "rules": [
                {"primitive": "descent_speed", "must_be": ["dive"]}
            ]
        },
        "UNSTABLE_MOVEMENT": {
            "description": "The movement was shaky or jerky, indicating a loss of balance.",
            "rules": [
                {"primitive": "movement_smoothness", "must_be": ["unstable", "shaky"]}
            ]
        },
        "HIP_ASYMMETRY": {
            "description": "The user's hips were not level, indicating a shift to one side.",
            "rules": [
                {"primitive": "hip_shift", "must_be": ["major_shift", "slight_shift"]}
            ]
        }
    },
    "bicep_curls": {
        "GOOD_REP": {
            "description": "A well-executed curl with full contraction and no excessive swing.",
            "rules": [
                {"primitive": "peak_flexion", "must_be": ["full_contraction", "partial_contraction"]},
                {"primitive": "swing_momentum", "must_not_be": ["body_swing"]}
            ]
        },
        "INCOMPLETE_RANGE": {
            "description": "The user did not lift the weight high enough.",
            "rules": [
                {"primitive": "peak_flexion", "must_be": ["incomplete_rep", "partial_contraction"]}
            ]
        },
        "BODY_SWING": {
            "description": "The user used their back and shoulders to lift the weight.",
            "rules": [
                {"primitive": "swing_momentum", "must_be": ["body_swing"]}
            ]
        },
        "ELBOW_DRIFT": {
            "description": "The elbow moved forward or backward during the curl.",
            "rules": [
                {"primitive": "elbow_position", "must_be": ["major_drift", "slight_drift"]}
            ]
        },
        "UNCONTROLLED_LOWER": {
            "description": "The user dropped the weight instead of lowering it with control.",
            "rules": [
                {"primitive": "lower_speed", "must_be": ["fast_drop"]}
            ]
        },
        "WRIST_STRAIN": {
            "description": "The wrist was bent at an improper angle, risking injury.",
            "rules": [
                {"primitive": "wrist_angle", "must_be": ["flexed_or_extended"]}
            ]
        },
        "NO_SQUEEZE": {
            "description": "The user did not pause at the top to contract the muscle.",
            "rules": [
                {"primitive": "pause_at_top", "must_be": ["touch_and_go"]}
            ]
        }
    }
}
