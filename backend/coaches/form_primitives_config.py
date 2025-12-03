PRIMITIVES_CONFIG = {
    "squats": {
        "static": {
            "squat_depth": {
                "type": "angle",
                "description": "Angle of the knee joint at the bottom of the squat.",
                "unit": "degrees",
                "categories": [
                    {"name": "deep", "v_max": 90},
                    {"name": "parallel", "v_min": 90, "v_max": 110},
                    {"name": "shallow", "v_min": 110}
                ]
            },
            "torso_angle": {
                "type": "pitch_roll",
                "description": "Angle of torso (hip-shoulder line) from vertical.",
                "unit": "degrees",
                "categories": [
                    {"name": "upright", "v_max": 20},
                    {"name": "leaning", "v_min": 20, "v_max": 45},
                    {"name": "bent_over", "v_min": 45}
                ]
            },
            "stance_width": {
                "type": "distance",
                "description": "Distance between feet relative to shoulder width.",
                "unit": "ratio",
                "categories": [
                    {"name": "narrow", "v_max": 0.9},
                    {"name": "shoulder_width", "v_min": 0.9, "v_max": 1.2},
                    {"name": "wide", "v_min": 1.2}
                ]
            },
            "knee_forward_travel": {
                "type": "relative_position",
                "description": "Horizontal distance between knee and foot on the Z-axis.",
                "unit": "meters",
                "categories": [
                    {"name": "behind_toes", "v_max": 0},
                    {"name": "over_toes", "v_min": 0}
                ]
            }
        },
        "dynamic": {
            "descent_speed": {
                "type": "velocity",
                "description": "Average vertical velocity of hips during descent.",
                "unit": "m/s",
                "categories": [
                    {"name": "controlled", "v_max": 0.5},
                    {"name": "fast", "v_min": 0.5, "v_max": 1.0},
                    {"name": "dive", "v_min": 1.0}
                ]
            },
            "ascent_speed": {
                "type": "velocity",
                "description": "Average vertical velocity of hips during ascent.",
                "unit": "m/s",
                "categories": [
                    {"name": "grind", "v_max": 0.5},
                    {"name": "controlled", "v_min": 0.5, "v_max": 1.0},
                    {"name": "explosive", "v_min": 1.0}
                ]
            },
            "movement_smoothness": {
                "type": "jerk",
                "description": "Average jerk (rate of change of acceleration) of hips.",
                "unit": "m/s^3",
                "categories": [
                    {"name": "smooth", "v_max": 2.5},
                    {"name": "shaky", "v_min": 2.5, "v_max": 5.0},
                    {"name": "unstable", "v_min": 5.0}
                ]
            },
            "knee_stability": {
                "type": "deviation",
                "description": "Maximum horizontal deviation of knees from their center path.",
                "unit": "meters",
                "categories": [
                    {"name": "stable", "v_max": 0.05},
                    {"name": "slight_wobble", "v_min": 0.05, "v_max": 0.1},
                    {"name": "unstable", "v_min": 0.1}
                ]
            },
            "hip_shift": {
                "type": "symmetry",
                "description": "Maximum difference in vertical position between left and right hip.",
                "unit": "meters",
                "categories": [
                    {"name": "level", "v_max": 0.03},
                    {"name": "slight_shift", "v_min": 0.03, "v_max": 0.07},
                    {"name": "major_shift", "v_min": 0.07}
                ]
            }
        }
    },
    "bicep_curls": {
        "static": {
            "peak_flexion": {
                "type": "angle",
                "description": "Minimum angle of the elbow joint at the top of the curl.",
                "unit": "degrees",
                "categories": [
                    {"name": "full_contraction", "v_max": 60},
                    {"name": "partial_contraction", "v_min": 60, "v_max": 90},
                    {"name": "incomplete_rep", "v_min": 90}
                ]
            },
            "wrist_angle": {
                "type": "angle",
                "description": "Angle of wrist relative to the forearm's line.",
                "unit": "degrees",
                "categories": [
                    {"name": "neutral", "v_min": 165, "v_max": 195},
                    {"name": "flexed_or_extended", "v_max": 165, "v_min": 195}  # Represents OR logic
                ]
            },
            "elbow_position": {
                "type": "relative_position",
                "description": "Horizontal drift of the elbow from starting position.",
                "unit": "normalized",
                "categories": [
                    {"name": "anchored", "v_max": 0.08},
                    {"name": "slight_drift", "v_min": 0.08, "v_max": 0.15},
                    {"name": "major_drift", "v_min": 0.15}
                ]
            }
        },
        "dynamic": {
            "lift_speed": {
                "type": "velocity",
                "description": "Average angular velocity of the elbow during the lift.",
                "unit": "degrees/s",
                "categories": [
                    {"name": "slow", "v_max": 100},
                    {"name": "controlled", "v_min": 100, "v_max": 200},
                    {"name": "explosive", "v_min": 200}
                ]
            },
            "lower_speed": {
                "type": "velocity",
                "description": "Average angular velocity of the elbow during the lowering phase.",
                "unit": "degrees/s",
                "categories": [
                    {"name": "controlled", "v_max": 150},
                    {"name": "fast_drop", "v_min": 150}
                ]
            },
            "swing_momentum": {
                "type": "deviation",
                "description": "Maximum forward/backward movement of the shoulder during the curl.",
                "unit": "normalized",
                "categories": [
                    {"name": "no_swing", "v_max": 0.15},
                    {"name": "slight_swing", "v_min": 0.15, "v_max": 0.25},
                    {"name": "body_swing", "v_min": 0.25}
                ]
            },
            "path_arc": {
                "type": "deviation",
                "description": "Deviation of the wrist from a perfect circular arc path.",
                "unit": "meters",
                "categories": [
                    {"name": "clean_arc", "v_max": 0.05},
                    {"name": "wandering_arc", "v_min": 0.05}
                ]
            },
            "pause_at_top": {
                "type": "duration",
                "description": "Time spent with the elbow angle near peak flexion.",
                "unit": "seconds",
                "categories": [
                    {"name": "squeeze", "v_min": 0.5},
                    {"name": "touch_and_go", "v_max": 0.5}
                ]
            }
        }
    }
}
