"""
Adaptive Form Checker.
Compares live metrics against User Capabilities (Personal Baseline)
and Form Constraints (Stability).
"""

from typing import Dict, List
from calibration_v2 import CalibrationParams


class FormChecker:
    def __init__(self, params: CalibrationParams):
        print(params)
        self.params = params

    def check(self, current_metrics: Dict[str, float], rep_progress: float, critic: float = 0.5) -> List[str]:
        """
        Evaluate form quality.

        Args:
            current_metrics: Dict of primary and secondary metrics.
            rep_progress: 0.0 (Bottom) to 1.0 (Top). Used to ignore warnings in wrong phases.
            critic: 0.0 (Loose) to 1.0 (Strict).
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
            allowed_gap = 15.0 * tolerance_multiplier

            if rep_progress < 0.3 and val > (target + allowed_gap):
                feedback.append("Go deeper!")

        # 2. CHECK FORM STABILITY (Secondary Metrics)
        # Check these continuously
        for metric, value in current_metrics.items():
            if metric == "primary": continue

            if metric in constraints:
                c = constraints[metric]
                user_limit = c["max"]  # Worst value seen during calibration

                # Strict limit
                limit = user_limit * tolerance_multiplier

                if value > limit:
                    msg = self._get_feedback_string(metric)
                    if msg: feedback.append(msg)

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