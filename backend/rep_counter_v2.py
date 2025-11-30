"""
Normalized rep counter driven by calibration parameters.
Focuses strictly on cycle detection (Counting).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from calibration_v2 import CalibrationParams

def _sign(x: float, eps: float = 1e-3) -> int:
    if x > eps: return 1
    if x < -eps: return -1
    return 0

@dataclass
class RepCounterState:
    rep_count: int = 0
    phase: str = "BOTTOM"  # BOTTOM, UP_PHASE, TOP, DOWN_PHASE
    reached_top: bool = False
    t_start: Optional[float] = None
    last_metric: Optional[float] = None
    progress: float = 0.0 # 0.0 to 1.0

class NormalizedRepCounter:
    """
    Online counter that uses calibrated ROM and timing for robustness.
    Counts ANY rep that completes the cycle, regardless of form.
    """

    def __init__(self, params: CalibrationParams):
        self.params = params
        self.state = RepCounterState()

    def reset(self):
        self.state = RepCounterState()

    def _progress(self, val: float) -> float:
        # Normalized progress 0.0 (Bottom) to 1.0 (Top)
        denom = max(1e-6, self.params.theta_high - self.params.theta_low)
        return (val - self.params.theta_low) / denom

    def update(self, metric: float, t: float) -> RepCounterState:
        """
        Update counter with the primary metric sample at time t.
        """
        p = self._progress(metric)
        self.state.progress = p

        last = self.state.last_metric if self.state.last_metric is not None else metric
        direction = _sign(metric - last)

        # Timeouts: If stuck in a phase too long, reset to start
        if self.state.t_start is not None:
            elapsed = t - self.state.t_start
            if elapsed > self.params.t_max and self.state.phase != "BOTTOM":
                self.state.phase = "BOTTOM"
                self.state.t_start = None
                self.state.reached_top = False

        phase = self.state.phase

        # State Machine (Trough -> Peak -> Trough)
        if phase == "BOTTOM":
            # Moving UP (towards theta_high)
            if p > 0.15 and direction > 0:
                self.state.phase = "UP_PHASE"
                self.state.t_start = t
                self.state.reached_top = False

        elif phase == "UP_PHASE":
            # Reached Top?
            if p >= 0.85:
                self.state.phase = "TOP"
                self.state.reached_top = True
            # Failed rep (went back down too early)
            elif p <= 0.1 and direction < 0:
                self.state.phase = "BOTTOM"
                self.state.t_start = None

        elif phase == "TOP":
            # Moving DOWN
            if p < 0.85 and direction < 0:
                self.state.phase = "DOWN_PHASE"

        elif phase == "DOWN_PHASE":
            # Reached Bottom (Rep Complete)
            if p <= 0.15:
                if self.state.reached_top and self.state.t_start is not None:
                    duration = t - self.state.t_start
                    # Minimal sanity check on duration
                    if duration >= self.params.t_min:
                        self.state.rep_count += 1

                self.state.phase = "BOTTOM"
                self.state.t_start = None
                self.state.reached_top = False

        self.state.last_metric = metric
        return self.state