"""
Normalized rep counter driven by calibration parameters.
Focuses on robust cycle detection and normative form validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict
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
    progress: float = 0.0  # 0.0 to 1.0
    
    # Stores the form metrics at the exact moment a rep is completed
    form_at_rep_complete: Dict[str, float] = field(default_factory=dict)

class NormalizedRepCounter:
    """
    Online counter that uses calibrated ROM, timing, and normative constraints.
    It counts reps only if they meet the ROM, Timing, and minimum Form Quality standards.
    """

    # Stricter thresholds derived from calibration
    BOTTOM_THRESHOLD = 0.10 
    TOP_THRESHOLD = 0.90    

    def __init__(self, params: CalibrationParams):
        self.params = params
        self.state = RepCounterState()
        
        # FIX: Absoluter Mindest-ROM Check. 
        # Verhindert Zählen, wenn das Signal nur Rauschen ist (z.B. 2 Grad Wackeln).
        # Wir nehmen an, dass 'wildes Fuchteln' oft kleine Amplituden hat oder chaotisch ist.
        self.min_absolute_rom_span = 15.0 # Mindestens 15 Einheiten (Grad/cm) Differenz nötig

    def reset(self):
        self.state = RepCounterState()

    def _progress(self, val: float) -> float:
        # Normalized progress 0.0 (Bottom) to 1.0 (Top)
        # FIX: Schutz gegen Division durch Null bei schlechter Kalibrierung
        rom_span = self.params.theta_high - self.params.theta_low
        denom = max(1e-6, rom_span)
        return (val - self.params.theta_low) / denom

    def _check_form_quality(self, current_metrics: Dict[str, float]) -> bool:
        """
        Internal check: Is the form quality sufficient to maintain the rep state?
        """
        # Wenn wir keine Normen haben (erste Nutzung), sind wir nachsichtig.
        if not self.params.normative_constraints:
            return True

        # FIX: Toleranz für wildes Fuchteln.
        # Wenn eine Metrik (z.B. Ellbogenstabilität) extrem abweicht, brich ab.
        for name, norm_val in self.params.normative_constraints.items():
            current_val = current_metrics.get(name)
            if current_val is not None:
                # Wir erlauben großzügige Abweichung (z.B. 30 Einheiten), 
                # aber "wildes Fuchteln" überschreitet das meistens.
                TOLERANCE = 15.0 
                if abs(current_val - norm_val) > TOLERANCE:
                    # Form is chaotic -> Abort rep
                    return False
        
        return True

    def update(self, primary_metric: float, t: float, current_form_metrics: Dict[str, float]) -> RepCounterState:
        """
        Update counter with the primary metric sample at time t.
        Requires current_form_metrics for active validation.
        """
        # FIX: Sanity Check - Ist der kalibrierte ROM überhaupt groß genug?
        # Wenn User bei Kalibrierung nur 2 Grad bewegt hat, zählt jedes Rauschen.
        rom_span = abs(self.params.theta_high - self.params.theta_low)
        if rom_span < self.min_absolute_rom_span:
            # ROM zu klein, wir zählen gar nichts.
            return self.state

        p = self._progress(primary_metric)
        self.state.progress = p

        last = self.state.last_metric if self.state.last_metric is not None else primary_metric
        direction = _sign(primary_metric - last)

        # Timeouts: If stuck in a phase too long, reset to start
        if self.state.t_start is not None:
            elapsed = t - self.state.t_start
            if elapsed > self.params.t_max and self.state.phase != "BOTTOM":
                self.state.phase = "BOTTOM"
                self.state.t_start = None
                self.state.reached_top = False

        # FIX: Continuous Stability Guard
        # Wenn wir NICHT im BOTTOM sind (also mitten in der Rep), prüfen wir, ob die Form explodiert.
        if self.state.phase != "BOTTOM":
            if not self._check_form_quality(current_form_metrics):
                # ABBRUCH! Form ist zu schlecht (wildes Fuchteln erkannt)
                self.state.phase = "BOTTOM"
                self.state.t_start = None
                self.state.reached_top = False
                self.state.last_metric = primary_metric
                return self.state

        phase = self.state.phase
        
        # State Machine (Trough -> Peak -> Trough)
        if phase == "BOTTOM":
            if p > self.BOTTOM_THRESHOLD and direction > 0:
                self.state.phase = "UP_PHASE"
                self.state.t_start = t
                self.state.reached_top = False

        elif phase == "UP_PHASE":
            if p >= self.TOP_THRESHOLD:
                self.state.phase = "TOP"
                self.state.reached_top = True
            elif p <= self.BOTTOM_THRESHOLD / 2 and direction < 0:
                self.state.phase = "BOTTOM"
                self.state.t_start = None

        elif phase == "TOP":
            if p < self.TOP_THRESHOLD and direction < 0:
                self.state.phase = "DOWN_PHASE"

        elif phase == "DOWN_PHASE":
            if p <= self.BOTTOM_THRESHOLD:
                if self.state.reached_top and self.state.t_start is not None:
                    duration = t - self.state.t_start
                    
                    # Final Checks
                    is_valid_tempo = duration >= self.params.t_min
                    # Form Check am Ende ist jetzt redundant durch Continuous Check, aber sicher ist sicher
                    is_valid_form = self._check_form_quality(current_form_metrics)

                    if is_valid_tempo and is_valid_form:
                        self.state.rep_count += 1
                        self.state.form_at_rep_complete = current_form_metrics.copy()

                self.state.phase = "BOTTOM"
                self.state.t_start = None
                self.state.reached_top = False

        self.state.last_metric = primary_metric
        return self.state