"""
Calibration utilities that derive exercise-specific ROM, counting thresholds,
and user capabilities (Adaptive, Tempo, Symmetry) for feedback.
"""

from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import uuid
import json
import numpy as np

# Ensure kinematics.py is also updated as per previous instructions
from kinematics import KinematicFeatureExtractor

DEFAULT_STORE_PATH = Path(__file__).resolve().parent / "calibrations.json"


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


feature_extractor = KinematicFeatureExtractor()


def extract_metric_series(
        landmark_seq: np.ndarray, landmark_dict, exercise: str
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Convert landmark time series into:
    1. A smoothed primary metric series (for counting).
    2. A dictionary of form metric series (for stability/constraint analysis).
    """
    primary_series: List[float] = []
    form_series: Dict[str, List[float]] = {}

    for frame in landmark_seq:
        metrics = feature_extractor.extract_metrics(frame, landmark_dict, exercise)

        # 1. Primary Signal
        primary_series.append(metrics["primary"])

        # 2. Form Signals
        for key, val in metrics["form"].items():
            if key not in form_series:
                form_series[key] = []
            form_series[key].append(val)

    if not primary_series:
        return np.array([], dtype=float), {}

    # Smooth the primary series
    arr = np.array(primary_series, dtype=float)
    w = 5
    if arr.size >= w:
        kernel = np.ones(w, dtype=float) / float(w)
        padded = np.pad(arr, (w - 1, 0), mode="edge")
        arr = np.convolve(padded, kernel, mode="valid")

    return arr, form_series


@dataclass
class RepSegment:
    start_idx: int
    peak_idx: int
    end_idx: int
    bottom_val: float
    top_val: float
    duration_seconds: float
    
    # NEW: Store form metrics exactly at the bottom point (start_idx)
    # This allows us to see what the user's form looked like at the hardest part of the rep.
    form_at_bottom: Dict[str, float] = field(default_factory=dict)

    # Phase Timings
    phase_1_duration: float = 0.0  # Descent / Eccentric
    phase_2_duration: float = 0.0  # Ascent / Concentric


@dataclass
class CalibrationParams:
    # --- Counting Parameters ---
    theta_low: float
    theta_high: float
    t_med: float
    t_min: float
    t_max: float

    # --- Adaptive Feedback Parameters ---
    capabilities: Dict[str, float] = field(default_factory=dict)
    form_constraints: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tempo_profile: Dict[str, float] = field(default_factory=dict)
    symmetry_profile: Dict[str, float] = field(default_factory=dict)
    
    # NEW: Stores the Median values of form metrics at the calibrated Bottom-ROM.
    # This acts as the "Gold Standard" or "Norm" for form validation during live counting.
    normative_constraints: Dict[str, float] = field(default_factory=dict)

    # Legacy fields
    theta_low_median: Optional[float] = None
    theta_high_median: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "theta_low": self.theta_low,
            "theta_high": self.theta_high,
            "t_med": self.t_med,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "capabilities": self.capabilities,
            "form_constraints": self.form_constraints,
            "tempo_profile": self.tempo_profile,
            "symmetry_profile": self.symmetry_profile,
            "normative_constraints": self.normative_constraints,
        }
        if self.theta_low_median is not None:
            data["theta_low_median"] = self.theta_low_median
        if self.theta_high_median is not None:
            data["theta_high_median"] = self.theta_high_median
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CalibrationParams":
        return CalibrationParams(
            theta_low=float(data["theta_low"]),
            theta_high=float(data["theta_high"]),
            t_med=float(data["t_med"]),
            t_min=float(data["t_min"]),
            t_max=float(data["t_max"]),
            capabilities=data.get("capabilities", {}),
            form_constraints=data.get("form_constraints", {}),
            tempo_profile=data.get("tempo_profile", {}),
            symmetry_profile=data.get("symmetry_profile", {}),
            normative_constraints=data.get("normative_constraints", {}),
            theta_low_median=float(data.get("theta_low_median", 0)) if "theta_low_median" in data else None,
            theta_high_median=float(data.get("theta_high_median", 0)) if "theta_high_median" in data else None,
        )


@dataclass
class CalibrationResult:
    params: CalibrationParams
    rep_segments: List[RepSegment]
    extrema: List[Tuple[int, float, str]]
    smoothed_signal: np.ndarray
    # NEW: Store raw data points for visualization or debugging
    raw_form_series: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class CalibrationRecord:
    id: str
    params: CalibrationParams
    timestamp: str

    def to_dict(self):
        params_dict = self.params.to_dict()
        params_dict["id"] = self.id
        params_dict["timestamp"] = self.timestamp
        return params_dict


@dataclass
class ExerciseEntry:
    records: List[CalibrationRecord] = field(default_factory=list)
    active: List[str] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)


class CalibrationStore:
    def __init__(self, path: Path = DEFAULT_STORE_PATH):
        self.path = path
        self.exercises: Dict[str, ExerciseEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return

        exercises_raw = raw.get("exercises", {})
        for name, entry in exercises_raw.items():
            records_raw = entry.get("records", [])
            active_raw = entry.get("active", [])
            params_raw = entry.get("params", {})

            records: List[CalibrationRecord] = []
            for r in records_raw:
                rec_id = r.get("id") or uuid.uuid4().hex
                rec_timestamp = r.get("timestamp") or _timestamp()
                params = CalibrationParams.from_dict(r)
                records.append(CalibrationRecord(id=rec_id, params=params, timestamp=rec_timestamp))

            active: List[str] = []
            if isinstance(active_raw, list) and active_raw:
                first = active_raw[0]
                if isinstance(first, int) and 0 <= first < len(records):
                    active = [records[first].id]
                elif isinstance(first, str):
                    active = list(active_raw)

            self.exercises[name] = ExerciseEntry(
                records=records,
                active=active,
                params=dict(params_raw) if isinstance(params_raw, dict) else {}
            )

    def _save(self) -> None:
        data: Dict[str, Dict] = {"exercises": {}}
        for name, entry in self.exercises.items():
            records_serialized = [rec.to_dict() for rec in entry.records]
            data["exercises"][name] = {
                "records": records_serialized,
                "active": entry.active,
                "params": entry.params,
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    # ---- API ----

    def add_calibration_record(self, exercise: str, params: CalibrationParams, make_active: bool = True) -> str:
        entry = self.exercises.setdefault(exercise, ExerciseEntry())
        rec_id = uuid.uuid4().hex
        a = CalibrationRecord(id=rec_id, params=params, timestamp=_timestamp())
        entry.records.append(a)
        if make_active:
            entry.active = [rec_id]
        self._save()
        return rec_id

    def set_active_record(self, exercise: str, record_id: str) -> None:
        entry = self.exercises.setdefault(exercise, ExerciseEntry())
        if any(r.id == record_id for r in entry.records):
            entry.active = [record_id]
            self._save()

    def get_active_params(self, exercise: str) -> Optional[CalibrationParams]:
        entry = self.exercises.get(exercise)
        if not entry or not entry.active:
            return None
        active_id = entry.active[0]
        for rec in entry.records:
            if rec.id == active_id:
                return rec.params
        return None

    def get_all_records(self, exercise: str) -> List[CalibrationRecord]:
        entry = self.exercises.get(exercise)
        return list(entry.records) if entry else []

    def delete_record(self, exercise: str, record_id: str) -> bool:
        entry = self.exercises.get(exercise)
        if not entry:
            return False

        initial_len = len(entry.records)
        entry.records = [r for r in entry.records if r.id != record_id]
        if len(entry.records) == initial_len:
            return False

        if entry.active and entry.active[0] == record_id:
            entry.active = [entry.records[0].id] if entry.records else []

        if not entry.records:
            self.exercises.pop(exercise, None)

        self._save()
        return True

    def set_exercise_params(self, exercise: str, critic: float, deviation: float) -> None:
        entry = self.exercises.setdefault(exercise, ExerciseEntry())
        entry.params = {"critic": float(critic), "deviation": float(deviation)}
        self._save()

    def get_exercise_params(self, exercise: str) -> Dict[str, float]:
        entry = self.exercises.get(exercise)
        return dict(entry.params) if entry else {}


def _find_extrema(signal: np.ndarray) -> List[Tuple[int, float, str]]:
    extrema = []
    for i in range(1, signal.size - 1):
        prev, curr, nxt = signal[i - 1], signal[i], signal[i + 1]
        if curr > prev and curr > nxt:
            extrema.append((i, float(curr), "peak"))
        elif curr < prev and curr < nxt:
            extrema.append((i, float(curr), "trough"))
    return extrema


def _segment_reps(extrema: List[Tuple[int, float, str]], fps: float) -> List[RepSegment]:
    reps = []
    if len(extrema) < 3:
        return reps

    # State Machine: Find Trough -> Peak -> Trough (Standard Rep)
    for i in range(len(extrema) - 2):
        a, b, c = extrema[i], extrema[i + 1], extrema[i + 2]
        if (a[2], b[2], c[2]) == ("trough", "peak", "trough"):
            start_idx, start_val = a[0], a[1]
            peak_idx, peak_val = b[0], b[1]
            end_idx, end_val = c[0], c[1]

            reps.append(RepSegment(
                start_idx=start_idx,
                peak_idx=peak_idx,
                end_idx=end_idx,
                bottom_val=min(start_val, end_val),
                top_val=peak_val,
                duration_seconds=max(0.0, (end_idx - start_idx) / fps),
                phase_1_duration=max(0.0, (peak_idx - start_idx) / fps),
                phase_2_duration=max(0.0, (end_idx - peak_idx) / fps)
                # Note: form_at_bottom is filled later in calibrate_from_landmarks 
                # because we don't have access to the raw form series here.
            ))

    return reps


def calibrate_from_landmarks(
        landmark_seq: np.ndarray,
        landmark_dict,
        exercise: str,
        smoothing_window: int = 5,
        fps: float = 30.0,
        theta_low_factor: float = 0.3,
        theta_high_factor: float = 3.0,
) -> CalibrationResult:
    """
    Derive calibration parameters: Counting Thresholds, User Capabilities, Form Constraints.
    """
    # 1. Extract Signals
    primary_signal, form_signals = extract_metric_series(landmark_seq, landmark_dict, exercise)
    if primary_signal.size == 0:
        raise ValueError("No valid metrics extracted for calibration.")

    # 2. Analyze Primary Signal (for Rep Counting)
    extrema = _find_extrema(primary_signal)

    reps = _segment_reps(extrema, fps=fps)

    if not reps:
        # Fallback if no clean reps found
        theta_low = float(np.min(primary_signal))
        theta_high = float(np.max(primary_signal))
        t_med = 2.0
        capabilities = {"rom_min": theta_low, "rom_max": theta_high}
        tempo_profile = {"concentric_eccentric_ratio": 1.0}
    else:
        bottoms = [r.bottom_val for r in reps]
        tops = [r.top_val for r in reps]
        durations = [r.duration_seconds for r in reps]

        theta_low = float(np.mean(bottoms))
        theta_high = float(np.mean(tops))
        t_med = float(np.median(durations))

        capabilities = {
            "rom_min": float(np.min(bottoms)),
            "rom_max": float(np.max(tops))
        }

        # Tempo Profile
        ratios = []
        for r in reps:
            if r.phase_2_duration > 0.1:
                ratios.append(r.phase_1_duration / r.phase_2_duration)
        avg_ratio = float(np.mean(ratios)) if ratios else 1.0
        tempo_profile = {"concentric_eccentric_ratio": avg_ratio}

        # NEW: Fill RepSegment.form_at_bottom and collect normative constraints
        # We look up the form metrics at the specific index where the rep bottomed out (start_idx).
        form_at_bottom_list = []
        for rep in reps:
            form_dict = {}
            for metric_name, series in form_signals.items():
                if rep.start_idx < len(series):
                    form_dict[metric_name] = series[rep.start_idx]
            rep.form_at_bottom = form_dict
            form_at_bottom_list.append(form_dict)

        # NEW: Calculate Normative Constraints (Median of form metrics at lowest point)
        normative_constraints = {}
        if form_at_bottom_list:
            for metric_name in form_signals.keys():
                values = [d.get(metric_name) for d in form_at_bottom_list if metric_name in d]
                if values:
                    normative_constraints[metric_name] = float(np.median(values))

    # 3. Analyze Form Signals (Statistical overview of the whole session)
    form_constraints = {}
    symmetry_profile = {}

    for metric_name, values in form_signals.items():
        arr = np.array(values)
        if arr.size == 0:
            continue

        if "symmetry" in metric_name:
            symmetry_profile[metric_name + "_max"] = float(np.max(arr))

        form_constraints[metric_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }

    params = CalibrationParams(
        theta_low=theta_low,
        theta_high=theta_high,
        t_med=t_med,
        t_min=theta_low_factor * t_med,
        t_max=theta_high_factor * t_med,
        capabilities=capabilities,
        form_constraints=form_constraints,
        tempo_profile=tempo_profile,
        symmetry_profile=symmetry_profile,
        normative_constraints=normative_constraints if reps else {}, # NEW: Pass the calculated norms
        theta_low_median=float(np.median([r.bottom_val for r in reps])) if reps else theta_low,
        theta_high_median=float(np.median([r.top_val for r in reps])) if reps else theta_high,
    )
    return CalibrationResult(
        params=params, 
        rep_segments=reps, 
        extrema=extrema, 
        smoothed_signal=primary_signal,
        raw_form_series=form_signals # Store raw data for debugging
    )