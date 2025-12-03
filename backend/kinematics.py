"""
Kinematic modeling utilities for FitCoachAR
Based on Lecture 5: Human Kinematics

Implements:
- Body-relative coordinate frame transformations
- Forward kinematics for joint angle computation
- Angular feature extraction (active and passive)
"""

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Default landmark dictionary using MediaPipe pose landmarks
DEFAULT_LANDMARK_DICT = {
    "LEFT_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "LEFT_ELBOW": mp_pose.PoseLandmark.LEFT_ELBOW.value,
    "RIGHT_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "LEFT_WRIST": mp_pose.PoseLandmark.LEFT_WRIST.value,
    "RIGHT_WRIST": mp_pose.PoseLandmark.RIGHT_WRIST.value,
    "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP.value,
    "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP.value,
    "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE.value,
    "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE.value,
    "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE.value,
    "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    "NOSE": mp_pose.PoseLandmark.NOSE.value,
}


class BodyRelativeFrame:
    """
    Body-centered coordinate frame for pose normalization.
    
    Following lecture convention:
    - Origin: pelvis/hip center
    - X: Left-right axis (mediolateral)
    - Y: Vertical axis (superior-inferior)
    - Z: Front-back axis (anteroposterior)
    """
    
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self._compute_frame()
    
    def _compute_frame(self):
        """Compute body-centered coordinate frame from landmarks."""
        # Get hip landmarks
        left_hip = np.array([
            self.landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]["x"],
            self.landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]["y"],
            self.landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]["z"]
        ])
        right_hip = np.array([
            self.landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]["x"],
            self.landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]["y"],
            self.landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]["z"]
        ])

        # Origin at pelvis center
        self.origin = (left_hip + right_hip) / 2

        # Right axis (mediolateral): from left hip to right hip
        self.x_axis = right_hip - left_hip
        self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)

        # Up axis (superior-inferior): world Y
        self.y_axis = np.array([0, -1, 0])  # MediaPipe Y is down, we want up

        # Forward axis (anteroposterior): cross product
        self.z_axis = np.cross(self.x_axis, self.y_axis)
        self.z_axis = self.z_axis / np.linalg.norm(self.z_axis)

        # Recompute Y to ensure orthogonal
        self.y_axis = np.cross(self.z_axis, self.x_axis)

    def to_body_frame(self, point):
        """Transform a point from world coordinates to body-relative frame."""
        point_array = np.array(point)
        centered = point_array - self.origin

        # Project onto body axes
        return np.array([
            np.dot(centered, self.x_axis),
            np.dot(centered, self.y_axis),
            np.dot(centered, self.z_axis)
        ])


"""
Kinematic modeling utilities for FitCoachAR
Based on Lecture 5: Human Kinematics & AIFit Methodology

Implements:
- Body-relative coordinate frame transformations
- Spatial Normalization (Scale Invariance)
- Active Feature Extraction (for Rep Counting)
- Passive/Stability Feature Extraction (for Form Checking)
"""

"""
Kinematic modeling utilities for FitCoachAR.
Implements 3D angle calculation and normalized spatial features.
"""


class KinematicFeatureExtractor:
    """
    Extracts geometric features (angles, normalized distances) from pose landmarks.
    """

    @staticmethod
    def _get_coords(landmarks, idx):
        # Handle list-of-dicts format [{"x":...}, ...]
        lm = landmarks[idx]
        return np.array([lm["x"], lm["y"], lm["z"]])

    @staticmethod
    def _compute_distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def compute_joint_angle(a, b, c):
        """Compute 3D angle at joint b."""
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    @staticmethod
    def compute_vertical_angle(p1, p2):
        """Compute angle of vector p1->p2 against vertical Y-axis."""
        vec = p2 - p1
        norm = np.linalg.norm(vec)
        if norm < 1e-6: return 0.0
        # MediaPipe Y is inverted? Assuming standard Y-up for calculation logic
        vertical = np.array([0, 1, 0])
        cosine = np.dot(vec, vertical) / norm
        return np.degrees(np.arccos(np.abs(cosine)))

    @staticmethod
    def extract_metrics(landmarks, landmark_dict, exercise_type):
        """
        Returns { "primary": float, "form": { ... } }
        """
        helper = KinematicFeatureExtractor
        lms = landmarks

        # 1. Scale Factor (Shoulder Width)
        try:
            l_shldr = helper._get_coords(lms, landmark_dict["LEFT_SHOULDER"])
            r_shldr = helper._get_coords(lms, landmark_dict["RIGHT_SHOULDER"])
            shoulder_width = helper._compute_distance(l_shldr, r_shldr)
            scale = 1.0 / shoulder_width if shoulder_width > 0.01 else 1.0
        except (IndexError, KeyError):
            return {"primary": 0.0, "form": {}}

        ex = exercise_type.lower()
        metrics = {"primary": 0.0, "form": {}}

        # --- BICEP CURL ---
        if "curl" in ex:
            # Use Right side by default for now
            p = lambda name: helper._get_coords(lms, landmark_dict[name])

            # Primary: Elbow Angle
            metrics["primary"] = helper.compute_joint_angle(
                p("RIGHT_SHOULDER"), p("RIGHT_ELBOW"), p("RIGHT_WRIST")
            )

            # Form: Elbow Drift (Normalized)
            elbow_dist = helper._compute_distance(p("RIGHT_SHOULDER"), p("RIGHT_ELBOW"))
            metrics["form"]["elbow_stability"] = elbow_dist * scale

            # Form: Torso Swing
            metrics["form"]["torso_swing"] = helper.compute_vertical_angle(
                p("RIGHT_SHOULDER"), p("RIGHT_HIP")
            )

        # --- SQUAT ---
        elif "squat" in ex:
            p = lambda name: helper._get_coords(lms, landmark_dict[name])

            # Primary: Knee Angle
            metrics["primary"] = helper.compute_joint_angle(
                p("LEFT_HIP"), p("LEFT_KNEE"), p("LEFT_ANKLE")
            )

            # Form: Knee Valgus (Knee Width / Ankle Width)
            kw = helper._compute_distance(p("LEFT_KNEE"), p("RIGHT_KNEE"))
            aw = helper._compute_distance(p("LEFT_ANKLE"), p("RIGHT_ANKLE"))
            metrics["form"]["knee_valgus_index"] = kw / aw if aw > 0.01 else 1.0

            # Form: Torso Lean
            metrics["form"]["torso_lean"] = helper.compute_vertical_angle(
                p("LEFT_HIP"), p("LEFT_SHOULDER")
            )

            # Form: Hip Symmetry (Vertical diff)
            metrics["form"]["hip_symmetry"] = abs(p("LEFT_HIP")[1] - p("RIGHT_HIP")[1]) * scale

        return metrics

class AngularFeatureExtractor:
    """
    Extract angular features from pose landmarks.

    Implements active and passive feature sets as described in AIFit paper
    and uses body-relative frames from kinematic modeling lecture.
    """

    @staticmethod
    def compute_joint_angle(a, b, c) -> float:
        """Calculate the planar angle at point b formed by points a-b-c in degrees."""

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return float(angle)

    @staticmethod
    def compute_joint_angle2(a, b, c):
        """
        Compute angle at joint b formed by points a-b-c.

        Args:
            a, b, c: 3D points [x, y, z]

        Returns:
            Angle in degrees
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    @staticmethod
    def compute_angle_with_axis(joint_a, joint_b, axis):
        """
        Compute angle between limb (joint_a to joint_b) and a body axis.

        Used for angles relative to 'Up', 'Right', 'Forward' directions.
        """
        limb = np.array(joint_b) - np.array(joint_a)
        limb_norm = limb / np.linalg.norm(limb)

        cosine_angle = np.dot(limb_norm, axis)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    @staticmethod
    def extract_all_features(landmarks, landmark_dict=None):
        """
        Extract comprehensive set of angular features.

        Returns dictionary with:
        - Active features: elbow, knee, shoulder angles
        - Passive features: spine angle, hip stability
        """
        if landmark_dict is None:
            landmark_dict = DEFAULT_LANDMARK_DICT
        features = {}

        # Helper to get 3D point
        def get_point(idx):
            lm = landmarks[idx]
            return [lm["x"], lm["y"], lm["z"]]

        # ACTIVE FEATURES (high motion energy)

        # Elbow angles (articulation)
        features['left_elbow_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(landmark_dict["LEFT_WRIST"]),
            get_point(landmark_dict["LEFT_ELBOW"]),
            get_point(landmark_dict["LEFT_SHOULDER"])
        )

        features['right_elbow_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(landmark_dict["RIGHT_SHOULDER"]),
            get_point(landmark_dict["RIGHT_ELBOW"]),
            get_point(landmark_dict["RIGHT_WRIST"])
        )

        # Knee angles (articulation)
        features['left_knee_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(landmark_dict["LEFT_HIP"]),
            get_point(landmark_dict["LEFT_KNEE"]),
            get_point(landmark_dict["LEFT_ANKLE"])
        )

        features['right_knee_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(landmark_dict["RIGHT_HIP"]),
            get_point(landmark_dict["RIGHT_KNEE"]),
            get_point(landmark_dict["RIGHT_ANKLE"])
        )

        # Shoulder angles with vertical axis
        body_frame = BodyRelativeFrame(landmarks)

        features['left_shoulder_vertical_angle'] = AngularFeatureExtractor.compute_angle_with_axis(
            get_point(landmark_dict["LEFT_SHOULDER"]),
            get_point(landmark_dict["LEFT_ELBOW"]),
            body_frame.y_axis
        )

        features['right_shoulder_vertical_angle'] = AngularFeatureExtractor.compute_angle_with_axis(
            get_point(landmark_dict["RIGHT_SHOULDER"]),
            get_point(landmark_dict["RIGHT_ELBOW"]),
            body_frame.y_axis
        )

        # PASSIVE FEATURES (low motion energy, should remain constant)

        # Spine angle (should stay straight ~180°)
        features['spine_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(landmark_dict["LEFT_HIP"]),
            get_point(landmark_dict["LEFT_SHOULDER"]),
            get_point(landmark_dict["NOSE"])
        )

        # Hip alignment (pelvis should stay level)
        left_hip_y = landmarks[landmark_dict["LEFT_HIP"]]["y"]
        right_hip_y = landmarks[landmark_dict["RIGHT_HIP"]]["y"]
        features['hip_tilt'] = abs(left_hip_y - right_hip_y) * 100  # Normalized difference

        # Shoulder alignment
        left_shoulder_y = landmarks[landmark_dict["LEFT_SHOULDER"]]["y"]
        right_shoulder_y = landmarks[landmark_dict["RIGHT_SHOULDER"]]["y"]
        features['shoulder_tilt'] = abs(left_shoulder_y - right_shoulder_y) * 100

        return features

    @staticmethod
    def primary_angle_from_landmarks(landmarks, exercise, landmark_dict):
        feature_angles = AngularFeatureExtractor.extract_all_features(landmarks, landmark_dict)
        """
        Extract the primary joint angle for a given exercise from a single frame of landmarks.

        Supported defaults:
        - "bicep_curl": right elbow angle
        - "squat": right knee angle
        - "push_up": right elbow angle
        """
        ex = exercise.lower()
        if ex == "bicep_curl":
            return feature_angles["left_elbow_angle"]
        elif ex == "squat":
            return feature_angles["left_knee_angle"]
        elif ex == "push_up":
            return feature_angles["left_elbow_angle"]
        else:
            return None


class RepetitionSegmenter:
    """
    Online repetition segmentation using state machine.
    
    Based on FitCoachAR proposal Section 4.2 and AIFit paper methodology.
    Detects phase transitions (descent, bottom, ascent) using derivative signs.
    """
    
    def __init__(self, exercise_type='bicep_curls'):
        self.exercise_type = exercise_type
        self.state = 'UP' if exercise_type == 'bicep_curls' else 'UP'
        self.history = []
        self.rep_count = 0
        self.hysteresis = 10  # Degrees of hysteresis to prevent jitter
    
    def update(self, angle, thresholds):
        """
        Update state machine with new angle measurement.
        
        Args:
            angle: Current joint angle
            thresholds: dict with 'up' and 'down' angle thresholds
        
        Returns:
            True if a new repetition was completed
        """
        new_rep = False
        
        if self.state == 'UP':
            if angle < (thresholds['down'] + self.hysteresis):
                self.state = 'DOWN'
        elif self.state == 'DOWN':
            if angle > (thresholds['up'] - self.hysteresis):
                self.state = 'UP'
                self.rep_count += 1
                new_rep = True
        
        self.history.append(angle)
        if len(self.history) > 100:
            self.history.pop(0)
        
        return new_rep
