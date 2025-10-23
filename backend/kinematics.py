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
            self.landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            self.landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            self.landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z
        ])
        right_hip = np.array([
            self.landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            self.landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            self.landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z
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


class AngularFeatureExtractor:
    """
    Extract angular features from pose landmarks.
    
    Implements active and passive feature sets as described in AIFit paper
    and uses body-relative frames from kinematic modeling lecture.
    """
    
    @staticmethod
    def compute_joint_angle(a, b, c):
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
    def extract_all_features(landmarks):
        """
        Extract comprehensive set of angular features.
        
        Returns dictionary with:
        - Active features: elbow, knee, shoulder angles
        - Passive features: spine angle, hip stability
        """
        features = {}
        
        # Helper to get 3D point
        def get_point(idx):
            lm = landmarks[idx]
            return [lm.x, lm.y, lm.z]
        
        # ACTIVE FEATURES (high motion energy)
        
        # Elbow angles (articulation)
        features['left_elbow_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            get_point(mp_pose.PoseLandmark.LEFT_ELBOW.value),
            get_point(mp_pose.PoseLandmark.LEFT_WRIST.value)
        )
        
        features['right_elbow_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            get_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            get_point(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        )
        
        # Knee angles (articulation)
        features['left_knee_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(mp_pose.PoseLandmark.LEFT_HIP.value),
            get_point(mp_pose.PoseLandmark.LEFT_KNEE.value),
            get_point(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        )
        
        features['right_knee_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(mp_pose.PoseLandmark.RIGHT_HIP.value),
            get_point(mp_pose.PoseLandmark.RIGHT_KNEE.value),
            get_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        )
        
        # Shoulder angles with vertical axis
        body_frame = BodyRelativeFrame(landmarks)
        
        features['left_shoulder_vertical_angle'] = AngularFeatureExtractor.compute_angle_with_axis(
            get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            get_point(mp_pose.PoseLandmark.LEFT_ELBOW.value),
            body_frame.y_axis
        )
        
        features['right_shoulder_vertical_angle'] = AngularFeatureExtractor.compute_angle_with_axis(
            get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            get_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            body_frame.y_axis
        )
        
        # PASSIVE FEATURES (low motion energy, should remain constant)
        
        # Spine angle (should stay straight ~180°)
        features['spine_angle'] = AngularFeatureExtractor.compute_joint_angle(
            get_point(mp_pose.PoseLandmark.LEFT_HIP.value),
            get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            get_point(mp_pose.PoseLandmark.NOSE.value)
        )
        
        # Hip alignment (pelvis should stay level)
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        features['hip_tilt'] = abs(left_hip_y - right_hip_y) * 100  # Normalized difference
        
        # Shoulder alignment
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        features['shoulder_tilt'] = abs(left_shoulder_y - right_shoulder_y) * 100
        
        return features


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
