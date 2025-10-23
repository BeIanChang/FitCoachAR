import numpy as np
from scipy.signal import savgol_filter
from pykalman import KalmanFilter


class KalmanLandmarkSmoother:
    """Applies a Kalman filter to smooth landmark trajectories."""

    def __init__(self):
        self.kalman_filters = {}

    def _get_filter(self, landmark_index):
        if landmark_index not in self.kalman_filters:
            # State: [x, y, dx, dy] (position and velocity)
            # We observe only position (x, y)
            transition_matrix = [[1, 0, 1, 0],  # x = x + dx
                               [0, 1, 0, 1],  # y = y + dy
                               [0, 0, 1, 0],  # dx = dx
                               [0, 0, 0, 1]]  # dy = dy

            observation_matrix = [[1, 0, 0, 0],
                                  [0, 1, 0, 0]]

            self.kalman_filters[landmark_index] = KalmanFilter(
                transition_matrices=transition_matrix,
                observation_matrices=observation_matrix,
                initial_state_mean=[0, 0, 0, 0],
                initial_state_covariance=np.eye(4) * 0.1,
                observation_covariance=np.eye(2) * 0.1,
                transition_covariance=np.eye(4) * 0.01
            )
        return self.kalman_filters[landmark_index]

    def smooth(self, landmarks):
        """Smooths a list of landmarks from a single frame."""
        if not landmarks:
            return []

        smoothed_landmarks = []
        for i, lm in enumerate(landmarks):
            kf = self._get_filter(i)

            # Get the current state mean and covariance
            if i in self.kalman_filters and hasattr(kf, 'kf_mean'):
                current_mean = kf.kf_mean
                current_cov = kf.kf_cov
            else:
                # Initialize at the first observed position
                current_mean = [lm.x, lm.y, 0, 0]
                current_cov = np.eye(4) * 0.1

            # Apply the filter
            new_mean, new_cov = kf.filter_update(
                current_mean,
                current_cov,
                observation=np.array([lm.x, lm.y])
            )

            # Store the updated state for the next iteration
            kf.kf_mean = new_mean
            kf.kf_cov = new_cov

            # Create a new landmark object with the smoothed position
            class SmoothedLandmark:
                def __init__(self, x, y, z, visibility):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility

            smoothed_landmarks.append(SmoothedLandmark(new_mean[0], new_mean[1], lm.z, lm.visibility))

        return smoothed_landmarks



class LandmarkSmoother:
    """Uses Savitzky-Golay filtering to smooth landmark trajectories."""
    def __init__(self, window_length=5, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder
        self.history = {}

    def smooth(self, landmarks):
        """Smooths a list of landmarks from a single frame."""
        if not landmarks:
            return []

        # Add current landmarks to history
        for i, lm in enumerate(landmarks):
            if i not in self.history:
                self.history[i] = []
            self.history[i].append([lm.x, lm.y, lm.z, lm.visibility])

        # Ensure history is not longer than window length
        if len(next(iter(self.history.values()))) > self.window_length:
            for i in self.history:
                self.history[i].pop(0)
        
        # If we don't have enough history, return the raw landmarks
        if len(next(iter(self.history.values()))) < self.window_length:
            return landmarks

        smoothed_landmarks = []
        for i, history_queue in self.history.items():
            history_array = np.array(history_queue)
            
            # Smooth x, y, z coordinates
            smoothed_x = savgol_filter(history_array[:, 0], self.window_length, self.polyorder)[-1]
            smoothed_y = savgol_filter(history_array[:, 1], self.window_length, self.polyorder)[-1]
            smoothed_z = savgol_filter(history_array[:, 2], self.window_length, self.polyorder)[-1]
            
            # Use the most recent visibility score
            current_visibility = history_array[-1, 3]

            # Create a new landmark object (mimicking MediaPipe's structure)
            class SmoothedLandmark:
                def __init__(self, x, y, z, visibility):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility
            
            smoothed_landmarks.append(SmoothedLandmark(smoothed_x, smoothed_y, smoothed_z, current_visibility))

        return smoothed_landmarks
