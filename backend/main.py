import asyncio
# import uvloop

# Install and use uvloop as the default event loop
# uvloop.install()

import cv2
import base64
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import logging
import math
import time
from collections import deque
from filters import KalmanLandmarkSmoother
from kinematics import AngularFeatureExtractor, RepetitionSegmenter, BodyRelativeFrame
from llm_feedback import LLMFeedbackGenerator

# Helper function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize MediaPipe Models
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
landmark_smoother = KalmanLandmarkSmoother()

# Initialize FitCoachAR components
llm_feedback = LLMFeedbackGenerator(use_api=False)  # Use template-based for low latency

# Bicep curl counter variables
curl_counter = 0 
curl_state = "DOWN"
feedback = ""
feedback_landmarks = []
elbow_baseline_x = 0

# Workout statistics
total_reps = 0
mistake_counter = {}

# Squat counter variables
squat_counter = 0
squat_state = "UP"

# Personalized calibration variables
arm_extended_angle = 160
arm_contracted_angle = 30
last_right_elbow_angle = 0
squat_up_angle = 160
squat_down_angle = 50
last_right_knee_angle = 0
last_plausible_left_elbow = 0
last_plausible_right_elbow = 0
last_plausible_left_knee = 0
last_plausible_right_knee = 0
selected_exercise = None

# Real-time performance tracking
frame_times = deque(maxlen=60)
last_processed_data = {}
landmarks = None  # Initialize landmarks

# Body normalization variables
user_profile = {
    "upper_arm_length": None,
    "thigh_length": None
}

@app.get("/")
def read_root():
    return {"message": "Welcome to FitCoachAR - Real-Time Adaptive Exercise Coaching API"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("WebSocket connection attempt received.")
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    logger.info("Client connected")

    # Declare all global variables at the top of the function scope
    global curl_counter, curl_state, feedback, elbow_baseline_x, arm_extended_angle, arm_contracted_angle, last_right_elbow_angle
    global squat_counter, squat_state, squat_up_angle, squat_down_angle, last_right_knee_angle, selected_exercise, user_profile
    global last_plausible_left_elbow, last_plausible_right_elbow, last_plausible_left_knee, last_plausible_right_knee, frame_times, last_processed_data

    try:
        while True:
            data = await websocket.receive_text()

            # Handle commands from the client
            if data.startswith('{"command"'):
                import json
                command_data = json.loads(data)
                command = command_data.get("command")
                exercise = command_data.get("exercise")
                if exercise:
                    selected_exercise = exercise

                if command == "calibrate_down":
                    arm_extended_angle = last_right_elbow_angle
                    # Calculate upper arm length
                    if landmarks:
                        shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
                        elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
                        user_profile['upper_arm_length'] = np.linalg.norm(shoulder - elbow)
                    logger.info(f"Calibrated DOWN angle to: {arm_extended_angle} and Upper Arm Length: {user_profile.get('upper_arm_length', 0):.4f}")
                elif command == "calibrate_up":
                    arm_contracted_angle = last_right_elbow_angle
                    logger.info(f"Calibrated UP angle to: {arm_contracted_angle}")
                elif command == "calibrate_squat_down":
                    squat_down_angle = last_right_knee_angle
                    logger.info(f"Calibrated SQUAT DOWN angle to: {squat_down_angle}")
                elif command == "calibrate_squat_up":
                    squat_up_angle = last_right_knee_angle
                    # Calculate thigh length
                    if landmarks:
                        hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
                        knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
                        user_profile['thigh_length'] = np.linalg.norm(hip - knee)
                    logger.info(f"Calibrated SQUAT UP angle to: {squat_up_angle} and Thigh Length: {user_profile.get('thigh_length', 0):.4f}")
                elif command == "reset":
                    # Send final stats before resetting
                    await websocket.send_json({"summary": {"total_reps": total_reps, "mistakes": mistake_counter}})
                    
                    # Reset all state variables
                    curl_counter = 0
                    curl_state = "DOWN"
                    squat_counter = 0
                    squat_state = "UP"
                    feedback = ""
                    total_reps = 0
                    mistake_counter = {}
                    logger.info("State has been reset")
                continue # Skip frame processing for command messages

            try:
                start_time = time.time()

                # Adaptive Frame Budgeting
                if frame_times:
                    avg_fps = len(frame_times) / sum(frame_times)
                    if avg_fps < 20: # If FPS is low, skip processing this frame
                        if last_processed_data:
                            await websocket.send_json(last_processed_data)
                        continue

                # Ensure the data is in the expected format
                if not data.startswith("data:image/jpeg;base64,"):
                    logger.warning("Received malformed data packet")
                    continue

                # Decode the Base64 string
                header, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                
                # Convert to a NumPy array and then to an image
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Check if image was decoded successfully
                if img is None:
                    logger.warning("Failed to decode image")
                    continue

                # Process the image with MediaPipe Pose
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = holistic.process(img_rgb)

                # Check if pose landmarks were detected
                if results.pose_landmarks:
                    # Smooth landmarks
                    landmarks = landmark_smoother.smooth(results.pose_landmarks.landmark)
                    if not landmarks:
                        continue # Not enough history to smooth yet
                    # Extract coordinates for angle calculation
                    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angles with plausibility checks
                    left_elbow_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
                    if 0 < left_elbow_angle < 180:
                        last_plausible_left_elbow = left_elbow_angle
                    else:
                        left_elbow_angle = last_plausible_left_elbow

                    right_elbow_angle = calculate_angle(shoulder_right, elbow_right, wrist_right)
                    if 0 < right_elbow_angle < 180:
                        last_plausible_right_elbow = right_elbow_angle
                    else:
                        right_elbow_angle = last_plausible_right_elbow

                    # Squat angles with plausibility checks
                    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    left_knee_angle = calculate_angle(hip_left, knee_left, ankle_left)
                    if 45 < left_knee_angle < 181:
                        last_plausible_left_knee = left_knee_angle
                    else:
                        left_knee_angle = last_plausible_left_knee

                    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_knee_angle = calculate_angle(hip_right, knee_right, ankle_right)
                    if 45 < right_knee_angle < 181:
                        last_plausible_right_knee = right_knee_angle
                    else:
                        right_knee_angle = last_plausible_right_knee

                    # Landmark visibility check
                    required_landmarks_visible = True
                    for lm in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]:
                        if landmarks[lm.value].visibility < 0.7:
                            required_landmarks_visible = False
                            break
                    
                    # Check for major occlusions first
                    required_landmarks = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
                    if selected_exercise == 'bicep_curls':
                        required_landmarks.extend([mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST])
                    elif selected_exercise == 'squats':
                        required_landmarks.extend([mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE])

                    is_occluded = any(landmarks[lm.value].visibility < 0.7 for lm in required_landmarks)

                    if is_occluded:
                        feedback = "Adjust camera to show full body"
                        feedback_landmarks = []
                    else:
                        # Reset feedback landmarks for this frame
                        feedback_landmarks = []
                        # Bicep curl counter logic
                        right_elbow_x = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x
                        last_right_elbow_angle = right_elbow_angle # Store the latest angle
                        last_right_knee_angle = right_knee_angle

                        if curl_state == 'DOWN':
                            elbow_baseline_x = right_elbow_x # Set baseline when arm is down
                            if right_elbow_angle < (arm_contracted_angle + 20):
                                curl_state = 'UP'
                            feedback = ""

                        elif curl_state == 'UP':
                            if right_elbow_angle > (arm_extended_angle - 20):
                                curl_state = 'DOWN'
                                curl_counter += 1
                                total_reps += 1
                                feedback = ""
                                print(f"CURL COUNT: {curl_counter}")
                            else:
                                # Form feedback logic during the curl
                                # Dynamic threshold for elbow stability based on user's arm length
                                stability_threshold = user_profile['upper_arm_length'] * 0.15 if user_profile.get('upper_arm_length') else 0.05
                                if abs(right_elbow_x - elbow_baseline_x) > stability_threshold:
                                    feedback = "Keep elbow stable!"
                                    mistake_counter['elbow_stability'] = mistake_counter.get('elbow_stability', 0) + 1
                                    feedback_landmarks = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]
                                elif right_elbow_angle > (arm_contracted_angle + 20):
                                    feedback = "Curl higher!"
                                    mistake_counter['curl_depth'] = mistake_counter.get('curl_depth', 0) + 1
                                    feedback_landmarks = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]
                                else:
                                    feedback = "Great curl!"

                        # Squat counter logic
                        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                        right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

                        if squat_state == 'UP':
                            if right_knee_angle < (squat_down_angle + 20):
                                squat_state = 'DOWN'
                            # We don't want curl feedback during squats
                            if feedback not in ["Keep elbow stable!", "Curl higher!", "Great curl!"]:
                                feedback = ""
                        
                        elif squat_state == 'DOWN':
                            if right_knee_angle > (squat_up_angle - 20):
                                squat_state = 'UP'
                                squat_counter += 1
                                total_reps += 1
                                feedback = ""
                                print(f"SQUAT COUNT: {squat_counter}")
                            else:
                                # Depth feedback
                                if right_hip_y > right_knee_y:
                                    feedback = "Good depth!"
                                else:
                                    feedback = "Go deeper!"
                                    mistake_counter['squat_depth'] = mistake_counter.get('squat_depth', 0) + 1
                                    feedback_landmarks = [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]

                    # Extract kinematic features
                    angular_features = AngularFeatureExtractor.extract_all_features(landmarks)
                    
                    # Generate LLM-driven feedback if there are errors
                    llm_message = ""
                    if feedback and feedback not in ["Adjust camera to show full body"]:
                        error_record = {
                            "exercise": selected_exercise,
                            "phase": curl_state if selected_exercise == 'bicep_curls' else squat_state,
                            "errors": [
                                {
                                    "joint": "right_elbow" if selected_exercise == 'bicep_curls' else "right_knee",
                                    "deviation_deg": abs(right_elbow_angle - arm_contracted_angle) if selected_exercise == 'bicep_curls' else abs(right_knee_angle - squat_down_angle),
                                    "type": feedback.lower().replace(" ", "_")
                                }
                            ],
                            "critic_level": 0.5,
                            "user_style": "friendly"
                        }
                        llm_message = llm_feedback.generate_feedback(error_record)
                    
                    # Prepare landmarks data to be sent
                    landmarks_data = [
                        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                        for lm in landmarks
                    ]
                    data_to_send = {
                        "landmarks": landmarks_data,
                        "left_elbow_angle": left_elbow_angle,
                        "right_elbow_angle": right_elbow_angle,
                        "curl_counter": curl_counter,
                        "squat_counter": squat_counter,
                        "left_knee_angle": left_knee_angle,
                        "right_knee_angle": right_knee_angle,
                        "feedback": feedback,
                        "llm_feedback": llm_message,
                        "feedback_landmarks": feedback_landmarks,
                        "kinematic_features": angular_features
                    }
                    last_processed_data = data_to_send
                    await websocket.send_json(data_to_send)
                else:
                    last_processed_data = {"landmarks": []}
                    await websocket.send_json(last_processed_data)
            
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
            finally:
                # Performance tracking
                end_time = time.time()
                frame_times.append(end_time - start_time)

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        logger.info("Client connection closed")
