# FitCoachAR: Real-Time Adaptive Exercise Coaching via Pose Estimation and AR Feedback

**FitCoachAR** is a lightweight, real-time fitness coaching system that monitors exercises through 2D pose estimation, detects common form errors, and provides adaptive feedback through augmented-reality overlays and LLM-driven natural language coaching.

This project implements the concepts from:
- **Lecture 5: Human Kinematic Modeling** - Body-relative coordinate frames, forward kinematics, angular features
- **AIFit (CVPR 2021)** - Exercise modeling, repetition segmentation, active/passive features
- **Modern AR and LLM technologies** - Real-time visual feedback and natural language generation

---

## 🎯 Project Objectives

1. **Real-Time Pose Analysis**: Track push-ups, squats, and bicep curls using MediaPipe Pose
2. **Personalized Calibration**: Adapt thresholds using user's "best-form" repetitions
3. **Dynamic AR Visualization**: Arrows, colored joints, angle indicators for instant feedback
4. **LLM-Driven Coaching**: Natural language feedback generated from quantitative error analysis
5. **Session Summaries**: Comprehensive post-workout reports with recommendations

---

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **`main.py`**: WebSocket server handling real-time pose processing
- **`kinematics.py`**: Body-relative frames, angular feature extraction, forward kinematics
- **`llm_feedback.py`**: Template-based and API-driven natural language generation
- **`filters.py`**: Kalman filtering for landmark smoothing

### Frontend (React/Vite)
- **`App.jsx`**: Main application flow (calibration → workout → summary)
- **`AROverlay.jsx`**: Dynamic AR visualization with colored feedback
- **`Avatar.jsx`**: 3D skeleton rendering using Three.js

---

## 📚 Key Technical Concepts

### 1. Body-Relative Coordinate Frames (Lecture 5)
Implemented in `kinematics.py`:
```python
class BodyRelativeFrame:
    """
    - Origin: pelvis/hip center
    - X: Left-right (mediolateral)
    - Y: Vertical (superior-inferior)
    - Z: Front-back (anteroposterior)
    """
```
This normalization removes dependency on global orientation, making angle measurements consistent regardless of camera position.

### 2. Forward Kinematics & Angular Features
Extracts active (elbow, knee angles) and passive (spine, hip stability) features:
```python
features = AngularFeatureExtractor.extract_all_features(landmarks)
# Returns: elbow_angle, knee_angle, spine_angle, hip_tilt, etc.
```

### 3. Online Repetition Segmentation
State machine detects phase transitions (up → down → up):
```python
segmenter = RepetitionSegmenter(exercise_type='bicep_curls')
new_rep = segmenter.update(angle, thresholds)
```

### 4. LLM-Driven Feedback
Two-tier approach for low latency:
- **Real-time**: Template-based feedback (<100ms)
- **Summary**: Full LLM-generated session report

```python
feedback = llm_feedback.generate_feedback({
    "exercise": "bicep_curls",
    "errors": [{"joint": "right_elbow", "deviation_deg": 12}],
    "critic_level": 0.5
})
# Output: "Nice pace! Try curling a bit higher to finish each rep."
```

### 5. Dynamic AR Visualization
Real-time overlay features:
- ✅ **Green joints**: Correct form
- ❌ **Red joints**: Error detected
- 📐 **Angle arcs**: Visual angle indicators
- ➡️ **Yellow arrows**: Correction guidance

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Webcam

### Backend Setup
```bash
cd fitcoachar/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Required packages** (`requirements.txt`):
```
fastapi
uvicorn[standard]
uvloop
opencv-python
mediapipe
numpy
pykalman
scipy
```

### Frontend Setup
```bash
cd fitcoachar/frontend
npm install
```

---

## ▶️ Running the Application

### 1. Start Backend Server
```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Start Frontend Dev Server
```bash
cd frontend
npm run dev
```

### 3. Open Application
Navigate to `http://localhost:5173`

---

## 📖 Usage Guide

### Step 1: Select Exercise
Choose between **Bicep Curls** or **Squats**

### Step 2: Calibration (Personalization)
- **For Bicep Curls**:
  1. Extend arm fully down → Record
  2. Curl to maximum height → Record
  
- **For Squats**:
  1. Stand straight → Record
  2. Descend to deepest squat → Record

This creates personalized thresholds adapted to your range of motion.

### Step 3: Workout
- Perform your exercise
- Watch for:
  - **Rep counter**: Automatically increments
  - **Form feedback**: "Keep elbow stable!" or "Go deeper!"
  - **LLM coaching**: Natural language tips
  - **AR overlay**: Visual error indicators

### Step 4: Summary
Review your session:
- Total reps completed
- Success rate
- Common mistakes identified
- Recommendations for next session

---

## 🔬 Evaluation Metrics (From Proposal)

| Metric | Description | Target | Status |
|--------|-------------|--------|--------|
| **Segmentation IoU** | Overlap of detected vs. manual rep boundaries | ≥ 0.70 | ✅ Implemented |
| **Latency** | End-to-end delay (camera → feedback) | < 100 ms | ✅ Template-based LLM |
| **Error Detection F1** | Accuracy of form error detection | > 0.80 | ⚠️ Needs validation |
| **Personalization Gain** | Improvement after calibration | +10% | ✅ Implemented |

---

## 🆚 Comparison with AIFit (CVPR 2021)

| Feature | AIFit | FitCoachAR |
|---------|-------|------------|
| **Processing** | Offline (complete video) | Real-time streaming |
| **Pose Estimation** | 3D (MubyNet) | 2D (MediaPipe) |
| **Accessibility** | Motion capture equipment | Webcam only |
| **Personalization** | Expert instructor baseline | User-calibrated thresholds |
| **Feedback** | Static text + images | Dynamic AR + LLM coaching |
| **Latency** | Post-session | <100ms real-time |

---

## 🧩 Project Structure

```
fitcoachar/
├── backend/
│   ├── main.py              # FastAPI WebSocket server
│   ├── kinematics.py        # Body frames, angular features, FK
│   ├── llm_feedback.py      # Natural language generation
│   ├── filters.py           # Kalman smoothing
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main application
│   │   ├── AROverlay.jsx    # Dynamic AR visualization
│   │   ├── Avatar.jsx       # 3D skeleton rendering
│   │   └── App.css          # Styling
│   ├── index.html           # Entry point
│   └── package.json         # Node dependencies
└── README.md                # This file
```

---

## 📊 Technical Implementation Details

### MediaPipe Pose Landmarks (33 joints)
- **Upper Body**: shoulders (11,12), elbows (13,14), wrists (15,16)
- **Core**: hips (23,24), pelvis center (computed)
- **Lower Body**: knees (25,26), ankles (27,28)

### Angle Calculation
Using 3D vector math for joint articulation:
```python
def compute_joint_angle(a, b, c):
    """Angle at joint b, formed by points a-b-c"""
    ba = a - b
    bc = c - b
    cosine = dot(ba, bc) / (norm(ba) * norm(bc))
    return arccos(cosine) * 180/π
```

### Calibration-Based Thresholds
Instead of fixed angles (e.g., "elbow must reach 45°"), we use:
```
threshold = user_calibrated_value ± hysteresis
```
This accounts for individual differences in flexibility and body proportions.

---

## 🎓 Educational Value

This project demonstrates:
1. **Human Kinematic Modeling**: Practical application of joint hierarchies, DoF, coordinate frames
2. **Real-Time Computer Vision**: Streaming pose estimation with <100ms latency
3. **State Machine Design**: Finite state automaton for repetition detection
4. **LLM Integration**: Prompt engineering for context-aware feedback
5. **Full-Stack Development**: React + FastAPI + WebSocket architecture

---

## 🔮 Future Enhancements

### Immediate (V1.1)
- [ ] Add plank exercise support
- [ ] Implement tempo analysis (rep speed)
- [ ] Export workout history to CSV

### Medium-Term (V2.0)
- [ ] Full LLM API integration (GPT-4-mini)
- [ ] Multi-person support
- [ ] Progressive workout plans

### Long-Term (V3.0)
- [ ] Mobile app (React Native + AR Foundation)
- [ ] 3D pose reconstruction for depth analysis
- [ ] Social features (share workouts, leaderboards)

---

## 📝 References

1. **Fieraru et al.** (2021). *AIFit: Automatic 3D Human-Interpretable Feedback Models for Fitness Training*. CVPR 2021.
2. **Lecture 5: Human Kinematics**. McMaster University, Fall 2025.
3. **MediaPipe Pose**. Google Research. https://google.github.io/mediapipe/solutions/pose
4. **Cao et al.** (2017). *Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields*. CVPR 2017.

---

## 👥 Contributors

**Course Project**: Mobile Data Analytics, McMaster University, Fall 2025

---

## 📄 License

This project is for educational purposes. See course guidelines for usage restrictions.

---

## 🤝 Acknowledgments

- **AIFit team** for the foundational methodology
- **MediaPipe team** for the pose estimation framework
- **Course instructors** for guidance on kinematic modeling
