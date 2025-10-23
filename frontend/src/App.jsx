import React, { useState, useEffect, useRef } from 'react';
// import Avatar from './Avatar'; // Disabled - avatar.glb missing
import AROverlay from './AROverlay';
import './App.css';

function App() {
  const [isMediaPipeReady, setIsMediaPipeReady] = useState(false);
  const [status, setStatus] = useState('Loading MediaPipe libraries...');
  const [leftElbowAngle, setLeftElbowAngle] = useState(null);
  const [rightElbowAngle, setRightElbowAngle] = useState(null);
  const [leftKneeAngle, setLeftKneeAngle] = useState(null);
  const [rightKneeAngle, setRightKneeAngle] = useState(null);
  const [repCounter, setRepCounter] = useState(0);
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [llmFeedback, setLlmFeedback] = useState('');
  const [feedbackLandmarks, setFeedbackLandmarks] = useState([]);
  const [poseLandmarks, setPoseLandmarks] = useState([]); // State to hold landmarks for the 3D avatar
  const [appState, setAppState] = useState('selection'); // 'selection', 'calibrating_down', 'calibrating_up', 'workout', 'summary'
  const [workoutSummary, setWorkoutSummary] = useState(null);
  const [selectedExercise, setSelectedExercise] = useState(null);
  const [countdown, setCountdown] = useState(null);
  const countdownIntervalId = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null); // For sending frames
  const ws = useRef(null);
  const frameSenderIntervalId = useRef(null);

  // Effect to check for MediaPipe libraries
  useEffect(() => {
    const mediaPipeCheckInterval = setInterval(() => {
      if (window.drawConnectors && window.POSE_CONNECTIONS) {
        setIsMediaPipeReady(true);
        clearInterval(mediaPipeCheckInterval);
      }
    }, 100);
    return () => clearInterval(mediaPipeCheckInterval);
  }, []);

  // Effect for WebSocket, Camera, and Drawing, when calibrating or working out
  useEffect(() => {
    if (isMediaPipeReady && (appState === 'calibrating_down' || appState === 'calibrating_up' || appState === 'workout')) {
      // Only establish connection if it doesn't exist or is closed
      if (!ws.current || ws.current.readyState === WebSocket.CLOSED) {
        setStatus('Connecting to server...');
        ws.current = new WebSocket('ws://localhost:8001/ws');

        ws.current.onopen = () => setStatus('Connected. Starting camera...');
        ws.current.onclose = () => setStatus('Disconnected.');
        ws.current.onmessage = (event) => {
          const data = JSON.parse(event.data);

          // Handle summary message
          if (data.summary) {
            setWorkoutSummary(data.summary);
            return; // Stop processing after handling summary
          }

          // Handle regular landmark and feedback messages
          if (data.landmarks) {
            if (data.hasOwnProperty('curl_counter')) setRepCounter(data.curl_counter);
            if (data.hasOwnProperty('squat_counter')) setRepCounter(data.squat_counter);
            if (data.left_knee_angle) setLeftKneeAngle(data.left_knee_angle.toFixed(2));
            if (data.right_knee_angle) setRightKneeAngle(data.right_knee_angle.toFixed(2));
            if (data.left_elbow_angle) setLeftElbowAngle(data.left_elbow_angle.toFixed(2));
            if (data.right_elbow_angle) setRightElbowAngle(data.right_elbow_angle.toFixed(2));
            if (data.feedback) setFeedbackMessage(data.feedback);
            if (data.llm_feedback) setLlmFeedback(data.llm_feedback);
            if (data.feedback_landmarks) setFeedbackLandmarks(data.feedback_landmarks);

            // Update the pose landmarks for the 3D avatar
            setPoseLandmarks(data.landmarks);
          }
        };
      }

      // Always try to start the camera if it's not already running
      if (!videoRef.current.srcObject) {
        const startCamera = async () => {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
            }
          } catch (err) {
            console.error('Error accessing camera:', err);
            setStatus('Error: Could not access camera.');
          }
        };
        startCamera();
      }

      // Cleanup function for this effect
      return () => {
        clearInterval(frameSenderIntervalId.current);
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.close();
        }
        if (videoRef.current?.srcObject) {
          videoRef.current.srcObject.getTracks().forEach(track => track.stop());
          videoRef.current.srcObject = null;
        }
      };
    }
  }, [isMediaPipeReady, appState]);

  const startSendingFrames = () => {
    setStatus('Camera running. Streaming frames...');
    frameSenderIntervalId.current = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN && videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        if (video.videoWidth === 0) return;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = canvas.toDataURL('image/jpeg', 0.8);
        if (frame.length > 100) {
          ws.current.send(frame);
        }
      }
    }, 1000 / 30);
  };

  const startCalibration = (exercise) => {
    setSelectedExercise(exercise);
    setAppState('calibrating_down');
  };

  const resetApp = () => {
    setAppState('selection');
    setWorkoutSummary(null);
    setRepCounter(0);
    setFeedbackMessage('');
    setLeftElbowAngle(null);
    setRightElbowAngle(null);
    setLeftKneeAngle(null);
    setRightKneeAngle(null);
  };

  const endWorkout = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ command: 'reset' }));
    }
    setAppState('summary');
  };

  const recordCalibration = (step) => {
    setCountdown(3);
    countdownIntervalId.current = setInterval(() => {
      setCountdown(prev => prev - 1);
    }, 1000);

    setTimeout(() => {
      clearInterval(countdownIntervalId.current);
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ command: `calibrate_${step}`, exercise: selectedExercise }));
      }
      if (step === 'down' || step === 'squat_up') {
        setAppState('calibrating_up');
      } else {
        setAppState('workout');
      }
      setCountdown(null);
    }, 3000);
  };

  return (
    <div className="App">
      <h1>FitCoachAR</h1>
      <p className="subtitle">Real-Time Adaptive Exercise Coaching via Pose Estimation and AR Feedback</p>
      <p>Status: {status}</p>

      {!isMediaPipeReady && <div>Loading MediaPipe libraries...</div>}

      {isMediaPipeReady && appState === 'selection' && (
        <div className="exercise-selection">
          <h2>Select an Exercise</h2>
          <button onClick={() => startCalibration('bicep_curls')}>Bicep Curls</button>
          <button onClick={() => startCalibration('squats')}>Squats</button>
        </div>
      )}

      {isMediaPipeReady && (appState === 'calibrating_down' || appState === 'calibrating_up') && (
        <div className="calibration">
          <h2>Calibrate: {selectedExercise === 'bicep_curls' ? 'Bicep Curl' : 'Squat'}</h2>
          {selectedExercise === 'bicep_curls' && (
            appState === 'calibrating_down' ? (
              <div>
                <p><strong>Step 1:</strong> Extend your right arm fully downwards.</p>
                <button onClick={() => recordCalibration('down')} disabled={countdown !== null}>
                  {countdown !== null ? `Recording in ${countdown}...` : 'Record Down Position'}
                </button>
              </div>
            ) : (
              <div>
                <p><strong>Step 2:</strong> Now, perform a full curl and hold it at the top.</p>
                <button onClick={() => recordCalibration('up')} disabled={countdown !== null}>
                  {countdown !== null ? `Recording in ${countdown}...` : 'Record Up Position'}
                </button>
              </div>
            )
          )}
          {selectedExercise === 'squats' && (
            appState === 'calibrating_down' ? (
              <div>
                <p><strong>Step 1:</strong> Stand up straight, feet shoulder-width apart.</p>
                <button onClick={() => recordCalibration('squat_up')} disabled={countdown !== null}>
                  {countdown !== null ? `Recording in ${countdown}...` : 'Record Up Position'}
                </button>
              </div>
            ) : (
              <div>
                <p><strong>Step 2:</strong> Now, go to your deepest squat and hold the position.</p>
                <button onClick={() => recordCalibration('squat_down')} disabled={countdown !== null}>
                  {countdown !== null ? `Recording in ${countdown}...` : 'Record Down Position'}
                </button>
              </div>
            )
          )}
          <div className="video-container" style={{ position: 'relative', width: '640px', height: '480px', marginTop: '20px' }}>
            <video 
              ref={videoRef} 
              onCanPlay={startSendingFrames}
              autoPlay 
              playsInline 
              muted 
              style={{ opacity: 0.3, width: '640px', height: '480px', position: 'absolute', left: 0, top: 0, zIndex: 1 }}
            ></video>
            <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
            {/* 3D Avatar disabled - avatar.glb missing */}
          </div>
        </div>
      )}

      {isMediaPipeReady && appState === 'summary' && workoutSummary && (
        <div className="summary-screen">
          <h2>Workout Summary</h2>
          <p>Total Reps: <strong>{workoutSummary.total_reps}</strong></p>
          <h3>Common Mistakes:</h3>
          <ul>
            {Object.entries(workoutSummary.mistakes).map(([mistake, count]) => (
              <li key={mistake}>{`${mistake.replace('_', ' ')}: ${count} times`}</li>
            ))}
          </ul>
          <button onClick={resetApp}>Done</button>
        </div>
      )}

      {isMediaPipeReady && appState === 'workout' && (
        <div>
          <div className="workout-stats">
            <button onClick={endWorkout} className="reset-button">End Workout</button>
            <h2>REPS: {repCounter}</h2>
            <h2 className="feedback">{feedbackMessage}</h2>
            {llmFeedback && <p className="llm-feedback">💬 {llmFeedback}</p>}
          </div>
          <div className="angle-display">
            {selectedExercise === 'bicep_curls' && (
              <>
                <p>Left Elbow Angle: <strong>{leftElbowAngle ? `${leftElbowAngle}°` : 'N/A'}</strong></p>
                <p>Right Elbow Angle: <strong>{rightElbowAngle ? `${rightElbowAngle}°` : 'N/A'}</strong></p>
              </>
            )}
            {selectedExercise === 'squats' && (
              <>
                <p>Left Knee Angle: <strong>{leftKneeAngle ? `${leftKneeAngle}°` : 'N/A'}</strong></p>
                <p>Right Knee Angle: <strong>{rightKneeAngle ? `${rightKneeAngle}°` : 'N/A'}</strong></p>
              </>
            )}
          </div>
          <div className="video-container" style={{ position: 'relative', width: '640px', height: '480px' }}>
            <video 
              ref={videoRef} 
              onCanPlay={startSendingFrames}
              autoPlay 
              playsInline 
              muted 
              style={{ opacity: 0.3, width: '640px', height: '480px', position: 'absolute', left: 0, top: 0, zIndex: 1 }}
            ></video>
            <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
            {/* 3D Avatar disabled - avatar.glb missing */}
            <AROverlay 
              landmarks={poseLandmarks}
              feedbackLandmarks={feedbackLandmarks}
              selectedExercise={selectedExercise}
              currentAngles={{
                rightElbow: rightElbowAngle ? parseFloat(rightElbowAngle) : 0,
                rightKnee: rightKneeAngle ? parseFloat(rightKneeAngle) : 0
              }}
              targetAngles={{
                rightElbow: 45,
                rightKnee: 90
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
