import React, { useState, useEffect, useRef, useCallback } from 'react';
// import Avatar from './Avatar'; // Disabled - avatar.glb missing
import AROverlay from './AROverlay';
import './App.css';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

const EXERCISES = [
  {
    id: 'bicep_curls',
    label: 'Bicep Curls',
    description: 'Track elbow angles and tempo for stronger curls.'
  },
  {
    id: 'squats',
    label: 'Squats',
    description: 'Monitor depth and knee alignment for safer squats.'
  }
];

// Preset session templates
const SESSION_PRESETS = [
  {
    id: 'quick_squats',
    name: '3x10 Squats',
    sets: [
      { exercise: 'squats', reps: 10 },
      { exercise: 'squats', reps: 10 },
      { exercise: 'squats', reps: 10 },
    ]
  },
  {
    id: 'quick_curls',
    name: '3x12 Bicep Curls',
    sets: [
      { exercise: 'bicep_curls', reps: 12 },
      { exercise: 'bicep_curls', reps: 12 },
      { exercise: 'bicep_curls', reps: 12 },
    ]
  },
  {
    id: 'circuit',
    name: 'Circuit: Squats + Curls',
    sets: [
      { exercise: 'squats', reps: 10 },
      { exercise: 'bicep_curls', reps: 10 },
      { exercise: 'squats', reps: 10 },
      { exercise: 'bicep_curls', reps: 10 },
    ]
  },
];

const CANONICAL_BASELINES = {
  bicep_curls: {
    extended: 160,
    contracted: 30,
  },
  squats: {
    up: 160,
    down: 50,
  },
};

const createDefaultSummary = () => ({
  records: [],
  active: { common: null, calibration: null },
  critics: { common: 0.2 }
});

function App() {
  const [isMediaPipeReady, setIsMediaPipeReady] = useState(false);
  const [status, setStatus] = useState('Loading MediaPipe libraries...');
  const [leftElbowAngle, setLeftElbowAngle] = useState(null);
  const [rightElbowAngle, setRightElbowAngle] = useState(null);
  const [leftKneeAngle, setLeftKneeAngle] = useState(null);
  const [rightKneeAngle, setRightKneeAngle] = useState(null);
  const [repCounter, setRepCounter] = useState(0);
  const repCounterRef = useRef(repCounter);
  const lastErrorRep = useRef(-1);
  const [errorReps, setErrorReps] = useState(0);
  const [repTimestamps, setRepTimestamps] = useState([]);
  const [repDurations, setRepDurations] = useState([]);
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [llmFeedback, setLlmFeedback] = useState('');
  const [feedbackLandmarks, setFeedbackLandmarks] = useState([]);
  const [arrowFeedback, setArrowFeedback] = useState([]);  // New: structured arrow data
  const [poseLandmarks, setPoseLandmarks] = useState([]); // State to hold landmarks for the 3D avatar
  const [appState, setAppState] = useState('selection'); // 'selection', 'session_builder', 'calibration_countdown', 'calibrating_live', 'calibration_saving', 'workout', 'summary'
  
  // Session state
  const [sessionConfig, setSessionConfig] = useState([]);
  const [sessionProgress, setSessionProgress] = useState(null);
  const [sessionName, setSessionName] = useState('My Workout');
  const [workoutSummary, setWorkoutSummary] = useState(null);
  const workoutSummaryRef = useRef(workoutSummary);
  const [llmSummary, setLlmSummary] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLlmLoading, setIsLlmLoading] = useState(false);
  const [formAnalysis, setFormAnalysis] = useState(null);  // Form analysis results
  const [formSnapshots, setFormSnapshots] = useState([]);  // Collected form snapshots
  const [postRepCommand, setPostRepCommand] = useState(null);  // Realtime coaching command after rep
  const [selectedExercise, setSelectedExercise] = useState(null);
  const [countdown, setCountdown] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const [roundTripMs, setRoundTripMs] = useState(null);
  const [backendName, setBackendName] = useState(null);
  const [appMode, setAppMode] = useState('common');
  const [calibrationSummary, setCalibrationSummary] = useState({});
  const [selectedRecordId, setSelectedRecordId] = useState(null);
  const [criticInputs, setCriticInputs] = useState({ common: '0.200' });
  const [latestCalibration, setLatestCalibration] = useState(null);
  const [calibrationProgress, setCalibrationProgress] = useState(null);
  const [showCalibrationManager, setShowCalibrationManager] = useState(false);
  const appModeRef = useRef(appMode);
  const showCalibrationManagerRef = useRef(showCalibrationManager);
  const selectedRecordIdRef = useRef(selectedRecordId);
  const countdownIntervalId = useRef(null);
  const latestCalibrationRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null); // For sending frames
  const ws = useRef(null);
  const frameSenderIntervalId = useRef(null);
  const selectedExerciseRef = useRef(null);
  const awaitingResponseRef = useRef(false);

  const sendCommand = useCallback((payload) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    const message = { ...payload };
    if (!message.exercise) {
      message.exercise = selectedExerciseRef.current;
    }
    ws.current.send(JSON.stringify(message));
  }, []);

  const updateSummary = useCallback((exercise, updater) => {
    setCalibrationSummary(prev => {
      const prevInfo = prev[exercise] ? { ...prev[exercise] } : createDefaultSummary();
      const nextInfo = updater(prevInfo);
      return { ...prev, [exercise]: nextInfo };
    });
  }, []);

  const resetMetrics = useCallback(() => {
    setWorkoutSummary(null);
    setRepCounter(0);
    setErrorReps(0);
    setRepTimestamps([]);
    setRepDurations([]);
    setFeedbackMessage('');
    setLlmFeedback('');
    setFeedbackLandmarks([]);
    setArrowFeedback([]);
    setPoseLandmarks([]);
    setLeftElbowAngle(null);
    setRightElbowAngle(null);
    setLeftKneeAngle(null);
    setRightKneeAngle(null);
    setLatencyMs(null);
    setRoundTripMs(null);
    setCalibrationProgress(null);
    setCountdown(null);
    awaitingResponseRef.current = false;
    setBackendName(null);
    setSessionProgress(null);
    setFormAnalysis(null);
    setFormSnapshots([]);
    setPostRepCommand(null);
  }, []);

  const handleCriticChange = useCallback((mode, value) => {
    setCriticInputs(prev => ({ ...prev, [mode]: value }));
  }, []);

  const handleCriticSubmit = useCallback((mode) => {
    const value = parseFloat(criticInputs[mode]);
    if (!Number.isFinite(value)) return;
    sendCommand({ command: 'set_critic', mode, value });
  }, [criticInputs, sendCommand]);

  const handleDeleteCalibration = useCallback((recordId) => {
    if (!recordId || !selectedExerciseRef.current) return;
    sendCommand({
      command: 'delete_calibration',
      exercise: selectedExerciseRef.current,
      record_id: recordId,
    });
  }, [sendCommand]);

  useEffect(() => {
    const summary = selectedExercise ? (calibrationSummary[selectedExercise] || createDefaultSummary()) : createDefaultSummary();
    const critics = summary.critics || { common: 0.2 };
    setCriticInputs({
      common: Number(critics.common ?? 0.2).toFixed(3),
    });
  }, [selectedExercise, calibrationSummary]);

  useEffect(() => {
    if (!selectedExercise) return;
    const summary = calibrationSummary[selectedExercise];
    if (!summary) return;
    if (showCalibrationManager) {
      if (selectedRecordId && !summary.records.some(r => r.id === selectedRecordId)) {
        const fallbackRecord = summary.records[0] ? summary.records[0].id : null;
        selectedRecordIdRef.current = fallbackRecord;
        setSelectedRecordId(fallbackRecord);
      } else if (!selectedRecordId && summary.records.length) {
        selectedRecordIdRef.current = summary.records[0].id;
        setSelectedRecordId(summary.records[0].id);
      }
    } else if (selectedRecordId !== null) {
      selectedRecordIdRef.current = null;
      setSelectedRecordId(null);
    }
  }, [selectedExercise, calibrationSummary, selectedRecordId, showCalibrationManager]);

  const currentSummary = selectedExercise ? (calibrationSummary[selectedExercise] || createDefaultSummary()) : createDefaultSummary();
  const currentRecords = currentSummary.records || [];
  const activeCommonId = currentSummary.active ? currentSummary.active.common : null;
  const activeCalibrationId = currentSummary.active ? currentSummary.active.calibration : null;
  const workoutRecord = currentRecords.find(r => r.id === activeCommonId) || null;
  const calibrationRecord = currentRecords.find(r => r.id === activeCalibrationId) || null;
  const selectedRecord = showCalibrationManager
    ? currentRecords.find(r => r.id === selectedRecordId) || null
    : null;
  const latestEta = latestCalibration && latestCalibration.record ? latestCalibration.record.eta : null;
  const showLatestCalibration = latestCalibration && latestCalibration.exercise === selectedExercise;
  const canonicalBaseline = selectedExercise ? CANONICAL_BASELINES[selectedExercise] || null : null;
  const usingDefaultWorkout = !workoutRecord;
  const usingDefaultCalibration = !calibrationRecord;
  const workoutBaselineLabel = usingDefaultWorkout || !workoutRecord
    ? 'Default canonical angles'
    : `Personalized capture on ${new Date(workoutRecord.timestamp).toLocaleString()}`;
  const calibrationBaselineLabel = usingDefaultCalibration || !calibrationRecord
    ? 'Default canonical angles'
    : `Personalized capture on ${new Date(calibrationRecord.timestamp).toLocaleString()}`;

  useEffect(() => {
    showCalibrationManagerRef.current = showCalibrationManager;
  }, [showCalibrationManager]);

  useEffect(() => {
    selectedRecordIdRef.current = selectedRecordId;
  }, [selectedRecordId]);

  useEffect(() => {
    appModeRef.current = appMode;
  }, [appMode]);

  useEffect(() => {
    latestCalibrationRef.current = latestCalibration;
  }, [latestCalibration]);

  useEffect(() => {
    repCounterRef.current = repCounter;
    console.log('repCounter state ->', repCounter);
  }, [repCounter]);

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

  // Maintain a WebSocket connection once MediaPipe is ready
  useEffect(() => {
    if (!isMediaPipeReady) return undefined;

    let cancelled = false;
    let reconnectTimer = null;

    const ensureConnection = () => {
      if (cancelled) return;
      if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
        return;
      }

      setStatus('Connecting to server...');
      const socket = new WebSocket(`${WS_URL}/ws`);
      ws.current = socket;

      socket.onopen = () => {
        if (cancelled) return;
        setStatus('Connected. Ready for actions.');
        awaitingResponseRef.current = false;
        if (selectedExerciseRef.current) {
          socket.send(JSON.stringify({ command: 'select_exercise', exercise: selectedExerciseRef.current }));
          socket.send(JSON.stringify({ command: 'list_calibrations', exercise: selectedExerciseRef.current }));
          socket.send(JSON.stringify({ command: 'set_mode', mode: appModeRef.current, exercise: selectedExerciseRef.current }));
        }
      };

      socket.onclose = () => {
        awaitingResponseRef.current = false;
        if (cancelled) return;
        setStatus('Disconnected. Retrying...');
        ws.current = null;
        reconnectTimer = setTimeout(ensureConnection, 1500);
      };

      socket.onerror = () => {
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
          socket.close();
        }
      };

      socket.onmessage = (event) => {
        awaitingResponseRef.current = false;
        const showManager = showCalibrationManagerRef.current;
        const currentSelectedId = selectedRecordIdRef.current;
        const data = JSON.parse(event.data);

        if (data.event) {
          const eventType = data.event;
          if (eventType === 'exercise_selected') {
            const exercise = data.exercise;
            updateSummary(exercise, () => ({
              records: data.records || [],
              active: data.active || { common: null, calibration: null },
              critics: data.critics || { common: 0.2, calibration: 0.2 }
            }));
            if (data.mode) setAppMode(data.mode);
            return;
          }
          if (eventType === 'mode_updated') {
            const exercise = data.exercise;
            setAppMode(data.mode);
            if (data.activeCalibration) {
              selectedRecordIdRef.current = data.activeCalibration.id || null;
              setSelectedRecordId(data.activeCalibration.id || null);
            }
            if (data.critics) {
              updateSummary(exercise, prev => ({
                ...prev,
                critics: data.critics,
                active: prev.active || { common: null, calibration: null }
              }));
            }
            return;
          }
          if (eventType === 'critic_updated') {
            const exercise = data.exercise;
            updateSummary(exercise, prev => ({
              ...prev,
              critics: data.critics || prev.critics
            }));
            return;
          }
          if (eventType === 'calibration_list') {
            const exercise = data.exercise;
            updateSummary(exercise, () => ({
              records: data.records || [],
              active: data.active || { common: null, calibration: null },
              critics: data.critics || { common: 0.2, calibration: 0.2 }
            }));
            if (exercise === selectedExerciseRef.current) {
              const active = data.active || {};
              const records = data.records || [];
              const fallback = showManager ? (active.calibration || active.common) : (active.common || active.calibration);
              let nextId = fallback || null;
              if (!nextId) {
                if (currentSelectedId && records.some(r => r.id === currentSelectedId)) {
                  nextId = currentSelectedId;
                } else if (currentSelectedId === null) {
                  nextId = null;
                } else if (records.length) {
                  nextId = records[0].id;
                }
              }
              if (nextId !== currentSelectedId) {
                selectedRecordIdRef.current = nextId;
                setSelectedRecordId(nextId);
              }
            }
            return;
          }
          if (eventType === 'calibration_applied') {
            const exercise = data.exercise;
            updateSummary(exercise, prev => ({
              ...prev,
              active: {
                ...prev.active,
                [data.mode]: data.activeCalibration ? data.activeCalibration.id : null
              }
            }));
            if (data.activeCalibration) {
              selectedRecordIdRef.current = data.activeCalibration.id;
              setSelectedRecordId(data.activeCalibration.id);
            } else {
              if (data.mode === 'common' && !showManager) {
                selectedRecordIdRef.current = null;
                setSelectedRecordId(null);
              }
              if (data.mode === 'calibration' && showManager) {
                selectedRecordIdRef.current = null;
                setSelectedRecordId(null);
              }
            }
            return;
          }
          if (eventType === 'calibration_deleted') {
            const exercise = data.exercise;
            updateSummary(exercise, () => ({
              records: data.records || [],
              active: data.active || { common: null, calibration: null },
              critics: data.critics || { common: 0.2, calibration: 0.2 }
            }));
            if (exercise === selectedExerciseRef.current) {
              const active = data.active || {};
              const fallback = showManager ? (active.calibration || active.common) : (active.common || active.calibration);
              const firstRecord = data.records && data.records.length ? data.records[0].id : null;
              const nextId = fallback || firstRecord || null;
              if (nextId !== currentSelectedId) {
                selectedRecordIdRef.current = nextId;
                setSelectedRecordId(nextId);
              }
              const latest = latestCalibrationRef.current;
              if (latest && latest.record && latest.record.id === data.deleted_id) {
                latestCalibrationRef.current = null;
                setLatestCalibration(null);
              }
              setStatus('Calibration deleted.');
            }
            return;
          }
          if (eventType === 'calibration_started') {
            setStatus('Auto calibration running. Perform a few full reps.');
            setCalibrationProgress(null);
            setAppState('calibrating_live');
            return;
          }
          if (eventType === 'calibration_complete') {
            const exercise = data.exercise;
            const record = data.record;
            setCalibrationProgress(null);
            if (record) {
              updateSummary(exercise, prev => {
                const existing = prev.records.filter(r => r.id !== record.id);
                return {
                  ...prev,
                  records: [record, ...existing],
                  active: {
                    ...prev.active,
                    [data.mode]: record.id
                  }
                };
              });
              const latestPayload = { exercise, record };
              latestCalibrationRef.current = latestPayload;
              setLatestCalibration(latestPayload);
              selectedRecordIdRef.current = record.id;
              setSelectedRecordId(record.id);
              showCalibrationManagerRef.current = true;
              setShowCalibrationManager(true);
              setAppMode('common');
              setAppState('selection');
              setStatus('Calibration saved. Review it in Manage Calibrations or begin a workout.');
              sendCommand({ command: 'set_mode', mode: 'common', exercise });
              sendCommand({ command: 'list_calibrations', exercise });
            }
            return;
          }
          if (eventType === 'calibration_error') {
            setCalibrationProgress(null);
            if (data.message) setStatus(`Calibration error: ${data.message}`);
            setAppState('selection');
            return;
          }
          if (eventType === 'calibration_cancelled') {
            setCalibrationProgress(null);
            setAppState('selection');
            setStatus('Calibration cancelled. Choose an action to continue.');
            return;
          }
          // Session events
          if (eventType === 'session_started') {
            setSessionProgress(data.progress);
            if (data.progress?.current_exercise) {
              setSelectedExercise(data.progress.current_exercise);
              selectedExerciseRef.current = data.progress.current_exercise;
            }
            setAppState('workout');
            setStatus(`Session started: ${data.progress?.session_name || 'Workout'}`);
            return;
          }
          if (eventType === 'set_started' || eventType === 'set_skipped') {
            setSessionProgress(data.progress);
            if (data.progress?.current_exercise) {
              setSelectedExercise(data.progress.current_exercise);
              selectedExerciseRef.current = data.progress.current_exercise;
            }
            setRepCounter(0);
            setStatus(`Set ${data.progress?.current_set_index + 1}/${data.progress?.total_sets}: ${data.progress?.current_exercise}`);
            return;
          }
          if (eventType === 'session_complete') {
            setSessionProgress(data.progress);
            // Save full session data (includes all sets)
            if (data.session) {
              saveSessionData(data.session);
              sendSessionToLLM(data.session);
            }
            setStatus('Session complete! Great workout!');
            return;
          }
          if (eventType === 'session_ended') {
            // Save full session data (includes all sets) before clearing
            if (data.session) {
              saveSessionData(data.session);
              sendSessionToLLM(data.session);
            }
            setSessionProgress(null);
            setStatus('Session ended.');
            return;
          }
          if (eventType === 'session_error') {
            setStatus(`Session error: ${data.message}`);
            return;
          }
          if (eventType === 'session_progress') {
            if (data.progress) setSessionProgress(data.progress);
            return;
          }
          return;
        }

        // Handle summary message
        if (data.summary) {
          const fullSummary = {
            total_reps: 0,
            success_rate: 0,
            mistakes: {},
            avg_tempo: 0,
            exercise: selectedExerciseRef.current || '',
            ...data.summary, // Overwrite defaults with any fields from the backend
          };
          setWorkoutSummary(fullSummary);
          return; // Stop processing after handling summary
        }

        // Handle regular landmark and feedback messages
        if (data.backend) setBackendName(data.backend);

        if (data.landmarks) {
          if (data.hasOwnProperty('rep_count')) {
            console.log('rep_count ->', data.rep_count);
            setRepCounter(data.rep_count);
          }
            
          if (data.hasOwnProperty('latency_ms')) setLatencyMs(data.latency_ms);
          if (data.hasOwnProperty('client_ts')) {
            const rtt = performance.now() - data.client_ts;
            if (Number.isFinite(rtt)) setRoundTripMs(rtt);
          }
          if (data.left_knee_angle) setLeftKneeAngle(data.left_knee_angle.toFixed(2));
          if (data.right_knee_angle) setRightKneeAngle(data.right_knee_angle.toFixed(2));
          if (data.left_elbow_angle) setLeftElbowAngle(data.left_elbow_angle.toFixed(2));
          if (data.right_elbow_angle) setRightElbowAngle(data.right_elbow_angle.toFixed(2));
          if (data.feedback) {
            // Only show text for errors, not for positive feedback
            const isPositive = ['good rep', 'great curl', 'good depth', 'perfect'].some(
              phrase => data.feedback.toLowerCase().includes(phrase)
            );
            setFeedbackMessage(isPositive ? '' : data.feedback);  // Hide positive text
            
            const isError = !isPositive && data.feedback.trim() !== '';
            if (isError && repCounterRef.current > 0 && lastErrorRep.current !== repCounterRef.current) {
              setErrorReps(prev => prev + 1);
              lastErrorRep.current = repCounterRef.current;
            }
          } else {
            setFeedbackMessage('');
          }
          // Arrow feedback from backend (visual coaching)
          if (data.arrow_feedback) {
            setArrowFeedback(data.arrow_feedback);
          } else {
            setArrowFeedback([]);
          }
          if (data.coach_tip) setLlmFeedback(data.coach_tip);  // Realtime coach tip
          // Post-rep coaching command (aligned with form states)
          if (data.post_rep_command !== undefined) {
            setPostRepCommand(data.post_rep_command);
          }
          if (data.feedback_landmarks) setFeedbackLandmarks(data.feedback_landmarks);
          if (data.calibration_progress){
            setCalibrationProgress(data.calibration_progress);
          } 
          if (data.rep_timestamps) setRepTimestamps(data.rep_timestamps);
          if (data.session_progress) setSessionProgress(data.session_progress);
          // Collect form snapshots (from WebSocket directly or from session progress)
          if (data.form_snapshots && data.form_snapshots.length > 0) {
            setFormSnapshots(data.form_snapshots);
          } else if (data.session_progress?.current_set?.form_snapshots) {
            setFormSnapshots(data.session_progress.current_set.form_snapshots);
          }

          // Update the pose landmarks for the 3D avatar
          setPoseLandmarks(data.landmarks);
        }
      };
    };

    ensureConnection();

    return () => {
      cancelled = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
    };
  }, [isMediaPipeReady, updateSummary, sendCommand]);

  // Manage camera lifecycle separately from the WebSocket
  useEffect(() => {
    if (!isMediaPipeReady) return;
    const streamingStates = ['calibration_countdown', 'calibrating_live', 'workout'];
    const shouldStream = streamingStates.includes(appState);

    if (!shouldStream) {
      clearInterval(frameSenderIntervalId.current);
      awaitingResponseRef.current = false;
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
      return;
    }

    if (!videoRef.current?.srcObject) {
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
  }, [appState, isMediaPipeReady]);

  useEffect(() => {
    return () => {
      clearInterval(frameSenderIntervalId.current);
      if (countdownIntervalId.current) {
        clearInterval(countdownIntervalId.current);
        countdownIntervalId.current = null;
      }
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
    };
  }, []);

  useEffect(() => {
    selectedExerciseRef.current = selectedExercise;
    if (
      appState === 'workout' &&
      selectedExercise &&
      ws.current?.readyState === WebSocket.OPEN
    ) {
      ws.current.send(JSON.stringify({ command: 'select_exercise', exercise: selectedExercise }));
    }
  }, [selectedExercise, appState]);

  const startSendingFrames = () => {
    awaitingResponseRef.current = false;
    setStatus('Camera running. Streaming frames...');
    frameSenderIntervalId.current = setInterval(() => {
      if (awaitingResponseRef.current) {
        return;
      }
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
          try {
            const payload = JSON.stringify({ frame, ts: performance.now() });
            ws.current.send(payload);
            awaitingResponseRef.current = true;
          } catch (err) {
            awaitingResponseRef.current = false;
            console.error('Failed to send frame payload:', err);
          }
        }
      }
    }, 1000 / 30);
  };

  const handleSelectExercise = (exercise) => {
    if (selectedExerciseRef.current !== exercise) {
      resetMetrics();
      latestCalibrationRef.current = null;
      setLatestCalibration(null);
      setSelectedRecordId(null);
      selectedRecordIdRef.current = null;
      showCalibrationManagerRef.current = false;
      setShowCalibrationManager(false);
    }
    setSelectedExercise(exercise);
    selectedExerciseRef.current = exercise;
    setStatus(`Selected ${EXERCISES.find(e => e.id === exercise)?.label || exercise}. Choose an action below.`);
    sendCommand({ command: 'select_exercise', exercise });
    sendCommand({ command: 'list_calibrations', exercise });
  };

  // Quick workout with single exercise (legacy mode)
  const beginWorkout = () => {
    if (!selectedExerciseRef.current) {
      setStatus('Select an exercise before starting.');
      return;
    }
    resetMetrics();
    const exercise = selectedExerciseRef.current;
    setAppMode('common');
    showCalibrationManagerRef.current = false;
    setShowCalibrationManager(false);
    sendCommand({ command: 'set_mode', mode: 'common', exercise });
    setAppState('workout');
    setStatus('Workout started. Perform reps while watching the overlay.');
  };

  // Open session builder
  const openSessionBuilder = () => {
    setSessionConfig([]);
    setSessionName('My Workout');
    setAppState('session_builder');
    setStatus('Build your workout session.');
  };

  // Add a set to session config
  const addSetToSession = (exercise, reps) => {
    setSessionConfig(prev => [...prev, { exercise, reps: parseInt(reps) || 10 }]);
  };

  // Remove a set from session config
  const removeSetFromSession = (index) => {
    setSessionConfig(prev => prev.filter((_, i) => i !== index));
  };

  // Load a preset session
  const loadPreset = (preset) => {
    setSessionConfig([...preset.sets]);
    setSessionName(preset.name);
  };

  // Start the configured session
  const startSession = () => {
    if (sessionConfig.length === 0) {
      setStatus('Add at least one set to your session.');
      return;
    }
    resetMetrics();
    setAppMode('common');
    showCalibrationManagerRef.current = false;
    setShowCalibrationManager(false);
    sendCommand({
      command: 'start_session',
      name: sessionName,
      sets: sessionConfig
    });
  };

  // Move to next set in session
  const nextSet = () => {
    sendCommand({ command: 'next_set' });
  };

  // Skip current set
  const skipSet = () => {
    sendCommand({ command: 'skip_set' });
  };

  // End session early - the backend will send session_ended event with full data
  const endSession = () => {
    sendCommand({ command: 'end_session' });
    // The WebSocket handler for 'session_ended' will save data and send to LLM
  };

  // Save complete session data to backend
  const saveSessionData = async (sessionData) => {
    try {
      await fetch(`${BACKEND_URL}/save_session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionData)
      });
    } catch (error) {
      console.error('Failed to save session data:', error);
    }
  };

  // Send full session data to LLM for summary (with form analysis)
  const sendSessionToLLM = (sessionData) => {
    // Calculate success rate (capped at 1.0 = 100%)
    const rawSuccessRate = sessionData.total_reps_target > 0 
      ? sessionData.total_reps_completed / sessionData.total_reps_target 
      : 0;
    const successRate = Math.min(1.0, rawSuccessRate);
    
    // Calculate tempo (seconds per rep)
    const avgTempo = sessionData.duration_seconds > 0 && sessionData.total_reps_completed > 0
      ? sessionData.duration_seconds / sessionData.total_reps_completed
      : 0;
    
    // Count total mistakes
    const mistakesObj = sessionData.all_mistakes || {};
    const totalErrors = Object.values(mistakesObj).reduce((sum, count) => sum + count, 0);
    
    // Build summary matching backend SessionData model
    const sessionSummary = {
      total_reps: sessionData.total_reps_completed,
      success_rate: successRate,
      mistakes: { errors_detected: totalErrors, ...mistakesObj },
      avg_tempo: avgTempo,
      exercise: `session: ${sessionData.name}`,
      // Extra fields for context (backend will ignore but useful for display)
      session_details: sessionData
    };
    
    setWorkoutSummary(sessionSummary);
    workoutSummaryRef.current = sessionSummary;
    setAppState('summary');
    
    // Collect all form snapshots from all sets in the session
    const allFormSnapshots = [];
    if (sessionData.sets) {
      sessionData.sets.forEach(set => {
        if (set.form_snapshots && set.form_snapshots.length > 0) {
          allFormSnapshots.push(...set.form_snapshots);
        }
      });
    }
    
    // If we have form snapshots, fetch form analysis first
    // Use the exercise from the first set, or fallback to a generic name
    const primaryExercise = sessionData.sets?.[0]?.exercise || 'workout';
    
    if (allFormSnapshots.length > 0) {
      fetchFormAnalysisAndSummary(
        primaryExercise, 
        allFormSnapshots, 
        sessionData.total_reps_completed, 
        sessionSummary
      );
    } else {
      // No form snapshots, just fetch LLM summary with basic data
      fetchLlmSummary(sessionSummary);
    }
  };

  const beginCalibration = () => {
    if (!selectedExerciseRef.current) {
      setStatus('Select an exercise before capturing a calibration.');
      return;
    }
    resetMetrics();
    const exercise = selectedExerciseRef.current;
    setAppMode('calibration');
    showCalibrationManagerRef.current = false;
    setShowCalibrationManager(false);
    sendCommand({ command: 'set_mode', mode: 'calibration', exercise });
    setCalibrationProgress(null);
    if (countdownIntervalId.current) {
      clearInterval(countdownIntervalId.current);
      countdownIntervalId.current = null;
    }
    let remaining = 5;
    setCountdown(remaining);
    setAppState('calibration_countdown');
    setStatus('Calibration starts in 5 seconds. Get into your starting position.');
    countdownIntervalId.current = setInterval(() => {
      remaining -= 1;
      setCountdown(remaining);
      if (remaining <= 0) {
        clearInterval(countdownIntervalId.current);
        countdownIntervalId.current = null;
        setCountdown(null);
        sendCommand({ command: 'start_auto_calibration', exercise });
        setAppState('calibrating_live');
        setStatus('Calibration: perform a few full reps. We will capture your extremes automatically.');
      }
    }, 1000);
  };

  const resetApp = () => {
    setAppState('selection');
    resetMetrics();
    setSelectedExercise(null);
    selectedExerciseRef.current = null;
    setAppMode('common');
    selectedRecordIdRef.current = null;
    setSelectedRecordId(null);
    latestCalibrationRef.current = null;
    setLatestCalibration(null);
    showCalibrationManagerRef.current = false;
    setShowCalibrationManager(false);
  };

  const calculateRepDurations = (timestamps) => {
    if (timestamps.length < 2) return [];
    const durations = [];
    for (let i = 1; i < timestamps.length; i++) {
      durations.push(timestamps[i] - timestamps[i - 1]);
    }
    return durations;
  };

  const calculateAverageTempo = (durations) => {
    if (durations.length === 0) return 0;
    const sum = durations.reduce((a, b) => a + b, 0);
    return sum / durations.length;
  };

  const endWorkout = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ command: 'reset' }));
    }
    // Create a complete summary object from the current state
    const durations = calculateRepDurations(repTimestamps);
    const avgTempo = calculateAverageTempo(durations);

    const successRate = repCounter > 0 ? (repCounter - errorReps) / repCounter : 0.0;
    const finalSummary = {
      total_reps: repCounter,
      success_rate: successRate,
      mistakes: { 'errors_detected': errorReps },
      avg_tempo: avgTempo,
      rep_durations: durations, // Add individual rep durations
      exercise: selectedExerciseRef.current
    };

    // Set the summary state and ref for the current session
    setWorkoutSummary(finalSummary);
    workoutSummaryRef.current = finalSummary;
    setAppState('summary');

    // Save workout data
    saveWorkoutData(finalSummary);
    
    // Fetch form analysis first, then pass to LLM for aligned summary
    if (formSnapshots.length > 0) {
      fetchFormAnalysisAndSummary(selectedExerciseRef.current, formSnapshots, repCounter, finalSummary);
    } else {
      // No form snapshots, just fetch LLM summary with basic data
      fetchLlmSummary(finalSummary);
    }
  };

  const fetchFormAnalysisAndSummary = async (exercise, snapshots, totalReps, sessionData) => {
    if (!exercise || snapshots.length === 0) return;
    try {
      // First, get the form analysis
      const analysisResponse = await fetch(`${BACKEND_URL}/analyze_form`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exercise,
          form_snapshots: snapshots,
          total_reps: totalReps
        })
      });
      const analysisData = await analysisResponse.json();
      
      if (analysisData.status === 'success') {
        setFormAnalysis(analysisData.analysis);
        
        // Now fetch LLM summary with the form analysis included
        const enrichedSessionData = {
          ...sessionData,
          form_analysis: {
            score: analysisData.analysis.score,
            good_reps: analysisData.analysis.good_reps,
            total_reps: analysisData.analysis.total_reps,
            top_issues: analysisData.analysis.top_issues,
            form_states_count: analysisData.analysis.form_states_count
          }
        };
        fetchLlmSummary(enrichedSessionData);
      } else {
        // Fallback to basic summary
        fetchLlmSummary(sessionData);
      }
    } catch (error) {
      console.error('Failed to fetch form analysis:', error);
      fetchLlmSummary(sessionData);
    }
  };

  const saveWorkoutData = async (sessionData) => {
    if (!sessionData.exercise) return;
    try {
      await fetch(`${BACKEND_URL}/save_workout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionData)
      });
    } catch (error) {
      console.error('Failed to save workout data:', error);
    }
  };

  const fetchLlmSummary = async (sessionData) => {
    if (!sessionData.exercise) return;
    setIsLlmLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionData)
      });
      const data = await response.json();
      setLlmSummary(data.summary);
      setChatHistory([]); // Reset chat history for the new summary
    } catch (error) {
      console.error('Failed to fetch LLM summary:', error);
      setLlmSummary('Could not load AI summary. Please try again.');
    } finally {
      setIsLlmLoading(false);
    }
  };

  const handleAskQuestion = async (question) => {
    if (!question.trim() || !workoutSummaryRef.current) return;

    const newChatHistory = [...chatHistory, { role: 'user', content: question }];
    setChatHistory(newChatHistory);
    setIsLlmLoading(true);

    try {
      const response = await fetch(`${BACKEND_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_data: workoutSummaryRef.current, question })
      });
      const data = await response.json();
      setChatHistory([...newChatHistory, { role: 'assistant', content: data.answer }]);
    } catch (error) {
      console.error('Failed to get answer from LLM:', error);
      setChatHistory([...newChatHistory, { role: 'assistant', content: 'Sorry, I had trouble getting an answer. Please try again.' }]);
    } finally {
      setIsLlmLoading(false);
    }
  };

  const cancelCalibration = useCallback(() => {
    if (countdownIntervalId.current) {
      clearInterval(countdownIntervalId.current);
      countdownIntervalId.current = null;
    }
    setCountdown(null);
    if (!selectedExerciseRef.current) {
      setAppState('selection');
      setStatus('Calibration cancelled.');
      return;
    }
    const exercise = selectedExerciseRef.current;
    setAppMode('common');
    appModeRef.current = 'common';
    showCalibrationManagerRef.current = false;
    setShowCalibrationManager(false);
    setAppState('selection');
    setStatus('Calibration cancelled. Choose an action to continue.');
    sendCommand({ command: 'set_mode', mode: 'common', exercise });
    sendCommand({ command: 'cancel_calibration', exercise });
  }, [sendCommand]);

  const finishCalibration = () => {
    const exercise = selectedExerciseRef.current;
    if (!exercise) {
      setStatus('Select an exercise before finishing calibration.');
      return;
    }
    setAppState('calibration_saving');
    setStatus('Finishing calibration...');
    sendCommand({ command: 'finalize_auto_calibration', exercise });
  };

  return (
    <div className="App">
      <h1>FitCoachAR</h1>
      <p className="subtitle">Real-Time Adaptive Exercise Coaching via Pose Estimation and AR Feedback</p>
      <p>Status: {status}</p>
      {backendName && <p>Backend: {backendName}</p>}
      {latencyMs !== null && <p>Backend latency: {latencyMs.toFixed(1)} ms</p>}
      {roundTripMs !== null && <p>Total latency: {roundTripMs.toFixed(1)} ms</p>}
      <div className="action-bar">
        <button onClick={beginWorkout} disabled={!selectedExercise}>Quick Workout</button>
        <button onClick={openSessionBuilder} className="primary-button">Build Session</button>
        <button onClick={beginCalibration} disabled={!selectedExercise}>New Calibration</button>
        <button
          onClick={() => {
            if (!selectedExerciseRef.current) return;
            const next = !showCalibrationManager;
            showCalibrationManagerRef.current = next;
            setShowCalibrationManager(next);
            if (next) {
              sendCommand({ command: 'list_calibrations', exercise: selectedExerciseRef.current });
            }
          }}
          disabled={!selectedExercise}
        >
          {showCalibrationManager ? 'Hide Calibrations' : 'Manage Calibrations'}
        </button>
      </div>
      {showLatestCalibration && latestEta && (
        <div className="calibration-summary-panel">
          <h3>Latest Calibration Saved</h3>
          <p>Deviation parameters:</p>
          <ul>
            {Object.entries(latestEta).map(([key, value]) => (
              <li key={key}>{key}: {(value * 100).toFixed(2)}%</li>
            ))}
          </ul>
        </div>
      )}
      {selectedExercise && !showCalibrationManager && canonicalBaseline && (
        <div className="calibration-summary-panel">
          <h3>Baseline Overview</h3>
          <div className="baseline-overview">
            <div>
              <strong>Workout:</strong>
              {usingDefaultWorkout || !workoutRecord ? (
                <ul>
                  {Object.entries(canonicalBaseline).map(([key, value]) => (
                    <li key={`workout-default-${key}`}>{key}: {Number(value).toFixed(1)}°</li>
                  ))}
                </ul>
              ) : (
                <>
                  <p className="baseline-meta">Captured {new Date(workoutRecord.timestamp).toLocaleString()}</p>
                  <ul>
                    {Object.entries(workoutRecord.angles || {}).map(([key, value]) => (
                      <li key={`workout-${key}`}>{key}: {Number(value).toFixed(1)}°</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
            <div>
              <strong>Calibration:</strong>
              {usingDefaultCalibration || !calibrationRecord ? (
                <ul>
                  {Object.entries(canonicalBaseline).map(([key, value]) => (
                    <li key={`cal-default-${key}`}>{key}: {Number(value).toFixed(1)}°</li>
                  ))}
                </ul>
              ) : (
                <>
                  <p className="baseline-meta">Captured {new Date(calibrationRecord.timestamp).toLocaleString()}</p>
                  <ul>
                    {Object.entries(calibrationRecord.angles || {}).map(([key, value]) => (
                      <li key={`cal-${key}`}>{key}: {Number(value).toFixed(1)}°</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          </div>
        </div>
      )}
      {selectedExercise && showCalibrationManager && (
        <div className="calibration-records-panel">
          <h3>Calibration Records</h3>
          <p className="calibration-note">
            "Use for Workout" applies the saved angles during normal rep counting. "Use for Calibration" makes it the active reference when you enter calibration mode again.
          </p>
          <div className="critic-control">
            <div>
              <label>Critic (common): </label>
              <input
                type="number"
                step="0.01"
                value={criticInputs.common}
                onChange={(e) => handleCriticChange('common', e.target.value)}
              />
              <button onClick={() => handleCriticSubmit('common')}>Apply</button>
            </div>
          </div>
          
          {currentRecords.length === 0 && <p>No calibrations saved yet. Record a new one to personalize thresholds.</p>}
          {currentRecords.length > 0 && (
            <div className="calibration-records-list">
              {currentRecords.map(record => {
                const activeCommon = currentSummary.active?.common === record.id;
                const activeCal = currentSummary.active?.calibration === record.id;
                const isSelected = selectedRecordId === record.id;
                return (
                  <div
                    key={record.id}
                    className={`calibration-record-card ${isSelected ? 'selected' : ''} ${(activeCommon || activeCal) ? 'active' : ''}`}
                    onClick={() => {
                      selectedRecordIdRef.current = record.id;
                      setSelectedRecordId(record.id);
                    }}
                  >
                    <div className="calibration-record-header">
                      <span>{new Date(record.timestamp).toLocaleString()}</span>
                      <div className="badges">
                        {activeCommon && <span className="badge">Workout</span>}
                        {activeCal && <span className="badge calibration">Calibration</span>}
                      </div>
                    </div>
                    <div className="calibration-record-body">
                      <p><strong>Mode:</strong> {record.mode}</p>
                      <ul>
                        {record.eta && Object.entries(record.eta).map(([key, value]) => (
                          <li key={key}>{key}: {(value * 100).toFixed(1)}%</li>
                        ))}
                      </ul>
                    </div>
                    <div className="calibration-record-actions">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          // New command: use_workout
                          sendCommand({ command: 'use_workout', record_id: record.id, mode: 'common' });
                        }}
                      >
                        Use for Workout
                      </button>
                      <button
                        type="button"
                        className="danger-button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteCalibration(record.id);
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          {selectedRecord && (
            <div className="calibration-record-detail">
              <h4>Calibration Details</h4>
              <p><strong>Critic at capture:</strong> {selectedRecord.critic ?? 'N/A'}</p>
              <p><strong>Angles:</strong></p>
              <ul>
                {selectedRecord.angles && Object.entries(selectedRecord.angles).map(([key, value]) => (
                  <li key={key}>{key}: {Number(value).toFixed(1)}°</li>
                ))}
              </ul>
              <p><strong>Deviation:</strong></p>
              <ul>
                {selectedRecord.eta && Object.entries(selectedRecord.eta).map(([key, value]) => (
                  <li key={key}>{key}: {(value * 100).toFixed(2)}%</li>
                ))}
              </ul>
              <div className="calibration-images">
                {selectedRecord.images && Object.entries(selectedRecord.images).map(([key, img]) => (
                  img ? (
                    <div key={key} className="calibration-image">
                      <p>{key}</p>
                      <img src={`data:image/jpeg;base64,${img}`} alt={`${key} pose`} />
                    </div>
                  ) : null
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      {!isMediaPipeReady && <div>Loading MediaPipe libraries...</div>}

      {isMediaPipeReady && appState === 'selection' && (
        <div className="exercise-selection">
          <h2>Select an Exercise</h2>
          <p className="calibration-note">Calibration is optional—jump straight into a workout or capture personalized ranges first.</p>
          <div className="exercise-grid">
            {EXERCISES.map(exercise => (
              <div
                key={exercise.id}
                className={`exercise-card ${selectedExercise === exercise.id ? 'selected' : ''}`}
                onClick={() => handleSelectExercise(exercise.id)}
              >
                <h3>{exercise.label}</h3>
                <p>{exercise.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {isMediaPipeReady && appState === 'session_builder' && (
        <SessionBuilder
          sessionConfig={sessionConfig}
          sessionName={sessionName}
          setSessionName={setSessionName}
          exercises={EXERCISES}
          presets={SESSION_PRESETS}
          onAddSet={addSetToSession}
          onRemoveSet={removeSetFromSession}
          onLoadPreset={loadPreset}
          onStartSession={startSession}
          onCancel={() => setAppState('selection')}
        />
      )}

      {isMediaPipeReady && appState === 'calibration_countdown' && (
        <>
          <div className="calibration">
            <h2>Calibrate: {selectedExercise === 'bicep_curls' ? 'Bicep Curl' : 'Squat'}</h2>
            <p>Calibration starts in {countdown ?? 0} seconds. Get into your starting position.</p>
            <div className="video-container" style={{ position: 'relative', width: '640px', height: '480px', marginTop: '20px' }}>
              <video 
                ref={videoRef} 
                onCanPlay={startSendingFrames}
                autoPlay 
                playsInline 
                muted 
                className="camera-feed"
              ></video>
              <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
              {/* 3D Avatar disabled - avatar.glb missing */}
              <AROverlay 
                landmarks={poseLandmarks}
                feedbackLandmarks={feedbackLandmarks}
                arrowFeedback={arrowFeedback}
                selectedExercise={selectedExercise}
                backend={backendName}
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
          <div className="calibration-actions">
            <button type="button" onClick={cancelCalibration} className="secondary-button">
              Cancel Calibration
            </button>
          </div>
        </>
      )}

      {isMediaPipeReady && appState === 'calibrating_live' && (
        <div className="calibration">
          <h2>Calibrate: {selectedExercise === 'bicep_curls' ? 'Bicep Curl' : 'Squat'}</h2>
          <p>Perform a few full reps. We will detect your max and min angles automatically.</p>
          {calibrationProgress && (
            <div className="calibration-progress">
              <p><strong>Detected range so far:</strong></p>
              <ul>
                {selectedExercise === 'bicep_curls' ? (
                  <>
                    <li>Extended (max): {calibrationProgress.max_angle ? calibrationProgress.max_angle.toFixed(1) : '—'}°</li>
                    <li>Contracted (min): {calibrationProgress.min_angle ? calibrationProgress.min_angle.toFixed(1) : '—'}°</li>
                  </>
                ) : (
                  <>
                    <li>Standing (max): {calibrationProgress.max_angle ? calibrationProgress.max_angle.toFixed(1) : '—'}°</li>
                    <li>Deepest (min): {calibrationProgress.min_angle ? calibrationProgress.min_angle.toFixed(1) : '—'}°</li>
                  </>
                )}
              </ul>
              {calibrationProgress.frozen && (
                <p className="calibration-note">Range locked. You can walk back to the screen and click Finish without affecting the captured angles.</p>
              )}
            </div>
          )}
          <div className="video-container" style={{ position: 'relative', width: '640px', height: '480px', marginTop: '20px' }}>
            <video 
              ref={videoRef} 
              onCanPlay={startSendingFrames}
              autoPlay 
              playsInline 
              muted 
              className="camera-feed"
            ></video>
            <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
            {/* 3D Avatar disabled - avatar.glb missing */}
            <AROverlay
              landmarks={poseLandmarks}
              feedbackLandmarks={feedbackLandmarks}
              arrowFeedback={arrowFeedback}
              selectedExercise={selectedExercise}
              backend={backendName}
              currentAngles={{
                rightElbow: rightElbowAngle ? parseFloat(rightElbowAngle) : 0,
                rightKnee: rightKneeAngle ? parseFloat(rightKneeAngle) : 0,
              }}
              targetAngles={{
                rightElbow: 45,
                rightKnee: 90
              }}
            />
          </div>
          <div className="calibration-actions">
            <button
              type="button"
              onClick={finishCalibration}
              disabled={
                !calibrationProgress ||
              calibrationProgress.min_angle === null ||
              calibrationProgress.max_angle === null
            }
          >
            Finish Calibration
          </button>
          <button type="button" onClick={cancelCalibration} className="secondary-button">
            Cancel Calibration
          </button>
        </div>
      </div>
    )}
      {isMediaPipeReady && appState === 'summary' && (
        <LLMFeedback
          summary={llmSummary}
          chatHistory={chatHistory}
          isLoading={isLlmLoading}
          onAskQuestion={handleAskQuestion}
          onReset={resetApp}
          formAnalysis={formAnalysis}
        />
      )}

      {isMediaPipeReady && appState === 'calibration_saving' && (
        <div className="calibration">
          <h2>Finishing Calibration</h2>
          <p>Processing snapshots and saving your deviation parameters...</p>
        </div>
      )}

      {isMediaPipeReady && appState === 'workout' && (
        <div>
          <div className="workout-stats">
            {sessionProgress ? (
              <button onClick={endSession} className="reset-button">End Session</button>
            ) : (
              <button onClick={endWorkout} className="reset-button">End Workout</button>
            )}
            
            {/* Session Progress Display */}
            {sessionProgress && (
              <div className="session-progress-bar">
                <h3>{sessionProgress.session_name}</h3>
                <div className="session-set-info">
                  <span className="set-counter">
                    Set {sessionProgress.current_set_index + 1} / {sessionProgress.total_sets}
                  </span>
                  <span className="exercise-name">
                    {EXERCISES.find(e => e.id === sessionProgress.current_exercise)?.label || sessionProgress.current_exercise}
                  </span>
                </div>
                {sessionProgress.current_set && (
                  <div className="rep-progress">
                    <div className="rep-progress-bar">
                      <div 
                        className="rep-progress-fill"
                        style={{ 
                          width: `${Math.min(100, (sessionProgress.current_set.completed_reps / sessionProgress.current_set.target_reps) * 100)}%` 
                        }}
                      />
                    </div>
                    <span className="rep-count">
                      {sessionProgress.current_set.completed_reps} / {sessionProgress.current_set.target_reps} reps
                    </span>
                  </div>
                )}
                <div className="session-actions">
                  {sessionProgress.current_set?.is_complete && !sessionProgress.is_complete && (
                    <button onClick={nextSet} className="primary-button">Next Set →</button>
                  )}
                  {!sessionProgress.current_set?.is_complete && (
                    <button onClick={skipSet} className="secondary-button">Skip Set</button>
                  )}
                </div>
                {sessionProgress.is_complete && (
                  <div className="session-complete-banner">
                    🎉 Session Complete! Great workout!
                  </div>
                )}
              </div>
            )}
            
            <h2>REPS: {sessionProgress?.current_set?.completed_reps ?? repCounter}</h2>
            {/* Post-rep coaching command (aligned with form states) */}
            {postRepCommand && (
              <div className="post-rep-command">
                {postRepCommand}
              </div>
            )}
            <h2 className="feedback">{feedbackMessage}</h2>
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
              className="camera-feed"
            ></video>
            <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
            {/* 3D Avatar disabled - avatar.glb missing */}
            <AROverlay 
              landmarks={poseLandmarks}
              feedbackLandmarks={feedbackLandmarks}
              arrowFeedback={arrowFeedback}
              selectedExercise={selectedExercise}
              backend={backendName}
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

function LLMFeedback({ summary, chatHistory, isLoading, onAskQuestion, onReset, formAnalysis }) {
  const [question, setQuestion] = useState('');
  const chatContainerRef = useRef(null);

  useEffect(() => {
    // Scroll to the bottom of the chat window when new messages are added
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onAskQuestion(question);
    setQuestion('');
  };

  return (
    <div className="llm-feedback-container">
      {/* Form Analysis Section */}
      {formAnalysis && (
        <div className="form-analysis-section">
          <h2>Form Analysis</h2>
          <div className="form-score-card">
            <div className="score-circle" style={{
              background: `conic-gradient(${formAnalysis.score >= 70 ? '#22c55e' : formAnalysis.score >= 40 ? '#eab308' : '#ef4444'} ${formAnalysis.score * 3.6}deg, #333 0deg)`
            }}>
              <span className="score-value">{formAnalysis.score}%</span>
            </div>
            <div className="score-details">
              <p><strong>Good Reps:</strong> {formAnalysis.good_reps} / {formAnalysis.total_reps}</p>
            </div>
          </div>
          
          {formAnalysis.top_issues && formAnalysis.top_issues.length > 0 && (
            <div className="form-issues">
              <h3>Areas to Improve</h3>
              {formAnalysis.top_issues.map((issue, idx) => (
                <div key={idx} className="issue-card">
                  <div className="issue-header">
                    <span className="issue-name">{(issue.super_form_code || issue.state || 'unknown').replace(/_/g, ' ')}</span>
                    <span className="issue-count">{issue.count} reps</span>
                  </div>
                  <p className="issue-description">{issue.description}</p>
                </div>
              ))}
            </div>
          )}

          {/* Per-Rep Breakdown */}
          {formAnalysis.snapshots && formAnalysis.snapshots.length > 0 && (
            <div className="rep-breakdown">
              <h3>Rep-by-Rep Breakdown</h3>
              <div className="rep-list">
                {formAnalysis.snapshots.map((snapshot, idx) => {
                  const feedback = snapshot.feedback || {};
                  const isGood = feedback.is_good;
                  return (
                    <div key={idx} className={`rep-item ${isGood ? 'good' : 'needs-work'}`}>
                      <div className="rep-header">
                        <span className="rep-number">Rep {idx + 1}</span>
                        <span className={`rep-status ${isGood ? 'good' : 'bad'}`}>
                          {isGood ? '✓ Good Form' : '⚠ Needs Work'}
                        </span>
                      </div>
                      
                      {/* Summary */}
                      <p className={`rep-summary ${isGood ? 'good' : 'bad'}`}>
                        {feedback.summary}
                      </p>
                      
                      {/* Detailed feedback from primitives */}
                      {feedback.details && feedback.details.length > 0 && (
                        <div className="rep-details">
                          {feedback.details.map((detail, i) => (
                            <p key={i} className="rep-detail">• {detail}</p>
                          ))}
                        </div>
                      )}
                      
                      {/* Key highlights/focus areas */}
                      {feedback.highlights && feedback.highlights.length > 0 && (
                        <div className="rep-highlights">
                          <span className="highlight-label">Focus on:</span>
                          {feedback.highlights.map((highlight, i) => (
                            <span key={i} className="highlight-tag">{highlight}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      <h2>AI Coach Summary</h2>
      <div className="summary-card">
        {isLoading && !summary ? <p>Generating your summary...</p> : <p>{summary}</p>}
      </div>

      <h3>Ask the Coach</h3>
      <div className="chat-window">
        <div className="chat-history" ref={chatContainerRef}>
          {chatHistory.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.role}`}>
              <p><strong>{msg.role === 'user' ? 'You' : 'Coach'}:</strong> {msg.content}</p>
            </div>
          ))}
          {isLoading && chatHistory.length > 0 && (
            <div className="chat-message assistant"><p><i>Coach is typing...</i></p></div>
          )}
        </div>
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your form..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>Send</button>
        </form>
      </div>

      <button onClick={onReset} className="primary-button">Finish Review</button>
    </div>
  );
}

function SessionBuilder({ 
  sessionConfig, 
  sessionName, 
  setSessionName, 
  exercises, 
  presets, 
  onAddSet, 
  onRemoveSet, 
  onLoadPreset, 
  onStartSession, 
  onCancel 
}) {
  const [selectedExercise, setSelectedExercise] = useState(exercises[0]?.id || 'squats');
  const [reps, setReps] = useState(10);

  const handleAddSet = () => {
    onAddSet(selectedExercise, reps);
  };

  const totalReps = sessionConfig.reduce((sum, set) => sum + set.reps, 0);

  return (
    <div className="session-builder">
      <h2>Build Your Session</h2>
      
      {/* Session Name */}
      <div className="session-name-input">
        <label>Session Name:</label>
        <input
          type="text"
          value={sessionName}
          onChange={(e) => setSessionName(e.target.value)}
          placeholder="My Workout"
        />
      </div>

      {/* Presets */}
      <div className="session-presets">
        <h3>Quick Presets</h3>
        <div className="preset-buttons">
          {presets.map(preset => (
            <button
              key={preset.id}
              onClick={() => onLoadPreset(preset)}
              className="preset-button"
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {/* Add Set Form */}
      <div className="add-set-form">
        <h3>Add a Set</h3>
        <div className="add-set-controls">
          <select 
            value={selectedExercise} 
            onChange={(e) => setSelectedExercise(e.target.value)}
          >
            {exercises.map(ex => (
              <option key={ex.id} value={ex.id}>{ex.label}</option>
            ))}
          </select>
          <input
            type="number"
            min="1"
            max="100"
            value={reps}
            onChange={(e) => setReps(parseInt(e.target.value) || 10)}
            placeholder="Reps"
          />
          <button onClick={handleAddSet} className="add-button">+ Add Set</button>
        </div>
      </div>

      {/* Current Session Config */}
      <div className="session-config-list">
        <h3>Your Session ({sessionConfig.length} sets, {totalReps} total reps)</h3>
        {sessionConfig.length === 0 ? (
          <p className="empty-session">No sets added yet. Add sets above or choose a preset.</p>
        ) : (
          <div className="set-list">
            {sessionConfig.map((set, index) => (
              <div key={index} className="set-item">
                <span className="set-number">#{index + 1}</span>
                <span className="set-exercise">
                  {exercises.find(e => e.id === set.exercise)?.label || set.exercise}
                </span>
                <span className="set-reps">{set.reps} reps</span>
                <button 
                  onClick={() => onRemoveSet(index)} 
                  className="remove-button"
                  title="Remove set"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="session-builder-actions">
        <button 
          onClick={onStartSession} 
          className="primary-button"
          disabled={sessionConfig.length === 0}
        >
          Start Session
        </button>
        <button onClick={onCancel} className="secondary-button">
          Cancel
        </button>
      </div>
    </div>
  );
}

export default App;