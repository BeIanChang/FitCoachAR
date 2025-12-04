import base64
import json
import logging
import os
import time
from collections import deque

import cv2
import numpy as np
import uvloop
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

from coaches import LLMCoach, FormAnalyzer
from pose_backends import build_pose_backend, get_available_backends

# Install and use uvloop as the default event loop
uvloop.install()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POSE_BACKEND_NAME = os.getenv("POSE_BACKEND", "mediapipe_2d")

app = FastAPI()

WORKOUTS_DIR = "workouts"
if not os.path.exists(WORKOUTS_DIR):
    os.makedirs(WORKOUTS_DIR)

LLM_LOGS_DIR = "llm_logs"
if not os.path.exists(LLM_LOGS_DIR):
    os.makedirs(LLM_LOGS_DIR)

# --- LLM Coach Integration ---
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "csk-pe9ve2dc58528hxp34jwd4v8jk426t4mk9223my3j3k6ej5c")
llm_coach = LLMCoach(api_key=CEREBRAS_API_KEY)

# --- Form Analyzer ---
form_analyzer = FormAnalyzer()

class SessionData(BaseModel):
    total_reps: int
    success_rate: float
    mistakes: Dict[str, Any]
    avg_tempo: float
    exercise: str
    session_details: Optional[Dict[str, Any]] = None  # Full session data for context
    form_analysis: Optional[Dict[str, Any]] = None  # Detailed form analysis from primitives/states
    
    class Config:
        extra = "ignore"  # Ignore extra fields

class AskRequest(BaseModel):
    session_data: SessionData
    question: str
# ----------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to FitCoachAR - Real-Time Adaptive Exercise Coaching API",
        "pose_backend": POSE_BACKEND_NAME,
        "available_backends": get_available_backends(),
    }


@app.post("/summary")
async def get_summary(session_data: SessionData):
    # Log the exact data sent to LLM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = {
        "type": "session_summary",
        "timestamp": timestamp,
        "input_to_llm": session_data.dict(),
    }
    
    summary = llm_coach.generate_session_summary(session_data.dict())
    
    # Save complete log with response
    log_data["llm_response"] = summary
    log_filepath = os.path.join(LLM_LOGS_DIR, f"summary_{timestamp}.json")
    with open(log_filepath, "w") as f:
        json.dump(log_data, f, indent=4)
    logger.info(f"LLM summary log saved to {log_filepath}")
    
    return {"summary": summary}


@app.post("/ask")
async def ask_question(request: AskRequest):
    # Log the exact data sent to LLM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = {
        "type": "question_answer",
        "timestamp": timestamp,
        "input_to_llm": {
            "session_data": request.session_data.dict(),
            "question": request.question
        },
    }
    
    answer = llm_coach.answer_question(request.session_data.dict(), request.question)
    
    # Save complete log with response
    log_data["llm_response"] = answer
    log_filepath = os.path.join(LLM_LOGS_DIR, f"qa_{timestamp}.json")
    with open(log_filepath, "w") as f:
        json.dump(log_data, f, indent=4)
    logger.info(f"LLM Q&A log saved to {log_filepath}")
    
    return {"answer": answer}


@app.post("/save_workout")
async def save_workout(session_data: SessionData):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_data.exercise}_{timestamp}.json"
        filepath = os.path.join(WORKOUTS_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(session_data.dict(), f, indent=4)

        return {"status": "success", "message": f"Workout saved to {filepath}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/save_session")
async def save_session(session_data: Dict[str, Any]):
    """Save complete session data (multi-set workout) to a JSON file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = session_data.get("name", "session").replace(" ", "_")
        filename = f"session_{session_name}_{timestamp}.json"
        filepath = os.path.join(WORKOUTS_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=4)

        # Also save to LLM logs for analysis
        log_filepath = os.path.join(LLM_LOGS_DIR, f"session_{timestamp}.json")
        with open(log_filepath, "w") as f:
            json.dump(session_data, f, indent=4)

        logger.info(f"Session saved to {filepath}")
        return {"status": "success", "message": f"Session saved to {filepath}"}
    except Exception as e:
        logger.error(f"Failed to save session: {e}")
        return {"status": "error", "message": str(e)}


class FormAnalysisRequest(BaseModel):
    """Request model for form analysis with snapshots data."""
    exercise: str
    form_snapshots: list  # List of FormSnapshot dicts
    total_reps: int = 0


@app.post("/analyze_form")
async def analyze_form(request: FormAnalysisRequest):
    """
    Analyze form snapshots and return detailed form analysis.
    Works for both quick workouts and session-based workouts.
    """
    try:
        from coaches.super_form_codes_config import SUPER_FORM_CODES_CONFIG
        from coaches.feedback_generator import generate_rep_feedback
        
        exercise_super_codes = SUPER_FORM_CODES_CONFIG.get(request.exercise, {})

        super_form_codes_count = {}
        good_reps = 0
        processed_snapshots = []

        for snapshot in request.form_snapshots:
            super_codes = snapshot.get("form_states", [])
            static_codes = snapshot.get("static_primitives", {})
            dynamic_codes = snapshot.get("dynamic_primitives", {})
            
            # Aggregate counting
            for super_code in super_codes:
                super_form_codes_count[super_code] = super_form_codes_count.get(super_code, 0) + 1
            if "GOOD_REP" in super_codes:
                good_reps += 1

            # Generate detailed feedback using the feedback generator
            feedback = generate_rep_feedback(
                exercise=request.exercise,
                static_form_codes=static_codes,
                dynamic_form_codes=dynamic_codes,
                super_form_codes=super_codes
            )
            
            new_snapshot = snapshot.copy()
            new_snapshot["feedback"] = feedback
            processed_snapshots.append(new_snapshot)

        total_reps = request.total_reps or len(request.form_snapshots)
        
        # Get top issues for the summary view
        issues = {k: v for k, v in super_form_codes_count.items() if k != "GOOD_REP"}
        sorted_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)
        
        top_issues = []
        for super_code_name, count in sorted_issues[:3]:
            description = exercise_super_codes.get(super_code_name, {}).get("description", "")
            top_issues.append({
                "super_form_code": super_code_name,
                "count": count,
                "percentage": round((count / total_reps) * 100, 1) if total_reps > 0 else 0,
                "description": description
            })
        
        score = round((good_reps / total_reps) * 100, 1) if total_reps > 0 else 0
        
        analysis = {
            "exercise": request.exercise,
            "total_reps": total_reps,
            "good_reps": good_reps,
            "score": score,
            "super_form_codes_count": super_form_codes_count,
            "top_issues": top_issues,
            "snapshots": processed_snapshots
        }
        
        return {"status": "success", "analysis": analysis}
    except Exception as e:
        logger.error(f"Form analysis failed: {e}")
        return {"status": "error", "message": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info(
        "WebSocket connection attempt received (backend=%s).", POSE_BACKEND_NAME
    )
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    backend = build_pose_backend(POSE_BACKEND_NAME)
    frame_times = deque(maxlen=60)
    last_payload = None

    try:
        while True:
            data = await websocket.receive_text()

            if data.startswith('{"command"'):
                command_data = json.loads(data)
                response = backend.handle_command(command_data)
                if response:
                    await websocket.send_json(response)
                continue

            frame_timestamp = None
            if data.startswith("{"):
                try:
                    parsed_payload = json.loads(data)
                except json.JSONDecodeError:
                    parsed_payload = None
                if isinstance(parsed_payload, dict) and "frame" in parsed_payload:
                    frame_timestamp = parsed_payload.get("ts")
                    data = parsed_payload["frame"]

            start_time = time.time()

            if frame_times:
                avg_fps = len(frame_times) / sum(frame_times)
                if avg_fps < 20 and last_payload:
                    payload = dict(last_payload)
                    if frame_timestamp is not None:
                        payload["client_ts"] = frame_timestamp
                    await websocket.send_json(payload)
                    end_time = time.time()
                    frame_times.append(end_time - start_time)
                    continue

            if not data.startswith("data:image/jpeg;base64,"):
                logger.warning("Received malformed data packet")
                continue

            try:
                _, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as decode_error:
                logger.warning("Failed to decode frame: %s", decode_error)
                continue

            if frame is None:
                logger.warning("Decoded frame is None")
                continue

            try:
                backend_name = getattr(backend, "name", POSE_BACKEND_NAME)
                process_start = time.perf_counter()
                result = backend.process_frame(frame)
                latency_ms = (time.perf_counter() - process_start) * 1000
                payload_to_send = None
                if result:
                    result.setdefault("backend", backend_name)
                    payload_to_send = result
                    last_payload = result
                elif last_payload:
                    payload_to_send = dict(last_payload)
                else:
                    payload_to_send = {"landmarks": [], "backend": backend_name}

                payload_to_send["latency_ms"] = latency_ms
                if frame_timestamp is not None:
                    payload_to_send["client_ts"] = frame_timestamp
                payload_to_send.setdefault("backend", backend_name)
                await websocket.send_json(payload_to_send)
            except Exception as processing_error:
                logger.error("Error processing frame: %s", processing_error)

            end_time = time.time()
            frame_times.append(end_time - start_time)

    except Exception as websocket_error:
        logger.error("WebSocket connection error: %s", websocket_error)
    finally:
        backend.close()
        logger.info("Client connection closed")
