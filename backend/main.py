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
from typing import Dict
from datetime import datetime

from llm_feedback import LLMFeedbackGenerator
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

# --- LLM Integration ---
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "csk-pe9ve2dc58528hxp34jwd4v8jk426t4mk9223my3j3k6ej5c")
llm_generator = LLMFeedbackGenerator(use_api=True, api_key=CEREBRAS_API_KEY)

class SessionData(BaseModel):
    total_reps: int
    success_rate: float
    mistakes: Dict[str, int]
    avg_tempo: float
    exercise: str

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
    summary = llm_generator.generate_session_summary(session_data.dict())
    return {"summary": summary}


@app.post("/ask")
async def ask_question(request: AskRequest):
    answer = llm_generator.answer_question(request.session_data.dict(), request.question)
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
