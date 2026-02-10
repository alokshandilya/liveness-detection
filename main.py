import asyncio
import json
import logging
import shutil
import time
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import httpx
from logic.liveness import check_liveness
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Service URLs
SERVICE_B_URL = os.getenv("EFFORT_MODEL_URL", "http://127.0.0.1:8001/detect")
SERVICE_C_URL = os.getenv("SYNCNET_MODEL_URL", "http://127.0.0.1:8002/check-sync")

# Data Models
class ChunkRequest(BaseModel):
    file_path: str

class SpawnRequest(BaseModel):
    url: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Avoid modifying the list while iterating
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                chunk_to_remove = connection
                # We might want to remove strict disconnect here or let the handler handle it
                # For now just log error.

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get():
    with open(os.path.join(os.path.dirname(__file__), "monitor.html"), "r") as f:
        return f.read()

@app.get("/logo.svg")
async def get_logo():
    return FileResponse(os.path.join(os.path.dirname(__file__), "logo.svg"), media_type="image/svg+xml")

@app.post("/spawn")
async def spawn_bot_endpoint(request: SpawnRequest):
    logger.info(f"Spawning bot for: {request.url}")
    try:
        # Run spawn_bot.py as a subprocess to keep environment/logic isolated
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "spawn_bot.py", "--url", request.url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        output = stdout.decode()
        if proc.returncode == 0:
            # Simple parse for Bot ID if present in output
            bot_id = "Unknown"
            for line in output.split('\n'):
                if "Bot ID:" in line:
                    bot_id = line.split("Bot ID:")[1].strip()
            return {"status": "ok", "bot_id": bot_id, "output": output}
        else:
            error_msg = stderr.decode() or output
            logger.error(f"Spawn failed: {error_msg}")
            return HTMLResponse(status_code=500, content=json.dumps({"detail": error_msg}))
            
    except Exception as e:
        logger.error(f"Spawn exception: {e}")
        return HTMLResponse(status_code=500, content=json.dumps({"detail": str(e)}))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe wait for a ping or just sleep
            # We are pushing data, so we don't necessarily expect input
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


async def call_service_b(file_path: str, client: httpx.AsyncClient):
    """Call Service B: Effort Model"""
    try:
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"file": (file_name, f, "video/mp4")}
            response = await client.post(SERVICE_B_URL, files=files, timeout=60.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Service B failed: {repr(e)}")
        # Return fallback or inconclusive
        return {"is_fake": False, "fake_probability": 0.0, "error": str(e)}

async def call_service_c(file_path: str, client: httpx.AsyncClient):
    """Call Service C: SyncNet"""
    try:
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
             files = {"file": (file_name, f, "video/mp4")}
             response = await client.post(SERVICE_C_URL, files=files, timeout=200.0)
             response.raise_for_status()
             return response.json()
    except Exception as e:
        logger.error(f"Service C failed: {repr(e)}")
        # Prompt says: mark audio as "Inconclusive" but don't crash
        # We return a structure that won't trigger the FAKE verdict clause (Sync Good == False)
        # So we set is_sync_good to True (Pass) or None and handle it in aggregation
        return {"status": "error", "is_sync_good": True, "average_distance": -1, "note": "Inconclusive"}

async def run_analysis_pipeline(file_path: str):
    logger.info(f"Processing video: {file_path}")

    async with httpx.AsyncClient() as client:
        # 1. Parallel External Calls
        task_b = call_service_b(file_path, client)
        task_c = call_service_c(file_path, client)
        
        # 2. Local Execution (Liveness) in Thread
        # We use asyncio.to_thread for blocking CPU task
        task_liveness = asyncio.to_thread(check_liveness, file_path)

        # 3. Aggregation (Wait for all)
        results = await asyncio.gather(task_b, task_c, task_liveness)
        
        service_b_res, service_c_res, liveness_res = results

    # 4. Verdict Logic
    # FAKE if (Visual Prob > 0.6) OR (Sync Good == False) OR (Liveness Fail == True).
    
    visual_prob = service_b_res.get("fake_probability", 0.0)
    sync_dist = service_c_res.get("average_distance", 100.0)
    is_sync_good = service_c_res.get("is_sync_good", True) # Default to True if missing
    is_liveness_fail = liveness_res.get("is_liveness_fail", False)
    
    verdict = "REAL"
    reasons = []

    if visual_prob > 0.6:
        verdict = "FAKE"
        reasons.append(f"Visual Probability High ({visual_prob})")

    # Sync Logic: Ignore failure if visual is low (< 20%) OR (visual < 50% AND sync dist <= 9.0)
    if not is_sync_good:
        should_ignore = (visual_prob < 0.2) or (visual_prob < 0.5 and sync_dist <= 9.0)
        if not should_ignore:
            verdict = "FAKE"
            reasons.append("Audio Sync Failed")

    if is_liveness_fail:
        verdict = "FAKE" 
        reasons.append("Liveness Check Failed")

    final_response = {
        "chunk_path": file_path,
        "verdict": verdict,
        "details": {
            "visual_check": service_b_res,
            "audio_check": service_c_res,
            "liveness_check": liveness_res
        },
        "reasons": reasons
    }
    return final_response

@app.post("/process_chunk")
async def process_chunk(request: ChunkRequest):
    file_path = request.file_path
    
    if not os.path.exists(file_path):
        return {"status": "error", "message": "File not found"}

    final_response = await run_analysis_pipeline(file_path)

    # 5. Broadcast
    await manager.broadcast(final_response)

    # 6. Return
    return final_response

@app.post("/analyze-video")
async def analyze_video(file: UploadFile):
    """
    Analyzes an uploaded video file using the liveness detection pipeline.
    Returns the analysis results including Visual, Audio, and Liveness checks.
    """
    # Save uploaded file
    temp_dir = "/tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_filename = f"upload_{int(time.time())}_{file.filename}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        final_response = await run_analysis_pipeline(temp_path)
        return final_response
        
    finally:
        # Cleanup uploaded file after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)
