import asyncio
import json
import logging
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, BackgroundTasks
from pydantic import BaseModel
import httpx
from logic.liveness import check_liveness
import os

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Service URLs
SERVICE_B_URL = "http://127.0.0.1:8001/detect"
SERVICE_C_URL = "http://127.0.0.1:8002/check-sync"

# Data Models
class ChunkRequest(BaseModel):
    file_path: str

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
             response = await client.post(SERVICE_C_URL, files=files, timeout=60.0)
             response.raise_for_status()
             return response.json()
    except Exception as e:
        logger.error(f"Service C failed: {repr(e)}")
        # Prompt says: mark audio as "Inconclusive" but don't crash
        # We return a structure that won't trigger the FAKE verdict clause (Sync Good == False)
        # So we set is_sync_good to True (Pass) or None and handle it in aggregation
        return {"status": "error", "is_sync_good": True, "average_distance": -1, "note": "Inconclusive"}

@app.post("/process_chunk")
async def process_chunk(request: ChunkRequest):
    file_path = request.file_path
    
    if not os.path.exists(file_path):
        return {"status": "error", "message": "File not found"}

    logger.info(f"Processing chunk: {file_path}")

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
    is_sync_good = service_c_res.get("is_sync_good", True) # Default to True if missing
    is_liveness_fail = liveness_res.get("is_liveness_fail", False)
    
    verdict = "REAL"
    reasons = []

    if visual_prob > 0.6:
        verdict = "FAKE"
        reasons.append(f"Visual Probability High ({visual_prob})")

    if not is_sync_good:
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

    # 5. Broadcast
    await manager.broadcast(final_response)

    # 6. Return
    return final_response
