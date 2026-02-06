import asyncio
import os
import shutil
import time
import json
import base64
from typing import List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import uvicorn

# Configuration
CHUNK_DIR = os.path.join(os.getcwd(), "chunks")
CHUNK_DURATION = 10.0  # Seconds
FFMPEG_CMD = "ffmpeg"  # Ensure ffmpeg is in your PATH

# Ensure the chunks directory exists
os.makedirs(CHUNK_DIR, exist_ok=True)

app = FastAPI()

# In-memory buffers
# Structure: {'data': bytes, 'time': float}
video_buffer: List[Dict] = []
audio_buffer: List[Dict] = []

# Locks to ensure thread-safe access to buffers
buffer_lock = asyncio.Lock()

def process_recall_message(message_text: str, buffer_list: list, prefix: str):
    """
    Parses a JSON message from Recall.ai, extracts the base64 payload,
    decodes it, and appends it to the buffer.
    """
    try:
        data = json.loads(message_text)
        
        # We need to robustly find the payload. 
        # Recall separate streams can be deeply nested:
        # data -> data -> buffer (Base64)
        # Or sometimes top level: data -> Base64
        # Or payload -> Base64
        
        payload = None
        
        # 1. Try deep nest (from user snippets/docs for separate streams)
        try:
            payload = data["data"]["data"]["buffer"]
        except (KeyError, TypeError):
            pass

        # 2. Try top-level 'data' key with base64 string (common fallback)
        if not payload and "data" in data and isinstance(data["data"], str):
            payload = data["data"]
            
        # 3. Try 'payload' key (another common pattern)
        if not payload and "payload" in data and isinstance(data["payload"], str):
            payload = data["payload"]
        
        if payload:
            decoded_bytes = base64.b64decode(payload)
            if decoded_bytes:
                buffer_list.append({"data": decoded_bytes, "time": time.time()})
                # print(f"[{prefix}] Decoded {len(decoded_bytes)} bytes")
            else:
                print(f"[{prefix}] Warning: Empty decoded payload")
    except json.JSONDecodeError:
        pass
    except Exception as e:
        # Don't spam logs if it's just a non-media event
        pass
        # print(f"[{prefix}] Error processing text frame: {e}")

@app.websocket("/recall-video-endpoint")
async def video_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Video stream connected")
    try:
        while True:
            if websocket.client_state == WebSocketState.DISCONNECTED:
                break
            
            message = await websocket.receive()

            # Case 1: Binary Frame (Raw bytes)
            if "bytes" in message and message["bytes"]:
                async with buffer_lock:
                    video_buffer.append({"data": message["bytes"], "time": time.time()})
            
            # Case 2: Text Frame (JSON with Base64)
            elif "text" in message:
                async with buffer_lock:
                    process_recall_message(message["text"], video_buffer, "Video")

    except WebSocketDisconnect:
        print("Video stream disconnected cleanly")
    except RuntimeError as e:
        if "disconnect message has been received" not in str(e):
             print(f"Video stream runtime error: {e}")
    except Exception as e:
        print(f"Video stream error: {e}")

@app.websocket("/recall-audio-endpoint")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Audio stream connected")
    try:
        while True:
            if websocket.client_state == WebSocketState.DISCONNECTED:
                break

            message = await websocket.receive()

            # Case 1: Binary Frame (Raw bytes)
            if "bytes" in message and message["bytes"]:
                async with buffer_lock:
                    audio_buffer.append({"data": message["bytes"], "time": time.time()})
            
            # Case 2: Text Frame (JSON with Base64)
            elif "text" in message:
                async with buffer_lock:
                    process_recall_message(message["text"], audio_buffer, "Audio")
                
    except WebSocketDisconnect:
        print("Audio stream disconnected cleanly")
    except RuntimeError as e:
        if "disconnect message has been received" not in str(e):
             print(f"Audio stream runtime error: {e}")
    except Exception as e:
        print(f"Audio stream error: {e}")

async def process_buffers():
    """
    Periodically flushes buffers to temporary files and muxes them 
    into an MP4 file using FFmpeg, then moves it to the chunks directory.
    """
    chunk_counter = 0
    consecutive_chunks = 0
    
    while True:
        if consecutive_chunks >= 10:
            print("Pausing for 10 seconds after 10 chunks...")
            await asyncio.sleep(10)
            async with buffer_lock:
                video_buffer.clear()
                audio_buffer.clear()
            consecutive_chunks = 0

        await asyncio.sleep(CHUNK_DURATION)
        
        # 1. Extract current data from buffers safely
        current_video = []
        current_audio = []
        
        async with buffer_lock:
            # Swap references to clear main buffers quickly
            if video_buffer:
                current_video = video_buffer[:]
                video_buffer.clear()
            
            if audio_buffer:
                current_audio = audio_buffer[:]
                audio_buffer.clear()
        
        # If we have no video, we skip (SyncNet requires video)
        if not current_video:
            continue

        print(f"Processing chunk {chunk_counter}: {len(current_video)} video packets, {len(current_audio)} audio packets")

        # 2. Create unique temporary filenames
        timestamp = int(time.time() * 1000)
        base_name = f"chunk_{timestamp}_{chunk_counter}"
        temp_video_path = f"/tmp/{base_name}.h264"
        temp_audio_path = f"/tmp/{base_name}.pcm"
        ffmpeg_log_path = f"/tmp/{base_name}.ffmpeg.log"
        output_filename = f"live_stream_{timestamp}.mp4"
        temp_output_path = f"/tmp/{output_filename}"
        final_output_path = os.path.join(CHUNK_DIR, output_filename)
        
        try:
            # 3. Write raw data to temp files
            with open(temp_video_path, "wb") as f:
                for packet in current_video:
                    f.write(packet["data"])
            
            has_audio = False
            if current_audio:
                has_audio = True
                with open(temp_audio_path, "wb") as f:
                    for packet in current_audio:
                        f.write(packet["data"])

            # 4. Mux using FFmpeg
            cmd = [FFMPEG_CMD, "-y"]
            
            # Input Video
            cmd.extend(["-f", "h264", "-i", temp_video_path])
            
            # Input Audio (if available)
            if has_audio:
                # Assuming PCM S16LE, 16000 Hz, Mono based on standard Recall output
                cmd.extend(["-f", "s16le", "-ar", "16000", "-ac", "1", "-i", temp_audio_path])
            
            # Output Options
            cmd.extend(["-c:v", "copy"]) # Copy video stream (fast, no re-encode)
            
            if has_audio:
                cmd.extend(["-c:a", "aac"])  # Encode audio to AAC
            
            cmd.extend(["-movflags", "faststart"]) # Optimize for web playback
            cmd.append(temp_output_path)
            
            # Run FFmpeg
            with open(ffmpeg_log_path, "w") as log_file:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_file,
                    stderr=log_file
                )
                await process.wait()
            
            if process.returncode == 0:
                shutil.move(temp_output_path, final_output_path)
                print(f"Created chunk: {output_filename}")
                consecutive_chunks += 1
                
                # Cleanup log on success
                if os.path.exists(ffmpeg_log_path):
                    os.remove(ffmpeg_log_path)
                
                # Cleanup temps on success
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
            else:
                print(f"FFmpeg failed to create chunk {chunk_counter} (Return code: {process.returncode})")
                print(f"See log at: {ffmpeg_log_path}")

        except Exception as e:
            print(f"Error processing chunk {chunk_counter}: {e}")
        
        finally:
            chunk_counter += 1

@app.on_event("startup")
async def startup_event():
    # Start the background task
    asyncio.create_task(process_buffers())

if __name__ == "__main__":
    # Run on port 5000 as requested
    print(f"Starting Bridge Server on port 5000...")
    print(f"Writing chunks to: {CHUNK_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=5000)