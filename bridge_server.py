import asyncio
import os
import sys
import time
import json
import base64
import subprocess
import shutil
from typing import List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# --- Configuration ---
CHUNK_DIR = os.path.join(os.getcwd(), "chunks")
CHUNK_DURATION = 15.0  # Seconds
FFMPEG_CMD = "ffmpeg"  # Ensure ffmpeg is in your PATH

# Ensure the chunks directory exists
os.makedirs(CHUNK_DIR, exist_ok=True)

app = FastAPI()

# Force unbuffered output for logging
sys.stdout.reconfigure(line_buffering=True)

# In-memory buffers
# Structure: {'data': bytes, 'time': float}
video_buffer: List[Dict] = []
audio_buffer: List[Dict] = []

# Store the critical H.264 Stream Header (SPS/PPS)
# We will prepend this to every chunk to ensure FFmpeg can decode it.
video_header: bytes = None

# Locks to ensure thread-safe access to buffers
buffer_lock = asyncio.Lock()

def extract_payload(message_text):
    """Robustly extracts base64 payload from JSON message."""
    try:
        data = json.loads(message_text)
        # 1. Try deep nested
        try:
            return base64.b64decode(data["data"]["data"]["buffer"])
        except (KeyError, TypeError):
            pass
        # 2. Try top-level 'data'
        if "data" in data and isinstance(data["data"], str):
            return base64.b64decode(data["data"])
        # 3. Try 'payload'
        if "payload" in data and isinstance(data["payload"], str):
            return base64.b64decode(data["payload"])
    except Exception:
        pass
    return None

@app.websocket("/recall-video-endpoint")
async def video_endpoint(websocket: WebSocket):
    global video_header
    await websocket.accept()
    print("📹 Video stream connected", flush=True)
    
    packet_count = 0
    try:
        while True:
            message = await websocket.receive()
            payload = None

            if "bytes" in message:
                payload = message["bytes"]
            elif "text" in message:
                payload = extract_payload(message["text"])
            
            if payload:
                # Save the first packet as the header (SPS/PPS usually)
                if video_header is None:
                    print(f"🔑 Captured Video Header ({len(payload)} bytes)", flush=True)
                    video_header = payload
                
                async with buffer_lock:
                    video_buffer.append({"data": payload, "time": time.time()})
                
                packet_count += 1
                if packet_count % 30 == 0:
                    print(f"📹 Video packets: {packet_count}", flush=True)

    except WebSocketDisconnect:
        print("📹 Video stream disconnected", flush=True)
    except Exception as e:
        print(f"❌ Video Error: {e}", flush=True)

@app.websocket("/recall-audio-endpoint")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🎙️ Audio stream connected", flush=True)
    
    packet_count = 0
    try:
        while True:
            message = await websocket.receive()
            payload = None

            if "bytes" in message:
                payload = message["bytes"]
            elif "text" in message:
                payload = extract_payload(message["text"])
            
            if payload:
                async with buffer_lock:
                    audio_buffer.append({"data": payload, "time": time.time()})
                
                packet_count += 1
                if packet_count % 50 == 0:
                    print(f"🎙️ Audio packets: {packet_count}", flush=True)
                
    except WebSocketDisconnect:
        print("🎙️ Audio stream disconnected", flush=True)
    except Exception as e:
        print(f"❌ Audio Error: {e}", flush=True)

async def process_buffers():
    """
    Periodically flushes buffers to temporary files and muxes them 
    into an MP4 file using FFmpeg, then moves it to the chunks directory.
    """
    chunk_counter = 0
    print(f"⏱️  Background Processor Started (Every {CHUNK_DURATION}s)", flush=True)
    
    while True:
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

        # --- VALIDATION 1: Duration Check ---
        # If the chunk represents < 15s of time, discard it (start/end partials).
        t_start = current_video[0]['time']
        t_end = current_video[-1]['time']
        duration = t_end - t_start
        
        if duration < 15.0:
            print(f"⚠️  Skipping Chunk {chunk_counter}: Duration too short ({duration:.2f}s < 15s)", flush=True)
            continue

        # --- DYNAMIC FPS CALCULATION ---
        num_packets = len(current_video)
        input_fps = 30.0 
        
        if duration > 0.1:
            input_fps = num_packets / duration
        
        input_fps = max(min(input_fps, 60.0), 1.0)
        input_fps = round(input_fps, 2)

        print(f"📦 Processing chunk {chunk_counter}: {num_packets} frames over {duration:.2f}s ({input_fps} fps)", flush=True)

        # 2. Create filenames
        timestamp = int(time.time() * 1000)
        base_name = f"chunk_{timestamp}_{chunk_counter}"
        temp_video_path = f"/tmp/{base_name}.h264"
        temp_audio_path = f"/tmp/{base_name}.pcm"
        ffmpeg_log_path = f"/tmp/{base_name}.ffmpeg.log"
        output_filename = f"live_stream_{timestamp}.mp4"
        
        # ATOMIC WRITE FIX:
        # Write to chunks/.temp_name.mp4 (hidden from watcher)
        # Then rename to chunks/name.mp4 (atomic)
        temp_output_filename = f".temp_{output_filename}"
        temp_output_path = os.path.join(CHUNK_DIR, temp_output_filename)
        final_output_path = os.path.join(CHUNK_DIR, output_filename)
        
        try:
            # 3. Write raw data to temp files
            with open(temp_video_path, "wb") as f:
                if video_header:
                    f.write(video_header)
                for packet in current_video:
                    f.write(packet["data"])
            
            has_audio = False
            if current_audio:
                has_audio = True
                with open(temp_audio_path, "wb") as f:
                    for packet in current_audio:
                        f.write(packet["data"])

            # 4. Mux using FFmpeg
            cmd = [
                FFMPEG_CMD, "-y",
                
                # Input Video
                "-r", str(input_fps), 
                "-f", "h264", 
                "-i", temp_video_path,
                
                # Input Audio (if available)
                *(["-f", "s16le", "-ar", "16000", "-ac", "1", "-i", temp_audio_path] if has_audio else []),
                
                # Output Video Options
                "-c:v", "libx264",
                "-r", "30",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                
                # Output Audio Options
                *(["-c:a", "aac", "-b:a", "128k"] if has_audio else []),
                
                # Optimization
                "-movflags", "+faststart",
                
                # Trim: Skip first 3s, keep next 14s
                "-ss", "3",
                "-t", "14",

                # Output Path (Hidden Temp)
                temp_output_path
            ]
            
            # Run FFmpeg
            with open(ffmpeg_log_path, "w") as log_file:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_file,
                    stderr=log_file
                )
                await process.wait()
            
            # --- VALIDATION 2: Size & Success Check ---
            if process.returncode == 0:
                if os.path.exists(temp_output_path):
                    size = os.path.getsize(temp_output_path)
                    if size > 50 * 1024: # > 50KB (Valid Video)
                        shutil.move(temp_output_path, final_output_path)
                        print(f"✅ Created chunk: {output_filename} ({size/1024:.1f} KB)", flush=True)
                    else:
                        print(f"⚠️  Discarding Chunk: File too small ({size} bytes)", flush=True)
                        os.remove(temp_output_path)
                else:
                     print("❌ FFmpeg Output File Missing", flush=True)
                
                # Cleanup logs/temps on success
                for p in [ffmpeg_log_path, temp_video_path, temp_audio_path]:
                    if os.path.exists(p):
                        os.remove(p)
            else:
                print(f"❌ FFmpeg failed (RC: {process.returncode}). See log: {ffmpeg_log_path}", flush=True)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

        except Exception as e:
            print(f"❌ Error processing chunk {chunk_counter}: {e}", flush=True)
        
        finally:
            chunk_counter += 1

@app.on_event("startup")
async def startup_event():
    # Start the background task
    asyncio.create_task(process_buffers())

if __name__ == "__main__":
    print(f"Starting Bridge Server on port 5000...", flush=True)
    print(f"Chunks will be written to: {CHUNK_DIR}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=5000)