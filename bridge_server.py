import asyncio
import os
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
CHUNK_DURATION = 10.0  # Seconds
FFMPEG_CMD = "ffmpeg"  # Ensure ffmpeg is in your PATH

# Ensure the chunks directory exists
os.makedirs(CHUNK_DIR, exist_ok=True)

app = FastAPI()

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
    print("📹 Video stream connected")
    
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
                    print(f"🔑 Captured Video Header ({len(payload)} bytes)")
                    video_header = payload
                
                async with buffer_lock:
                    video_buffer.append({"data": payload, "time": time.time()})
                
                packet_count += 1
                if packet_count % 30 == 0:
                    pass 
                    # print(f"   [Video] Buffered 30 packets")

    except WebSocketDisconnect:
        print("📹 Video stream disconnected")
    except Exception as e:
        print(f"❌ Video Error: {e}")

@app.websocket("/recall-audio-endpoint")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🎙️ Audio stream connected")
    
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
                    pass
                    # print(f"   [Audio] Buffered 50 packets")
                
    except WebSocketDisconnect:
        print("🎙️ Audio stream disconnected")
    except Exception as e:
        print(f"❌ Audio Error: {e}")

async def process_buffers():
    """
    Periodically flushes buffers to temporary files and muxes them 
    into an MP4 file using FFmpeg, then moves it to the chunks directory.
    """
    chunk_counter = 0
    print(f"⏱️  Background Processor Started (Every {CHUNK_DURATION}s)")
    
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
            # print("   (No video data in this interval, skipping chunk)")
            continue

        # --- DYNAMIC FPS CALCULATION ---
        # If Recall sends sparse frames (e.g. 70 frames in 10s -> 7fps),
        # FFmpeg default 25fps interprets 70 frames as 2.8s video.
        # We calculate the effective FPS to tell FFmpeg how to stretch the input.
        num_packets = len(current_video)
        input_fps = max(num_packets / CHUNK_DURATION, 1.0) # Ensure at least 1.0
        
        # Round to 2 decimals for cleaner logs/cmd
        input_fps = round(input_fps, 2)

        print(f"📦 Processing chunk {chunk_counter}: {num_packets} frames ({input_fps} fps), {len(current_audio)} audio samples")

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
                # INJECT HEADER: Ensure every chunk starts with SPS/PPS
                if video_header:
                    f.write(video_header)
                
                # Write the actual frames for this chunk
                for packet in current_video:
                    f.write(packet["data"])
            
            has_audio = False
            if current_audio:
                has_audio = True
                with open(temp_audio_path, "wb") as f:
                    for packet in current_audio:
                        f.write(packet["data"])

            # 4. Mux using FFmpeg
            # We map inputs and force output to a clean MP4
            cmd = [
                FFMPEG_CMD, "-y",
                
                # Input Video
                "-r", str(input_fps), # Interpret input at calculated FPS to fill 10s
                "-f", "h264", 
                "-i", temp_video_path,
                
                # Input Audio (if available)
                *(["-f", "s16le", "-ar", "44100", "-ac", "1", "-i", temp_audio_path] if has_audio else []),
                
                # Output Video Options
                # We RE-ENCODE to Libx264 to normalize to 30fps output
                # This duplicates frames if input_fps < 30, ensuring valid 10s video.
                "-c:v", "libx264",
                "-r", "30",           # Output Framerate
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", # Ensure compatible pixel format
                
                # Output Audio Options
                *(["-c:a", "aac", "-b:a", "128k"] if has_audio else []),
                
                # Optimization
                "-movflags", "+faststart",
                
                # Output Path
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
            
            if process.returncode == 0:
                shutil.move(temp_output_path, final_output_path)
                print(f"✅ Created chunk: {output_filename}")
                
                # Cleanup logs/temps on success
                for p in [ffmpeg_log_path, temp_video_path, temp_audio_path]:
                    if os.path.exists(p):
                        os.remove(p)
            else:
                print(f"❌ FFmpeg failed to create chunk {chunk_counter} (Return code: {process.returncode})")
                print(f"   See log at: {ffmpeg_log_path}")
                # Keep temp files for debugging

        except Exception as e:
            print(f"❌ Error processing chunk {chunk_counter}: {e}")
        
        finally:
            chunk_counter += 1

@app.on_event("startup")
async def startup_event():
    # Start the background task
    asyncio.create_task(process_buffers())

if __name__ == "__main__":
    print(f"Starting Bridge Server on port 5000...")
    print(f"Chunks will be written to: {CHUNK_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=5000)