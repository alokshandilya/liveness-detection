import subprocess
import time
import signal
import sys
import os
from threading import Thread
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MEETING_URL = os.getenv("MEETING_URL", "https://meet.google.com/wga-srzk-hyj")
NGROK_FULL_URL = os.getenv("NGROK_URL", "")

processes = []

def stream_logs(process, prefix):
    """Reads stdout from a subprocess and prints it with a prefix."""
    try:
        for line in iter(process.stdout.readline, ''):
            print(f"[{prefix}] {line.strip()}")
    except ValueError:
        pass
    finally:
        process.stdout.close()

def start_process(command, prefix):
    """Starts a subprocess and spawns a thread to stream its logs."""
    print(f"🚀 Starting {prefix}...")
    # setsid creates a new session, allowing us to kill the whole group later
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid 
    )
    processes.append(process)
    
    # Start a thread to print logs
    t = Thread(target=stream_logs, args=(process, prefix))
    t.daemon = True
    t.start()
    return process

def cleanup(signum, frame):
    """Gracefully stops all subprocesses on exit."""
    print("\n🛑 Stopping all services...")
    for p in processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    sys.exit(0)

# Register signal handlers for Ctrl+C
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def extract_domain(url):
    """Extracts the domain from a full URL for the ngrok command."""
    # Remove protocol
    clean = url.replace("https://", "").replace("http://", "")
    # Remove path
    clean = clean.split("/")[0]
    return clean

def main():
    # 1. Start Main API (Port 8000)
    # Serves monitor.html and handles chunk processing
    start_process("uvicorn main:app --host 0.0.0.0 --port 8000", "API")

    # 2. Start Bridge Server (Port 5000)
    # Receives streams from Recall.ai
    start_process("python bridge_server.py", "BRIDGE")

    # 3. Start Watcher
    # Watches /chunks folder
    start_process("python watcher.py", "WATCHER")

    # 4. Start Ngrok (Port 5000)
    # We attempt to use the domain from .env if it looks like a custom ngrok domain
    ngrok_cmd = "ngrok http 5000"
    if "ngrok-free.dev" in NGROK_FULL_URL or "ngrok.io" in NGROK_FULL_URL:
        domain = extract_domain(NGROK_FULL_URL)
        print(f"ℹ️  Using Ngrok Domain: {domain}")
        ngrok_cmd = f"ngrok http --domain={domain} 5000"
    
    start_process(ngrok_cmd, "NGROK")

    print("⏳ Waiting 5 seconds for services to initialize...")
    time.sleep(5)

    # 5. Spawn Bot
    # This runs once to initiate the bot connection
    print(f"🤖 Spawning Bot for {MEETING_URL}...")
    try:
        # spawn_bot.py now reads API_KEY, NGROK, REGION from .env
        subprocess.run(
            [sys.executable, "spawn_bot.py", "--url", MEETING_URL],
            check=True
        )
    except subprocess.CalledProcessError:
        print("❌ Failed to spawn bot.")
    
    print("\n✅ System Running. Open http://localhost:8000 to monitor.")
    print("Press Ctrl+C to stop.")
    
    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()