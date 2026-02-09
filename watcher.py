import time
import os
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

TARGET_FOLDER = os.path.abspath("./chunks")
API_URL = "http://localhost:8000/process_chunk"

class PayloadHandler(FileSystemEventHandler):
    def on_closed(self, event):
        # This event is fired when a file is closed. 
        # On Linux (inotify), this is reliable for "finish writing".
        # On other OSs, this might not fire predictably or might fire on read close.
        if event.is_directory:
            return
        self.process(event.src_path)

    def on_moved(self, event):
        # Support atomic writes: .temp.mp4 -> live.mp4 triggers a move event
        if event.is_directory:
            return
        self.process(event.dest_path)

    # Fallback or alternative if "on_closed" isn't firing expectedly (e.g. strict copy)
    # But usually copying a file triggers Create -> Modify... -> Close.
    # We'll use on_closed to ensure we don't read partial files.

    def process(self, file_path):
        filename = os.path.basename(file_path)
        
        # 1. Check extension and temp files
        if not filename.endswith(".mp4"):
            return
        if filename.startswith("."):
            return

        print(f"[Watcher] New MP4 detected: {filename}")
        
        # 2. Send to API
        try:
            payload = {"file_path": file_path}
            print(f"[Watcher] Sending request for {filename}...")
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print(f"[Watcher] Success: {response.json().get('verdict')}")
            else:
                print(f"[Watcher] Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[Watcher] Failed to connect to API: {e}")

if __name__ == "__main__":
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)
        print(f"Created directory: {TARGET_FOLDER}")

    event_handler = PayloadHandler()
    observer = Observer()
    # Watch recursively? No, just the folder.
    observer.schedule(event_handler, TARGET_FOLDER, recursive=False)
    
    print(f"[Watcher] Monitoring {TARGET_FOLDER} for .mp4 files...")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()
