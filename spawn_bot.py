import os
import sys
import requests
import argparse
from dotenv import load_dotenv

load_dotenv()

# Check for API Key
API_KEY = os.getenv("RECALL_API_KEY")
NGROK_URL = os.getenv("NGROK_URL")
REGION = os.getenv("REGION", "us-east-1")

def spawn_bot(meeting_url, bot_name="Liveness Detector"):
    if not API_KEY:
        print("Error: RECALL_API_KEY environment variable not set.")
        print("Please set it in your .env file.")
        sys.exit(1)

    if not NGROK_URL:
        print("Error: NGROK_URL environment variable not set.")
        print("Please set it in your .env file.")
        sys.exit(1)

    # Clean the ngrok URL to ensure it doesn't end with a slash
    ngrok_url = NGROK_URL.rstrip("/")

    # Convert HTTP/HTTPS to WS/WSS for the config
    # Note: Recall.ai expects the full websocket URL in the config
    if ngrok_url.startswith("https://"):
        ws_base = ngrok_url.replace("https://", "wss://")
    elif ngrok_url.startswith("http://"):
        ws_base = ngrok_url.replace("http://", "ws://")
    else:
        # Fallback if user just provided domain
        ws_base = f"wss://{ngrok_url}"
    
    # Define endpoints matching bridge_server.py
    video_ws_url = f"{ws_base}/recall-video-endpoint"
    audio_ws_url = f"{ws_base}/recall-audio-endpoint"

    # Construct the payload based on Recall.ai API v1 documentation
    payload = {
        "meeting_url": meeting_url,
        "bot_name": bot_name,
        "variant": {
            "zoom": "web_4_core",
            "google_meet": "web_4_core",
            "microsoft_teams": "web_4_core", 
            "webex": "web_4_core"
        },
        "recording_config": {
            "audio_mixed_raw": {},
            "video_separate_h264": {},
            "realtime_endpoints": [
                {
                    "type": "websocket",
                    "url": audio_ws_url,
                    "events": ["audio_mixed_raw.data"]
                },
                {
                    "type": "websocket",
                    "url": video_ws_url,
                    # We request separate H.264 video streams.
                    # Note: If multiple participants have video on, the bridge server
                    # will receive data for all of them.
                    "events": ["video_separate_h264.data"]
                }
            ]
        }
    }

    headers = {
        "Authorization": f"Token {API_KEY}" if not API_KEY.startswith("Token") else API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    print("-" * 50)
    print(f"Spawning Bot: {bot_name}")
    print(f"Target Meeting: {meeting_url}")
    print(f"Audio Pipe: {audio_ws_url}")
    print(f"Video Pipe: {video_ws_url}")
    print("-" * 50)

    try:
        api_url = f"https://{REGION}.recall.ai/api/v1/bot/"
        print(f"Connecting to Region: {REGION} ({api_url})")

        response = requests.post(api_url, json=payload, headers=headers)
        
        if response.status_code in [200, 201]:
            data = response.json()
            print("\n✅ Bot Successfully Created!")
            print(f"Bot ID: {data.get('id')}")
            print("Check your bridge server logs for connection...")
        else:
            print(f"\n❌ Failed to spawn bot. Status Code: {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spawn a Recall.ai bot for Liveness Detection")
    parser.add_argument("--url", required=True, help="The Zoom/Google Meet/Teams meeting URL")
    parser.add_argument("--name", default="Deepfake Detector", help="Name of the bot to appear in meeting")
    
    args = parser.parse_args()
    
    spawn_bot(args.url, args.name)