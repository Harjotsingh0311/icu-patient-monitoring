import requests
import threading
import cv2

ENABLE_BACKEND = True

BACKEND_URL = "http://172.20.10.2:8000/api/face-eye/features"
STREAM_URL  = "http://172.20.10.2:8000/api/stream/face/live"

session = requests.Session()

def send_to_backend(payload):

    if not ENABLE_BACKEND:
        return

    try:
        response = session.post(
            BACKEND_URL,
            json=payload,
            timeout=1.0
        )
        print(response.status_code)
        print(response.text)

    except Exception as e:
        print(f"[Backend Error] {e}")

def send_frame_to_backend(frame):
    def worker():
        try:
            _, buffer = cv2.imencode(".jpg", frame)

            requests.post(
                STREAM_URL,
                files={
                    "frame": ("frame.jpg", buffer.tobytes(), "image/jpeg")
                },
                timeout=0.2,
            )

        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()