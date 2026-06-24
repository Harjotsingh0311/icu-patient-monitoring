import requests

ENABLE_BACKEND = False

BACKEND_URL = "http://127.0.0.1:8000/face"

def send_to_backend(payload):

    if not ENABLE_BACKEND:
        return

    try:
        requests.post(
            BACKEND_URL,
            json=payload,
            timeout=0.1
        )

    except Exception as e:
        print(f"[Backend Error] {e}")