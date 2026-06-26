import requests

ENABLE_BACKEND = True

BACKEND_URL = "http://172.20.10.3:8000/face"

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

    except Exception as e:
        print(f"[Backend Error] {e}")