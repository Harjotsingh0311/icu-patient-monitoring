import requests

ENABLE_BACKEND = True

BACKEND_URL = "http://172.20.10.2:8001/api/face-eye/features"

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