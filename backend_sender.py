import requests

BACKEND_URL = "http://172.20.10.3:8000/face"


def send_to_backend(face_payload):

    try:

        requests.post(
            BACKEND_URL,
            json=face_payload,
            timeout=1
        )

    except Exception as e:

        print(
            f"[Backend Error] {e}"
        )