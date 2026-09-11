import threading
import cv2
import requests


# ==========================================================
# Configuration
# ==========================================================

ENABLE_BACKEND = True

BACKEND_URL = "http://127.0.0.1:8000/api/face-eye/features"
STREAM_URL = "http://127.0.0.1:8000/api/stream/face"

# Persistent HTTP session
session = requests.Session()


# ==========================================================
# Clinical Feature Sender
# ==========================================================

def send_to_backend(payload):
    """
    Sends clinical feature data to the FastAPI backend.
    """

    if not ENABLE_BACKEND:
        return

    try:
        response = session.post(
            BACKEND_URL,
            json=payload,
            timeout=1.0,
        )

        if response.status_code != 200:
            print(
                f"[Backend Error] "
                f"Status: {response.status_code}"
            )

    except requests.RequestException as e:
        print(f"[Backend Error] {e}")


# ==========================================================
# Dashboard Video Streaming
# ==========================================================

_latest_frame = None

_frame_lock = threading.Lock()

_frame_available = threading.Condition(_frame_lock)


def _frame_sender():
    """
    Background thread responsible for sending
    the latest camera frame to the dashboard.
    """

    global _latest_frame

    while True:

        # --------------------------------------------------
        # Wait for a new frame
        # --------------------------------------------------

        with _frame_available:

            while _latest_frame is None:
                _frame_available.wait()

            frame = _latest_frame

            # Remove reference so producer can provide
            # the next frame.
            _latest_frame = None

        # --------------------------------------------------
        # Encode frame
        # --------------------------------------------------

        try:

            success, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not success:
                continue

            # --------------------------------------------------
            # Send frame to backend
            # --------------------------------------------------

            response = session.post(
                STREAM_URL,
                files={
                    "frame": (
                        "frame.jpg",
                        buffer.tobytes(),
                        "image/jpeg",
                    )
                },
                timeout=0.2,
            )

            if response.status_code != 200:
                print(
                    f"[Stream Error] "
                    f"Status: {response.status_code}"
                )

        except requests.RequestException:
            # Video streaming is non-critical.
            # Never stop the clinical pipeline.
            pass

        except Exception as e:
            print(f"[Stream Error] {e}")


# ==========================================================
# Start Background Sender
# ==========================================================

_sender_thread = threading.Thread(
    target=_frame_sender,
    daemon=True,
)

_sender_thread.start()


# ==========================================================
# Frame Producer
# ==========================================================

def send_frame_to_backend(frame):
    """
    Called by the main Face + Eye pipeline.

    Only the newest frame is retained because the
    dashboard does not need every camera frame.
    """

    if not ENABLE_BACKEND:
        return

    global _latest_frame

    with _frame_available:

        # Replace old frame with newest frame
        _latest_frame = frame.copy()

        # Wake the sender thread
        _frame_available.notify()