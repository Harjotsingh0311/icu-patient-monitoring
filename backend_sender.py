import threading
import cv2
import requests

# ==========================================================
# Configuration
# ==========================================================

ENABLE_BACKEND = False

BACKEND_URL = "http://172.20.10.2:8000/api/face-eye/features"
STREAM_URL = "http://172.20.10.2:8000/api/stream/face"

# Persistent HTTP session
session = requests.Session()

# ==========================================================
# Clinical Feature Sender
# ==========================================================

def send_to_backend(payload):
    """
    Sends clinical feature vectors to the backend.

    Clinical packets must never be silently dropped.
    """

    if not ENABLE_BACKEND:
        return

    try:
        session.post(
            BACKEND_URL,
            json=payload,
            timeout=1.0,
        )

    except Exception as e:
        print(f"[Backend Error] {e}")


# ==========================================================
# Dashboard Video Streaming
# ==========================================================

_latest_frame = None

_frame_lock = threading.Lock()

_frame_available = threading.Condition(_frame_lock)


def _frame_sender():
    """
    Background thread.

    Waits until a new frame becomes available,
    then uploads the latest frame to the dashboard.
    """

    global _latest_frame

    while True:

        # Wait until producer provides a frame
        with _frame_available:

            while _latest_frame is None:
                _frame_available.wait()

            frame = _latest_frame
            _latest_frame = None

        try:

            success, buffer = cv2.imencode(".jpg", frame)

            if not success:
                continue

            session.post(
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

        except Exception:
            # Streaming failures should never interrupt
            # the clinical pipeline.
            pass


# Start exactly one sender thread
_sender_thread = threading.Thread(
    target=_frame_sender,
    daemon=True,
)

_sender_thread.start()


def send_frame_to_backend(frame):
    """
    Producer.

    Called from the main clinical pipeline.

    Stores only the latest dashboard frame and wakes
    the sender thread.
    """

    if not ENABLE_BACKEND:
        return

    global _latest_frame

    with _frame_available:

        # Keep only the newest frame.
        _latest_frame = frame.copy()

        # Wake sender thread.
        _frame_available.notify()