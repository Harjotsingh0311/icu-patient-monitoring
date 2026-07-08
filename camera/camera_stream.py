import cv2
import threading


class CameraStream:

    def __init__(self, camera_id=0, width=1280, height=720, fps=30):

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.lock = threading.Lock()

        self.frame = None
        self.running = False

        self.thread = threading.Thread(
            target=self._update,
            daemon=True,
        )

    def start(self):

        self.running = True
        self.thread.start()

        return self

    def _update(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                print("[Camera] Failed to grab frame")
                continue

            print("[Camera] Frame received")

            with self.lock:
                self.frame = frame

    def read(self):

        with self.lock:

            if self.frame is None:
                return False, None

            return True, self.frame.copy()

    def stop(self):

        self.running = False

        if self.thread.is_alive():
            self.thread.join()

        self.cap.release()