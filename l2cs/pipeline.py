import pathlib
import time 

import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
try:
    from face_detection import RetinaFace
except ImportError:
    RetinaFace = None
from .utils import prep_input_numpy
from .results import GazeResultContainer


class Pipeline:

    def __init__(
        self, 
        weights: pathlib.Path, 
        arch: str,
        device: str = 'cpu', 
        include_detector:bool = True,
        confidence_threshold:float = 0.5
        ):

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Create L2CS model
        # Create ONNX Runtime session
        onnx_path = pathlib.Path(self.weights).with_suffix(".onnx")

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        # Create RetinaFace if requested
        if self.include_detector:

            if RetinaFace is None:
                raise ImportError(
                    "RetinaFace requested but face_detection package not installed"
                )

            if device.type == 'cpu':
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=device.index)

        

        self.idx_tensor = np.arange(90, dtype=np.float32)

    def step(self, frame: np.ndarray) -> GazeResultContainer:

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detector(frame)

            if faces is not None: 
                for box, landmark, score in faces:

                    # Apply threshold
                    if score < self.confidence_threshold:
                        continue

                    # Extract safe min and max of x,y
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Save data
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # Predict gaze
                pitch, yaw = self.predict_gaze(np.stack(face_imgs))

            else:

                pitch = np.empty((0,1))
                yaw = np.empty((0,1))

        else:
            pitch, yaw = self.predict_gaze(frame)

        # Save data
        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.array(bboxes),
            landmarks=np.array(landmarks),
            scores=np.array(scores)
        )

        return results

    import time

    def predict_gaze(self, frame):

        # ---------------- Preprocessing ----------------
        t = time.perf_counter()

        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame)
        else:
            raise RuntimeError("Invalid dtype for input")


        # ---------------- ONNX ----------------
        t = time.perf_counter()

        yaw_logits, pitch_logits = self.session.run(
            None,
            {self.input_name: img}
        )


        # ---------------- Post ----------------
        t = time.perf_counter()

        def softmax(x):
            x = x - np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

        yaw_prob = softmax(yaw_logits)
        pitch_prob = softmax(pitch_logits)

        yaw_predicted = np.sum(yaw_prob * self.idx_tensor, axis=1) * 4 - 180
        pitch_predicted = np.sum(pitch_prob * self.idx_tensor, axis=1) * 4 - 180

        yaw_predicted = yaw_predicted * np.pi / 180.0
        pitch_predicted = pitch_predicted * np.pi / 180.0


        return pitch_predicted, yaw_predicted