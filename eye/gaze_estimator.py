import time
from pathlib import Path
import torch
from l2cs import Pipeline


class GazeEstimator:

    def __init__(self):

        root = Path(__file__).resolve().parent.parent

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        

        self.pipeline = Pipeline(
            weights=root / "models" / "L2CSNet_gaze360.pkl",
            arch="ResNet50",
            device=device,
            include_detector=False
        )

    def estimate(self, frame):

        start = time.time()

        results = self.pipeline.step(frame)
      
        yaw = float(results.yaw[0])
        pitch = float(results.pitch[0])
        

        return yaw, pitch