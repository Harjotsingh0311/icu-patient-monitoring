import cv2
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from l2cs.utils import getArch, prep_input_numpy

# ----------------------------
# CHANGE THIS
# ----------------------------
IMAGE_PATH = "test.JPG"

# ----------------------------
# Load image
# ----------------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(IMAGE_PATH)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (448, 448))

# ----------------------------
# PyTorch
# ----------------------------
model = getArch("ResNet50", 90)
model.load_state_dict(torch.load("models/L2CSNet_gaze360.pkl", map_location="cpu"))
model.eval()

x = prep_input_numpy(img, "cpu")

with torch.no_grad():
    yaw_pt, pitch_pt = model(x)

yaw_pt = yaw_pt.numpy()
pitch_pt = pitch_pt.numpy()

# ----------------------------
# ONNX
# ----------------------------
session = ort.InferenceSession(
    "models/L2CSNet_gaze360.onnx",
    providers=["CPUExecutionProvider"]
)

yaw_onnx, pitch_onnx = session.run(
    None,
    {"input": x.numpy()}
)

print("Yaw max diff   :", np.max(np.abs(yaw_pt - yaw_onnx)))
print("Pitch max diff :", np.max(np.abs(pitch_pt - pitch_onnx)))