import torch
import torchvision
from pathlib import Path

from l2cs.model import L2CS

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "L2CSNet_gaze360.pkl"
ONNX_PATH = ROOT / "models" / "L2CSNet_gaze360_trt.onnx"

model = L2CS(
    torchvision.models.resnet.Bottleneck,
    [3, 4, 6, 3],
    90,
)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

dummy = torch.randn(1, 3, 448, 448)

torch.onnx.export(
    model,
    dummy,
    str(ONNX_PATH),
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["yaw_logits", "pitch_logits"],
)

print("Done!")