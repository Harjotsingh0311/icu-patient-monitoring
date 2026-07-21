import onnxruntime as ort

session = ort.InferenceSession(
    "models/L2CSNet_gaze360.onnx",
    providers=["CPUExecutionProvider"]
)

print("========== INPUTS ==========")
for inp in session.get_inputs():
    print("Name :", inp.name)
    print("Shape:", inp.shape)
    print("Type :", inp.type)

print()

print("========== OUTPUTS ==========")
for out in session.get_outputs():
    print("Name :", out.name)
    print("Shape:", out.shape)
    print("Type :", out.type)