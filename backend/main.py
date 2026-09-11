from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

import io


app = FastAPI()


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)


@app.get("/")
def dashboard():

    return FileResponse(
        "frontend/index.html"
    )


# ============================================================
# DATA STORAGE
# ============================================================

face_eye_data = {}

posture_data = {}

latest_frame = None


# ============================================================
# FACE + EYE
# ============================================================

@app.post("/api/face-eye/features")
def receive_face_eye(data: dict):

    global face_eye_data

    face_eye_data = data

    print(
        "[Backend] Received Face + Eye:",
        face_eye_data
    )

    return {
        "received": True
    }


@app.get("/api/face-eye/features")
def get_face_eye():

    return face_eye_data


# ============================================================
# POSTURE
# ============================================================

@app.post("/api/posture")
def receive_posture(data: dict):

    global posture_data

    posture_data = data

    return {
        "received": True
    }


@app.get("/api/posture")
def get_posture():

    return posture_data


# ============================================================
# FUSION
# ============================================================

@app.get("/api/fusion")
def fusion():

    return {
        "face_eye": face_eye_data,
        "posture": posture_data
    }


# ============================================================
# FACE FRAME STREAM
# ============================================================

@app.post("/api/stream/face")
async def receive_frame(
    frame: UploadFile = File(...)
):

    global latest_frame

    latest_frame = await frame.read()

    return {
        "received": True
    }


@app.get("/api/stream/face")
def get_frame():

    if latest_frame is None:

        return {
            "error": "No frame available"
        }

    return StreamingResponse(
        io.BytesIO(latest_frame),
        media_type="image/jpeg"
    )