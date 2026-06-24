from fastapi import FastAPI

app = FastAPI()

face_data = {}
posture_data = {}


@app.post("/face")
def receive_face(data: dict):
    global face_data
    face_data = data

    print("Received Face Data:", face_data)

    return {"received": True}


@app.get("/face")
def get_face():
    return face_data


@app.post("/posture")
def receive_posture(data: dict):
    global posture_data
    posture_data = data

    print("Received Posture Data:", posture_data)

    return {"received": True}


@app.get("/posture")
def get_posture():
    return posture_data


@app.get("/fusion")
def fusion():

    return {
        "face_eye": face_data,
        "posture": posture_data
    }