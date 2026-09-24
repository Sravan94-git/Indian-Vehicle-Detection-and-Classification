import asyncio
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.activations import swish
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import InputLayer, MultiHeadAttention
from tensorflow.keras.models import load_model
from ultralytics import YOLO

BACKEND_DIR = Path(__file__).resolve().parent
RESULT_DIR = BACKEND_DIR / "generated" / "results"
CROP_DIR = BACKEND_DIR / "generated" / "crops"
UPLOAD_DIR = BACKEND_DIR / "generated" / "uploads"
for directory in (RESULT_DIR, CROP_DIR, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Motorized2wheeler", "ambasador_taxi", "autorickshaw", "bicycle",
    "bus", "car", "minitruck", "motarvan", "rickshaw", "toto", "truck", "van",
]


class CompatibleInputLayer(InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs["batch_size"] = batch_shape[0]
            kwargs["input_shape"] = batch_shape[1:]
        super().__init__(**kwargs)


class CompatibleMultiHeadAttention(MultiHeadAttention):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)


keras.saving.get_custom_objects().update({
    "swish": swish,
    "InputLayer": CompatibleInputLayer,
    "MultiHeadAttention": CompatibleMultiHeadAttention,
})

yolo_model = None
classifier = None
model_lock = threading.Lock()


def load_models():
    global yolo_model, classifier
    if yolo_model is not None and classifier is not None:
        return yolo_model, classifier
    with model_lock:
        if yolo_model is None or classifier is None:
            yolo_model = YOLO(str(BACKEND_DIR / "weights" / "yolov8n.pt"))
            classifier = load_model(
                BACKEND_DIR / "weights" / "classifier.h5",
                compile=False,
                custom_objects={
                    "swish": swish,
                    "InputLayer": CompatibleInputLayer,
                    "MultiHeadAttention": CompatibleMultiHeadAttention,
                },
            )
    return yolo_model, classifier

app = FastAPI(title="Vehix API", version="1.0.0")


@app.on_event("startup")
async def warm_models():
    asyncio.create_task(asyncio.to_thread(load_models))


allowed_origins = [origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=BACKEND_DIR / "generated"), name="media")


def media_url(request, folder, filename):
    return f"{str(request.base_url).rstrip('/')}/media/{folder}/{filename}"


def classify_crop(crop):
    _, classifier_model = load_models()
    resized = cv2.resize(crop, (128, 128))
    normalized = resized.astype("float32") / 255.0
    prediction = classifier_model.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
    class_id = int(np.argmax(prediction))
    return CLASS_NAMES[class_id], round(float(np.max(prediction)) * 100, 1)


def detect_image(image, request):
    yolo_model_instance, _ = load_models()
    detections = yolo_model_instance(image, verbose=False)[0]
    vehicles = []
    annotated = image.copy()
    height, width = image.shape[:2]

    for index, box in enumerate(detections.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label, classification_confidence = classify_crop(crop)
        crop_filename = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(CROP_DIR / crop_filename), crop)
        detection_confidence = round(float(box.conf[0].item()) * 100, 1)
        vehicle = {
            "number": index + 1,
            "label": label,
            "classification_confidence": classification_confidence,
            "detection_confidence": detection_confidence,
            "crop_url": media_url(request, "crops", crop_filename),
            "position": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }
        vehicles.append(vehicle)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{index + 1}. {label} {detection_confidence}%", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    vehicles.sort(key=lambda item: item["detection_confidence"], reverse=True)
    return annotated, vehicles


@app.get("/health")
def health():
    if yolo_model is None or classifier is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok"}


@app.post("/api/analyze/image")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded image could not be read")

    annotated, vehicles = detect_image(image, request)
    result_filename = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(RESULT_DIR / result_filename), annotated)
    return {
        "type": "image",
        "result_url": media_url(request, "results", result_filename),
        "vehicles": vehicles,
    }


@app.post("/api/analyze/video")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    if file.content_type not in {"video/mp4", "video/quicktime", "video/x-msvideo"}:
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, or AVI video")
    source_filename = f"{uuid.uuid4().hex}.mp4"
    source_path = UPLOAD_DIR / source_filename
    source_path.write_bytes(await file.read())
    capture = cv2.VideoCapture(str(source_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = capture.get(cv2.CAP_PROP_FPS) or 24
    result_filename = f"{uuid.uuid4().hex}.mp4"
    result_path = RESULT_DIR / result_filename
    writer = cv2.VideoWriter(str(result_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))
    processed_frames = 0
    total_frames = 0
    started = time.time()
    yolo_model_instance, _ = load_models()

    try:
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            total_frames += 1
            detections = yolo_model_instance(frame, verbose=False)[0]
            annotated = frame.copy()
            detected = False
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0:
                    continue
                label, _ = classify_crop(crop)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                detected = True
            if detected:
                processed_frames += 1
            writer.write(annotated)
    finally:
        capture.release()
        writer.release()
        source_path.unlink(missing_ok=True)

    elapsed_time = time.time() - started
    return {
        "type": "video",
        "result_url": media_url(request, "results", result_filename),
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "fps": round(processed_frames / elapsed_time, 2) if elapsed_time else 0,
        "elapsed_time": round(elapsed_time, 1),
    }
