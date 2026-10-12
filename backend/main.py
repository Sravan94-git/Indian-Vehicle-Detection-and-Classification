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

# COCO vehicle classes: bicycle (1), car (2), motorcycle (3), bus (5), truck (7).
# Explicitly excludes humans (COCO class 0: person) and other non-vehicle entities.
COCO_VEHICLE_CLASSES = [1, 2, 3, 5, 7]


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
model_status = "starting"
model_error = None


def load_models():
    global yolo_model, classifier, model_status, model_error
    if yolo_model is not None and classifier is not None:
        return yolo_model, classifier
    with model_lock:
        if yolo_model is None or classifier is None:
            try:
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
                model_status = "ready"
            except Exception as error:
                model_status = "error"
                model_error = str(error)
                raise
    return yolo_model, classifier

app = FastAPI(title="Vehix API", version="1.0.0")


@app.on_event("startup")
async def warm_models():
    async def warm_in_background():
        try:
            await asyncio.to_thread(load_models)
        except Exception:
            pass

    asyncio.create_task(warm_in_background())


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.mount("/media", StaticFiles(directory=BACKEND_DIR / "generated"), name="media")


def media_url(request: Request, folder: str, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    proto = request.headers.get("x-forwarded-proto")
    if proto:
        scheme = proto.split(",")[0].strip()
        if scheme and base.startswith("http://"):
            base = f"{scheme}://{base[7:]}"
    return f"{base}/media/{folder}/{filename}"


def classify_crop(crop):
    _, classifier_model = load_models()
    resized = cv2.resize(crop, (128, 128))
    normalized = resized.astype("float32") / 255.0
    prediction = classifier_model.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
    class_id = int(np.argmax(prediction))
    return CLASS_NAMES[class_id], round(float(np.max(prediction)) * 100, 1)


def detect_image(image, request):
    yolo_model_instance, _ = load_models()
    # Explicitly filter YOLO to only detect vehicle categories (excluding humans/pedestrians)
    detections = yolo_model_instance(
        image,
        classes=COCO_VEHICLE_CLASSES,
        conf=0.25,
        verbose=False,
    )[0]
    vehicles = []
    annotated = image.copy()
    height, width = image.shape[:2]

    detected_count = 0
    for box in detections.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in COCO_VEHICLE_CLASSES:
            continue

        detection_confidence = round(float(box.conf[0].item()) * 100, 1)
        if detection_confidence < 25.0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1 < 12) or (y2 - y1 < 12):
            continue

        label, classification_confidence = classify_crop(crop)
        detected_count += 1
        crop_filename = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(CROP_DIR / crop_filename), crop)
        vehicle = {
            "number": detected_count,
            "label": label,
            "classification_confidence": classification_confidence,
            "detection_confidence": detection_confidence,
            "crop_url": media_url(request, "crops", crop_filename),
            "position": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }
        vehicles.append(vehicle)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{detected_count}. {label} {detection_confidence}%",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    vehicles.sort(key=lambda item: item["detection_confidence"], reverse=True)
    for idx, v in enumerate(vehicles):
        v["number"] = idx + 1
    return annotated, vehicles


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "status": "ok",
        "service": "Vehix API",
        "version": "1.0.0",
        "model_status": model_status,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "analyze_image": "/api/analyze/image",
            "analyze_video": "/api/analyze/video",
        },
    }


@app.api_route("/favicon.ico", methods=["GET", "HEAD"])
def favicon():
    return JSONResponse(status_code=204, content=None)


@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/api/health", methods=["GET", "HEAD"])
def health():
    if model_status == "error":
        return JSONResponse(status_code=200, content={"status": "error", "detail": model_error})
    return {"status": "ok" if model_status == "ready" else "starting"}


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "detail": f"Path '{request.url.path}' not found.",
            "hint": "Check /docs for valid endpoints or call /health for service status.",
            "available_endpoints": ["/", "/health", "/api/health", "/docs", "/api/analyze/image", "/api/analyze/video"],
        },
    )


@app.post("/api/analyze/image")
@app.post("/analyze/image")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    valid_image_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type and file.content_type not in valid_image_types and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WEBP image")
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded image could not be read or decoded")

    annotated, vehicles = detect_image(image, request)
    result_filename = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(RESULT_DIR / result_filename), annotated)
    return {
        "type": "image",
        "result_url": media_url(request, "results", result_filename),
        "vehicles": vehicles,
    }


@app.post("/api/analyze/video")
@app.post("/analyze/video")
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
            detections = yolo_model_instance(
                frame,
                classes=COCO_VEHICLE_CLASSES,
                conf=0.25,
                verbose=False,
            )[0]
            annotated = frame.copy()
            detected = False
            for box in detections.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in COCO_VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0 or (x2 - x1 < 12) or (y2 - y1 < 12):
                    continue
                label, _ = classify_crop(crop)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
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
