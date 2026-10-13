import os

# Limit internal memory pools, thread stacks, and disable GPU scanning
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_OFFLINE"] = "1"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "False"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"

import asyncio
import gc
import threading
import time
import uuid
from pathlib import Path

# Setup C-level memory trimming for Linux containers (Render free tier)
try:
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    def trim_memory():
        gc.collect()
        try:
            libc.malloc_trim(0)
        except Exception:
            pass
except Exception:
    def trim_memory():
        gc.collect()

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# Restrict PyTorch execution to 2 threads without gradient tracking
torch.set_grad_enabled(False)
torch.set_num_threads(2)

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
COCO_VEHICLE_CLASSES = [1, 2, 3, 5, 7]


def cleanup_old_files():
    """Remove generated images older than 20 minutes to prevent storage buildup."""
    now = time.time()
    for folder in (RESULT_DIR, CROP_DIR, UPLOAD_DIR):
        try:
            for item in folder.glob("*"):
                if item.is_file() and now - item.stat().st_mtime > 1200:
                    item.unlink(missing_ok=True)
        except Exception:
            pass


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
                if yolo_model is None:
                    yolo_model = YOLO(str(BACKEND_DIR / "weights" / "yolov8n.pt"))
                    trim_memory()
                if classifier is None:
                    classifier = torch.jit.load(
                        str(BACKEND_DIR / "weights" / "classifier.pt"),
                        map_location="cpu",
                    )
                    classifier.eval()
                    trim_memory()
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
        await asyncio.sleep(1)
        try:
            await asyncio.to_thread(load_models)
        except Exception as e:
            print(f"Error warming models: {e}")

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


def detect_image(image, request):
    t_start = time.time()
    yolo_model_instance, classifier_model = load_models()
    height, width = image.shape[:2]

    # 1. Run YOLO detection with strict 640 input resolution and inference mode
    with torch.inference_mode():
        detections = yolo_model_instance(
            image,
            classes=COCO_VEHICLE_CLASSES,
            conf=0.25,
            imgsz=640,
            verbose=False,
        )[0]

    t_yolo = time.time()
    annotated = image.copy()

    # 2. Extract valid vehicle bounding boxes, capped at top 10 to protect memory
    valid_boxes = []
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
        if (x2 - x1 < 12) or (y2 - y1 < 12):
            continue
        valid_boxes.append((detection_confidence, x1, y1, x2, y2))

    valid_boxes.sort(key=lambda item: item[0], reverse=True)
    valid_boxes = valid_boxes[:10]

    vehicles = []
    crops_to_classify = []
    crops_meta = []

    for idx, (detection_confidence, x1, y1, x2, y2) in enumerate(valid_boxes):
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (128, 128))
        crops_to_classify.append(crop_resized.astype("float32") / 255.0)

        crop_filename = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(CROP_DIR / crop_filename), crop)

        crops_meta.append({
            "number": idx + 1,
            "detection_confidence": detection_confidence,
            "crop_url": media_url(request, "crops", crop_filename),
            "position": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    # 3. Classify all detected crops in a single fast forward pass
    if crops_to_classify:
        batch = torch.from_numpy(np.array(crops_to_classify, dtype=np.float32))
        with torch.inference_mode():
            predictions = classifier_model(batch).numpy()
        del batch
        del crops_to_classify

        for meta, pred in zip(crops_meta, predictions):
            class_id = int(np.argmax(pred))
            label = CLASS_NAMES[class_id]
            classification_confidence = round(float(np.max(pred)) * 100, 1)
            x1, y1 = meta["position"]["x1"], meta["position"]["y1"]
            x2, y2 = meta["position"]["x2"], meta["position"]["y2"]

            vehicles.append({
                "number": meta["number"],
                "label": label,
                "classification_confidence": classification_confidence,
                "detection_confidence": meta["detection_confidence"],
                "crop_url": meta["crop_url"],
                "position": meta["position"],
            })

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{meta['number']}. {label} {meta['detection_confidence']}%",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    t_done = time.time()
    print(f"[DETECT_TIMING] YOLO: {t_yolo - t_start:.2f}s | Classifier: {t_done - t_yolo:.2f}s | Total: {t_done - t_start:.2f}s")

    del detections
    trim_memory()
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
    t0 = time.time()
    # Trigger cleanup in background without blocking current request
    asyncio.create_task(asyncio.to_thread(cleanup_old_files))

    valid_image_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type and file.content_type not in valid_image_types and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WEBP image")
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    del contents
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded image could not be read or decoded")

    # Downscale high-resolution images to max dimension 640 to run 4x faster on CPU
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > 640:
        scale = 640.0 / max_dim
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Offload CPU inference to worker thread so event loop never stalls
    annotated, vehicles = await asyncio.to_thread(detect_image, image, request)

    result_filename = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(RESULT_DIR / result_filename), annotated)
    del annotated
    trim_memory()

    print(f"[API_ANALYZE_IMAGE] Total request time: {time.time() - t0:.2f}s")
    return {
        "type": "image",
        "result_url": media_url(request, "results", result_filename),
        "vehicles": vehicles,
    }


@app.post("/api/analyze/video")
@app.post("/analyze/video")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    asyncio.create_task(asyncio.to_thread(cleanup_old_files))
    if file.content_type not in {"video/mp4", "video/quicktime", "video/x-msvideo"}:
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, or AVI video")
    source_filename = f"{uuid.uuid4().hex}.mp4"
    source_path = UPLOAD_DIR / source_filename
    source_path.write_bytes(await file.read())
    capture = cv2.VideoCapture(str(source_path))
    orig_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = capture.get(cv2.CAP_PROP_FPS) or 24

    # Cap video resolution to max 640 to prevent CPU stall
    scale = 1.0
    max_dim = max(orig_w, orig_h)
    if max_dim > 640:
        scale = 640.0 / max_dim
    out_w = int(orig_w * scale)
    out_h = int(orig_h * scale)

    result_filename = f"{uuid.uuid4().hex}.mp4"
    result_path = RESULT_DIR / result_filename
    writer = cv2.VideoWriter(str(result_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (out_w, out_h))
    processed_frames = 0
    total_frames = 0
    started = time.time()
    yolo_model_instance, classifier_model = load_models()

    try:
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            total_frames += 1

            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            with torch.inference_mode():
                detections = yolo_model_instance(
                    frame,
                    classes=COCO_VEHICLE_CLASSES,
                    conf=0.25,
                    imgsz=640,
                    verbose=False,
                )[0]

            annotated = frame.copy()
            detected = False
            for box in detections.boxes[:5]:  # limit to top 5 vehicles per video frame
                cls_id = int(box.cls[0].item())
                if cls_id not in COCO_VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0 or (x2 - x1 < 12) or (y2 - y1 < 12):
                    continue
                crop_in = torch.from_numpy(
                    np.expand_dims(cv2.resize(crop, (128, 128)).astype("float32") / 255.0, axis=0)
                )
                with torch.inference_mode():
                    pred = classifier_model(crop_in).numpy()[0]
                label = CLASS_NAMES[int(np.argmax(pred))]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
                detected = True

            if detected:
                processed_frames += 1
            writer.write(annotated)

            if total_frames % 25 == 0:
                trim_memory()
    finally:
        capture.release()
        writer.release()
        source_path.unlink(missing_ok=True)
        trim_memory()

    elapsed_time = time.time() - started
    return {
        "type": "video",
        "result_url": media_url(request, "results", result_filename),
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "fps": round(processed_frames / elapsed_time, 2) if elapsed_time else 0,
        "elapsed_time": round(elapsed_time, 1),
    }
