# 🚗 Vehix - Indian Vehicle Detection & Classification

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb.svg)](https://vitejs.dev/)
[![Model](https://img.shields.io/badge/Detection-YOLOv8n-green.svg)](https://github.com/ultralytics/ultralytics)
[![Runtime](https://img.shields.io/badge/Runtime-PyTorch%20CPU-ee4c2c.svg)](https://pytorch.org/)
[![Deploy](https://img.shields.io/badge/Backend-Render-46e3b7.svg)](https://render.com/)
[![Deploy](https://img.shields.io/badge/Frontend-Vercel-black.svg)](https://vercel.com/)

Vehix is an AI-powered **full-stack application** for detecting and classifying vehicles in Indian road imagery. It combines **YOLOv8n** for spatial localization with a custom **MobileNetV2 + Multi-Head Attention** classifier converted to **TorchScript** for fast, memory-efficient inference on free-tier cloud platforms. Users can upload images or videos and review annotated results, confidence scores, and individual vehicle crops.

***

## ✨ Key Features

* **Two-stage vehicle detection pipeline:** YOLOv8n locates vehicles → custom classifier identifies the vehicle type.
* **12-class Indian vehicle taxonomy:** Trained on Indian road categories including autorickshaws, ambassador taxis, totos, and more.
* **Image analysis:** Returns annotated images with bounding boxes, individual vehicle crops, labels, and dual confidence scores (detection + classification).
* **Video analysis:** Frame-by-frame processing of MP4, MOV, or AVI files with annotated output video.
* **Pure PyTorch runtime:** TensorFlow-free — the classifier runs as a TorchScript module on the PyTorch CPU backend, minimizing RAM from ~1.2 GB to ~325 MB.
* **JIT-warmup at startup:** Both YOLO and the TorchScript classifier run a dummy inference pass during server boot, eliminating the 2-minute first-request JIT compilation penalty.
* **Memory-optimized for free-tier deployment:** Fits within Render's 512 MB RAM limit using `malloc_trim`, batched inference, and aggressive garbage collection.
* **Readiness screen:** The React frontend polls the backend health endpoint and shows a loading overlay until models are warm and ready.
* **Interactive API documentation:** FastAPI auto-generates Swagger UI at `/docs`.

***

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **Detection Model** | Ultralytics YOLOv8n (nano) at 320px input |
| **Classifier Model** | MobileNetV2 + Multi-Head Attention → TorchScript (`.pt`) |
| **Deep Learning Runtime** | PyTorch 2.5 (CPU-only, no TensorFlow) |
| **Backend Framework** | FastAPI + Uvicorn |
| **Frontend** | React 18 + Vite |
| **Computer Vision** | OpenCV (headless) |
| **Containerization** | Docker (python:3.12-slim) |
| **Backend Hosting** | Render (free tier) |
| **Frontend Hosting** | Vercel |

***

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Vercel)                           │
│                                                                     │
│   React + Vite SPA                                                  │
│   ├── Health polling  →  GET /health  (3.5s interval)               │
│   ├── Image upload    →  POST /api/analyze/image                    │
│   ├── Video upload    →  POST /api/analyze/video                    │
│   └── Live progress tracker with pipeline stage animation           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ HTTPS
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BACKEND (Render Free Tier)                       │
│                                                                     │
│   Docker Container  (python:3.12-slim, 512 MB RAM limit)            │
│   ├── FastAPI + Uvicorn (1 worker, 2 threads)                       │
│   ├── CORS middleware for Vercel origin                              │
│   └── Static file serving at /media/*                               │
│                                                                     │
│   ┌─── STARTUP ──────────────────────────────────────────────────┐  │
│   │  1. Load YOLOv8n weights (.pt)                               │  │
│   │  2. Load TorchScript classifier (.pt)                        │  │
│   │  3. JIT Warmup: dummy inference on both models               │  │
│   │  4. malloc_trim() → reclaim C-heap memory                    │  │
│   │  5. Health endpoint returns { status: "ok" }                 │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   ┌─── INFERENCE PIPELINE ───────────────────────────────────────┐  │
│   │                                                              │  │
│   │  Input Image/Frame                                           │  │
│   │       │                                                      │  │
│   │       ▼  Downscale to max 480px                              │  │
│   │       │                                                      │  │
│   │       ▼  YOLOv8n Detection (imgsz=320, conf≥0.25)           │  │
│   │       │  → Bounding boxes for COCO vehicle classes           │  │
│   │       │    [bicycle, car, motorcycle, bus, truck]             │  │
│   │       │                                                      │  │
│   │       ▼  Crop & Resize to 128×128                            │  │
│   │       │  → Batch up to 10 vehicle crops                      │  │
│   │       │                                                      │  │
│   │       ▼  TorchScript Classifier (batched forward pass)       │  │
│   │       │  → 12-class prediction per crop                      │  │
│   │       │                                                      │  │
│   │       ▼  Annotate + Save results                             │  │
│   │       │  → Bounding boxes, labels, confidence bars           │  │
│   │       │                                                      │  │
│   │       ▼  malloc_trim() → reclaim memory                      │  │
│   │                                                              │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

***

## 🚀 Inference Pipeline Flow

| Stage | Operation | Details |
|-------|-----------|---------|
| **1. Ingestion** | Image decode & downscale | `cv2.imdecode` → resize to max 480px (INTER_AREA) |
| **2. Detection** | YOLOv8n forward pass | `imgsz=320`, filters COCO vehicle classes `[1,2,3,5,7]`, `conf≥0.25` |
| **3. Extraction** | Crop bounding boxes | Top 10 detections sorted by confidence, each resized to 128×128 |
| **4. Classification** | TorchScript batch inference | Single batched forward pass through MobileNetV2+MHA classifier |
| **5. Annotation** | Draw results | Bounding boxes, class labels, and confidence % rendered on image |
| **6. Cleanup** | Memory reclamation | `gc.collect()` + `libc.malloc_trim(0)` to stay under 512 MB |

***

## 🎯 Vehicle Classes

The classifier recognizes 12 Indian vehicle categories:

| Class | Class | Class |
|---|---|---|
| Motorized two-wheeler | Ambassador taxi | Autorickshaw |
| Bicycle | Bus | Car |
| Minitruck | Motorvan | Rickshaw |
| Toto (E-rickshaw) | Truck | Van |

***

## ⚡ Performance Optimizations

These optimizations were implemented to run within Render's free-tier constraints (512 MB RAM, 0.1 shared vCPU):

| Optimization | What it does | Impact |
|---|---|---|
| **TensorFlow removal** | Replaced TF/Keras with pure PyTorch + TorchScript | RAM: 1.2 GB → ~325 MB |
| **TorchScript classifier** | Converted `.h5` Keras model → `.pt` TorchScript | Faster load, no Keras overhead |
| **JIT warmup at startup** | Dummy inference on both models during boot | Eliminates 2-min first-request delay |
| **YOLO imgsz 640 → 320** | 4× fewer pixels processed by YOLO | ~4× faster detection |
| **Image pre-resize to 480px** | Downscale before YOLO to reduce pre-processing | Less memory, faster decode |
| **Batched classification** | All crops classified in one forward pass | Eliminates per-crop overhead |
| **malloc_trim()** | C-level memory reclamation on Linux | Prevents OOM restarts |
| **2-thread limit** | `OMP/OpenBLAS/MKL` threads capped at 2 | Balances speed vs. RAM |
| **File cleanup** | Auto-delete generated files older than 20 min | Prevents storage buildup |
| **torch.inference_mode** | Disables autograd entirely during inference | Lower memory, faster execution |

***

## 📂 Project Structure

```text
Vehix/
│
├── backend/
│   ├── main.py                 # FastAPI app, inference pipeline, memory management
│   ├── requirements.txt        # PyTorch CPU, Ultralytics, FastAPI, OpenCV
│   ├── Dockerfile              # python:3.12-slim with optimized env vars
│   └── weights/
│       ├── classifier.pt       # TorchScript classifier (MobileNetV2 + MHA)
│       └── yolov8n.pt          # YOLOv8 nano detection weights
│
├── frontend/
│   ├── src/
│   │   ├── main.jsx            # React app with health polling, upload, analysis UI
│   │   └── styles.css          # Dark theme, glassmorphism, micro-animations
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

***

## 🌐 Deployment

### Backend (Render)

The backend is deployed as a **Docker Web Service** on Render's free tier:

- **Build:** Docker image from `backend/Dockerfile`
- **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
- **Cold start:** ~45–60s for the container to boot and warm up models
- **RAM budget:** ~325 MB out of 512 MB limit

### Frontend (Vercel)

The React frontend is deployed on Vercel with the `VITE_API_URL` environment variable pointing to the Render backend URL.

***

## 📊 Applications

* 🚦 Traffic monitoring and vehicle counting
* 🛣️ Indian road-scene analysis
* 🏙️ Smart-city transportation systems
* 📹 Traffic video processing
* 🔍 Vehicle dataset exploration
* 📈 Road-safety and mobility research

***

## 💻 Local Development

### Clone the Repository

```bash
git clone https://github.com/Sravan94-git/Indian-Vehicle-Detection-and-Classification.git
cd Indian-Vehicle-Detection-and-Classification
```

### Start the Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://localhost:8000` and its interactive documentation at `http://localhost:8000/docs`.

### Start the Frontend

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open the frontend URL shown by Vite, usually `http://localhost:5173`.

***

## 📸 Output

* Upload an image containing one or more vehicles.
* Watch the real-time progress indicator track inference progress from 0% to 100% across the detection and classification pipeline.
* Review the annotated image with bounding boxes and detected vehicle crops.
* Upload a supported video and review the processed result.
* Inspect vehicle labels and dual confidence scores (YOLO detection + classifier).

***

## 🎯 Future Enhancements

* Real-time camera and CCTV stream detection.
* ONNX Runtime integration for further inference speedup.
* Vehicle counting and traffic-density analytics.
* GPS-based traffic mapping.
* Mobile application integration.
* Historical analysis and reporting dashboards.

***

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository, create a feature branch, and submit a pull request.

***

## 📜 License

This project is licensed under the MIT License.

***

## 👨‍💻 Author

**Sravan Kumar Sunkara**
