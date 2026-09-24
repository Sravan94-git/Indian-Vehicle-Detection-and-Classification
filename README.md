# Vehix

Vehix is a vehicle detection and classification application with a React frontend and FastAPI inference backend.

```text
frontend/  React + Vite client deployed on Vercel
backend/   FastAPI API deployed as a Docker Space on Hugging Face
backend/weights/   classifier.h5 and yolov8n.pt model files
```

## Render backend

Create a new **Web Service** on Render and connect this repository. Set **Root Directory** to `backend`, choose **Docker** as the runtime, and deploy. Render will use the included `Dockerfile` and its `PORT` variable.

In the Render service's environment variables, add this after you deploy the frontend:

```text
FRONTEND_ORIGINS=https://your-frontend-domain.vercel.app
```

The API exposes `/health`, `/docs`, `/api/analyze/image`, and `/api/analyze/video`. Keep the Render service on a plan that allows enough memory for TensorFlow and YOLO models.

## Vercel frontend

Import this repository into Vercel and set **Root Directory** to `frontend`. Vercel detects Vite automatically.

In Vercel, open **Project Settings -> Environment Variables** and add:

```text
VITE_API_URL=https://your-render-service.onrender.com
```

The value is the URL of your Render service, without a trailing slash. Redeploy Vercel after adding or changing it.

After you know the Vercel URL, copy it into Render's `FRONTEND_ORIGINS` variable above and restart the service.

## Local development

Start the backend locally from the repository root:

```powershell
pip install -r backend/requirements.txt
$env:FRONTEND_ORIGINS = "http://localhost:5173"
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
Set-Location frontend
npm install
Copy-Item .env.example .env
npm run dev
```
