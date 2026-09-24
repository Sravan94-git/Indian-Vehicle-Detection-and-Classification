# Vehix

Vehix is a vehicle detection and classification application with a React frontend and FastAPI inference backend.

```text
frontend/  React + Vite client deployed on Vercel
backend/   FastAPI API deployed as a Docker Space on Hugging Face
backend/weights/   classifier.h5 and yolov8n.pt model files
```

## Hugging Face backend

Create a new **Docker Space** on Hugging Face. Upload the contents of the `backend/` folder into the Space. The `backend/Dockerfile` is designed for this exact folder layout and starts FastAPI on port `7860`.

In the Space settings, add this variable after you deploy the frontend:

```text
FRONTEND_ORIGINS=https://your-vercel-project.vercel.app
```

The API exposes `/health`, `/docs`, `/api/analyze/image`, and `/api/analyze/video`.

## Vercel frontend

Import this repository into Vercel and set **Root Directory** to `frontend`. Vercel detects Vite automatically.

In Vercel, open **Project Settings -> Environment Variables** and add:

```text
VITE_API_URL=https://your-huggingface-space.hf.space
```

The value is the URL of your Hugging Face Space, without a trailing slash. Redeploy Vercel after adding or changing it.

After you know the Vercel URL, copy it into the Hugging Face `FRONTEND_ORIGINS` variable above and restart the Space.

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
