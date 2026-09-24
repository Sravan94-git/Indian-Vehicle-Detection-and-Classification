# Vehix API

FastAPI inference backend for the Vehix React client.

## Local development

From the repository root:

```powershell
pip install -r backend/requirements.txt
$env:FRONTEND_ORIGINS = "http://localhost:5173"
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://localhost:8000`, with interactive docs at `/docs`.

## Hugging Face Spaces

Create a Docker Space and upload the contents of this `backend/` folder directly. The included `Dockerfile` starts the API automatically on port `7860`.

Set `FRONTEND_ORIGINS` to the deployed Vercel URL, for example `https://vehix.vercel.app`. The frontend uses the resulting Space URL as its `VITE_API_URL` value in Vercel.
