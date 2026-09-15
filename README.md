# AutoSight

AutoSight is a Flask web application that detects vehicles with YOLOv8 and classifies each detected crop with a TensorFlow model. It accepts JPG, JPEG, and PNG images, plus MP4, AVI, and MOV videos.

The interface includes image and video upload modes, responsive result reports, confidence metrics, annotated media, and a four-step methodology: Capture, Locate, Identify, and Understand.

## Processing methodology

1. **Capture**: An uploaded image or video is saved with a generated UUID filename.
2. **Locate**: YOLOv8 scans the image or each video frame for vehicle regions.
3. **Identify**: Each detected crop is resized and passed to the TensorFlow classifier.
4. **Understand**: AutoSight returns annotated media, vehicle labels, confidence scores, and positions.

The application loads the models from `weights/` when the server starts, so model files must be available before deployment.

## Project structure

```text
app.py                 Flask entry point and inference pipeline
requirements.txt       Python dependencies
Procfile               Gunicorn start command
render.yaml            Render Blueprint configuration
.python-version        Python runtime version
templates/             Upload and result views
static/                Temporary uploads, crops, and generated results
weights/               YOLO and TensorFlow model files
README.md              Project documentation
```

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SECRET_KEY = "local-development-secret"
python app.py
```

Open `http://127.0.0.1:5000`.

## Render deployment

This repository includes `render.yaml`, `Procfile`, and `.python-version`. On Render, create a Blueprint from the repository. Render will install `requirements.txt`, generate `SECRET_KEY`, and start the app with Gunicorn.

The configured production command is:

```text
gunicorn --workers 1 --threads 2 --timeout 300 app:app
```

One worker is intentional because the TensorFlow and PyTorch models are loaded into memory at startup. Set the Render service health check path to `/` if a health check is configured.

The model files must be present under `weights/`:

- `classifier.h5`
- `yolov8n.pt`

The application uses an ephemeral filesystem on Render. Uploaded files and generated results are temporary and may disappear when the service restarts or redeploys. Use object storage if results need to persist.

## Security defaults

- `SECRET_KEY` is read from the environment.
- Debug mode is disabled unless `FLASK_DEBUG=1` is explicitly set.
- Upload requests are limited to 100 MB.
- Upload filenames are replaced with UUID-based names.
- Session cookies are HTTP-only and use `SameSite=Lax`.
- Responses include `nosniff`, `SAMEORIGIN`, and strict referrer-policy headers.
- Generated media and local secrets are excluded by `.gitignore`.

## Runtime notes

- Debug mode is never enabled by the Render start command.
- Render's filesystem is ephemeral. Generated uploads, crops, and result files are temporary.
- For persistent reports, store generated media in object storage and save only the URLs in a database.
- The included `.venv` is for local development only and is excluded from version control.
