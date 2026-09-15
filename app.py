import os
from pathlib import Path
from secrets import token_urlsafe

from flask import Flask, render_template, request, redirect, url_for, flash
from ultralytics import YOLO
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import InputLayer, MultiHeadAttention
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import uuid
import time
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', token_urlsafe(32)),
    MAX_CONTENT_LENGTH=100 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Configuration
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
RESULT_FOLDER = BASE_DIR / 'static' / 'results'
CROP_FOLDER = BASE_DIR / 'static' / 'crops'
for folder in (UPLOAD_FOLDER, RESULT_FOLDER, CROP_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# Load models
class CompatibleInputLayer(InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs['batch_size'] = batch_shape[0]
            kwargs['input_shape'] = batch_shape[1:]
        super().__init__(**kwargs)


class CompatibleMultiHeadAttention(MultiHeadAttention):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)


keras.saving.get_custom_objects().update({
    'swish': swish,
    'InputLayer': CompatibleInputLayer,
    'MultiHeadAttention': CompatibleMultiHeadAttention,
})


try:
    yolo_model = YOLO(str(BASE_DIR / 'weights' / 'yolov8n.pt'))
    classifier = load_model(
        BASE_DIR / 'weights' / 'classifier.h5',
        compile=False,
        custom_objects={
            'swish': swish,
            'InputLayer': CompatibleInputLayer,
            'MultiHeadAttention': CompatibleMultiHeadAttention,
        },
    )
except Exception as e:
    raise RuntimeError(f"Failed to load models: {str(e)}")

CLASS_NAMES = [
    'Motorized2wheeler', 'ambasador_taxi', 'autorickshaw', 'bicycle',
    'bus', 'car', 'minitruck', 'motarvan', 'rickshaw', 'toto', 'truck', 'van'
]


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    # Check if it's an image or video
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png']:
        return process_image(file)
    elif file_ext in ['.mp4', '.avi', '.mov']:
        return process_video(file)
    else:
        flash('Unsupported file format. Please upload an image (JPEG, PNG) or video (MP4, AVI, MOV).', 'error')
        return redirect(url_for('index'))

def process_image(file):
    try:
        # Save uploaded image
        filename = secure_filename(f"{uuid.uuid4().hex}.jpg")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process image
        image = cv2.imread(str(filepath))
        if image is None:
            flash('Failed to read the uploaded image', 'error')
            return redirect(url_for('index'))

        # Detect vehicles
        detections = yolo_model(image)[0]
        vehicle_results = []
        for i, box in enumerate(detections.boxes):
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                det_conf = float(box.conf[0].item())
                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Save cropped vehicle
                crop_filename = f"crop_{i}_{filename}"
                crop_path = CROP_FOLDER / crop_filename
                cv2.imwrite(crop_path, crop)

                # Classify vehicle
                resized = cv2.resize(crop, (128, 128))
                norm_img = resized.astype("float32") / 255.0
                prediction = classifier.predict(np.expand_dims(norm_img, axis=0))[0]
                class_id = int(np.argmax(prediction))
                cls_conf = float(np.max(prediction))

                vehicle_results.append({
                    'number': i+1,
                    'label': CLASS_NAMES[class_id],
                    'classification_confidence': round(cls_conf * 100, 1),
                    'detection_confidence': round(det_conf * 100, 1),
                    'crop_image': crop_filename,
                    'position': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                })

            except Exception as e:
                print(f"Error processing box {i}: {str(e)}")
                continue

        # Sort by detection confidence (highest first)
        vehicle_results.sort(key=lambda x: x['detection_confidence'], reverse=True)

        # Draw bounding boxes on output image
        output_image = image.copy()
        for vehicle in vehicle_results:
            pos = vehicle['position']
            cv2.rectangle(output_image, (pos['x1'], pos['y1']), (pos['x2'], pos['y2']), (0, 255, 0), 2)
            label = f"{vehicle['number']}. {vehicle['label']} {vehicle['detection_confidence']}%"
            cv2.putText(output_image, label, (pos['x1'], pos['y1']-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save output image
        output_filename = f"result_{filename}"
        output_path = RESULT_FOLDER / output_filename
        cv2.imwrite(output_path, output_image)

        return render_template('image_result.html',
                            original_image=filename,
                            output_image=output_filename,
                            vehicles=vehicle_results)

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        flash('An error occurred while processing the image', 'error')
        return redirect(url_for('index'))

def process_video(file):
    try:
        # Save uploaded video
        filename = f"{uuid.uuid4().hex}.mp4"
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)

        # Open video
        cap = cv2.VideoCapture(str(filepath))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        # Output video
        result_filename = f"{uuid.uuid4().hex}.mp4"
        result_path = RESULT_FOLDER / result_filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(result_path), fourcc, frame_rate, (frame_width, frame_height))

        # Performance tracking
        total_frames = 0
        processed_frames = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            annotated_frame = frame.copy()

            # Run YOLO detection
            results = yolo_model(frame)[0]
            detected = False

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                try:
                    # Preprocess for classification
                    resized = cv2.resize(crop, (128, 128))
                    array = img_to_array(resized)
                    array = preprocess_input(array)
                    array = np.expand_dims(array, axis=0)

                    # Predict class
                    pred = classifier.predict(array)
                    class_id = np.argmax(pred[0])
                    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "Unknown"

                    # Annotate frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    detected = True

                except Exception as e:
                    print(f"Error during classification: {e}")
                    continue

            if detected:
                processed_frames += 1

            out.write(annotated_frame)

        # Cleanup
        cap.release()
        out.release()
        end_time = time.time()

        # Performance results
        elapsed_time = end_time - start_time
        fps = processed_frames / elapsed_time if elapsed_time > 0 else 0

        print(f"Total frames read: {total_frames}")
        print(f"Frames processed (with detections): {processed_frames}")
        print(f"Effective Processing FPS: {fps:.2f}")

        return render_template('video_result.html',
                     result_video=url_for('static', filename=f'results/{result_filename}'),
                             fps=fps,
                             processed_frames=processed_frames,
                             elapsed_time=elapsed_time)

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        flash('An error occurred while processing the video', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG') == '1',
    )