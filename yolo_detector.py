import os
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('weights/vehicle_detection_model.pt')  # Adjust if needed
model.conf = 0.4  # Set confidence threshold

# Class names list (make sure it's in the same order as your training)
CLASS_NAMES = [
    'Motorized2wheeler', 'ambasador_taxi', 'autorickshaw', 'bicycle',
    'bus', 'car', 'minitruck', 'motarvan', 'rickshaw', 'toto', 'truck', 'van'
]

def detect_vehicles(image_path):
    image = cv2.imread(image_path)
    results = model(image)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': conf,
            'class_id': cls_id
        })

        # Draw bounding box and label on image
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Save the annotated image
    os.makedirs('static/results', exist_ok=True)
    cv2.imwrite('static/results/output.jpg', image)

    return detections
