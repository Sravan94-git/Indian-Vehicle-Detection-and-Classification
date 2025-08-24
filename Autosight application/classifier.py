import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import swish


# Load model with custom swish activation
model = load_model('weights/classifier.h5', custom_objects={'swish': swish})

# Vehicle class names (ensure order matches training)
CLASS_NAMES = [
    'Motorized2wheeler', 'ambasador_taxi', 'autorickshaw', 'bicycle',
    'bus', 'car', 'minitruck', 'motarvan', 'rickshaw', 'toto', 'truck', 'van'
]

def classify_vehicle(cropped_image):
    # Resize and normalize image
    resized = cv2.resize(cropped_image, (224, 224))  # Adjust if input size differs
    input_tensor = np.expand_dims(resized / 255.0, axis=0)  # Normalize and batchify

    # Predict class probabilities
    predictions = model.predict(input_tensor)
    class_id = np.argmax(predictions, axis=-1)[0]
    confidence = float(np.max(predictions))

    class_name = CLASS_NAMES[class_id]
    return class_name, confidence
