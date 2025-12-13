import numpy as np
import cv2
import time
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "hand_gesture.tflite"
LABELS_PATH = "labels.txt"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6

# -----------------------------
# LOAD LABELS
# -----------------------------
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# -----------------------------
# LOAD TFLITE MODEL
# -----------------------------
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# CAMERA SETUP
# -----------------------------
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)
picam2.start()
time.sleep(2)

print("📷 Camera started. Looking for hand gestures...")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    frame = picam2.capture_array()

    # Resize and normalize
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    class_id = np.argmax(predictions)
    confidence = predictions[class_id]

    if confidence > CONFIDENCE_THRESHOLD:
        print(f"Gesture: {labels[class_id]}  |  Confidence: {confidence:.2f}")

    time.sleep(0.2)  # limit console spam