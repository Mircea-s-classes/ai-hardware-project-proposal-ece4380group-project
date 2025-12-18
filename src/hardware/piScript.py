import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# ==============================
# Configuration
# ==============================
MODEL_PATH = "good_gesture_model.tflite"

CLASSES = [
    "fist",
    "like",
    "no_gesture",
    "palm",
    "point"
]

CAMERA_INDEX = 0

# Load model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]["shape"][1]
input_width = input_details[0]["shape"][2]

# ==============================
# Open USB Camera
# ==============================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open USB camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image to look better
    frame = cv2.flip(frame, 1)

    # Preprocess
    img = cv2.resize(frame, (input_width, input_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    class_id = np.argmax(output)
    confidence = output[class_id]
    label = CLASSES[class_id]

    # Display results
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()