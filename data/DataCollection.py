import cv2
import os
import time

# ==============================
# Configuration
# ==============================
TOTAL_IMAGES = 200
CAMERA_INDEX = 0
SAVE_DELAY = 0.1  # Seconds between shots (adjust if too fast)

# ==============================
# Setup
# ==============================
label_name = input("Enter the label name (e.g., 'fist', 'like'): ").strip()
if not label_name:
    print("Error: Label name cannot be empty.")
    exit()

# Create the folder if it doesn't exist
save_path = os.path.join("dataset", label_name)
os.makedirs(save_path, exist_ok=True)

print(f"\n--- DATA COLLECTION FOR: '{label_name}' ---")
print(f"Saving images to: {save_path}")
print("1. Press 's' to START capturing 200 images.")
print("2. Press 'q' to QUIT at any time.")
print("-------------------------------------------\n")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera.")

count = 0
capturing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror view (easier to position hand)
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # ------------------------------
    # Capturing Logic
    # ------------------------------
    if capturing:
        # Save the raw frame (clean, no text)
        filename = f"{label_name}_{count}.jpg"
        filepath = os.path.join(save_path, filename)
        cv2.imwrite(filepath, frame)
        
        print(f"Saved {count+1}/{TOTAL_IMAGES}: {filename}")
        count += 1
        time.sleep(SAVE_DELAY) # Small pause to allow movement

        # Visual indicator (Red circle while recording)
        cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)

        # Stop automatically after target reached
        if count >= TOTAL_IMAGES:
            print(f"\n✅ SUCCESS: Collected {TOTAL_IMAGES} images for '{label_name}'!")
            capturing = False
            # Optional: Break loop immediately or wait for user to quit
            # break 

    # ------------------------------
    # UI Text
    # ------------------------------
    cv2.putText(display_frame, f"Label: {label_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Count: {count}/{TOTAL_IMAGES}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if not capturing and count < TOTAL_IMAGES:
        cv2.putText(display_frame, "Press 's' to START", (10, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Data Collector", display_frame)

    # ------------------------------
    # Controls
    # ------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not capturing:
        capturing = True
        count = 0 # Reset count if you want to overwrite or restart

cap.release()
cv2.destroyAllWindows()
