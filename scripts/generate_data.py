import os
import time

import cv2

SAVE_DIR = "captured_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_counter = 0
save_every_n_frames = 30

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_counter += 1

    if frame_counter % save_every_n_frames == 0:
        timestamp = int(time.time())
        filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
