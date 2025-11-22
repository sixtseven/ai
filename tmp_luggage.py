from ultralytics import YOLO
import cv2
from collections import deque

buf = deque(maxlen=200)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolo11n.pt")

TARGET_CLASSES = {"person", "backpack", "handbag", "suitcase"}
CONF_THRESHOLD = 0.5  

classNames = model.names if hasattr(model, "names") else list(TARGET_CLASSES)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    person_count = 0
    luggage_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            confidence = float(box.conf[0])
            if confidence < CONF_THRESHOLD:
                continue

            cls = int(box.cls[0])
            if isinstance(classNames, dict):
                label_lookup = classNames.get(cls, "")
            else:
                label_lookup = classNames[cls] if cls < len(classNames) else ""

            if label_lookup not in TARGET_CLASSES:
                continue

            if label_lookup == "person":
                person_count += 1
            elif label_lookup in ("backpack", "handbag", "suitcase"):
                luggage_count += 1

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
            org = (x1, max(y1 - 10, 0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            color = (0, 255, 0)
            thickness = 2
            label_text = f"{label_lookup} {confidence:.2f}"
            cv2.putText(img, label_text, org, font, fontScale, color, thickness)

    print(f"Persons in frame: {person_count} | Luggage items in frame: {luggage_count}")

    summary_text = f"Persons: {person_count}  Luggage: {luggage_count}"
    cv2.putText(img, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    buf.append((person_count, luggage_count))
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
