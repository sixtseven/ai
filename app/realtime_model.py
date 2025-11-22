import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from features import extract_features_from_buf
from PIL import Image
from state import buf
from torchvision import models, transforms
from ultralytics.models import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel("CRITICAL")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolo11n.pt", verbose=False)

TARGET_CLASSES = {"person", "backpack", "handbag", "suitcase"}
CONF_THRESHOLD = 0.5

classNames = model.names if hasattr(model, "names") else list(TARGET_CLASSES)

CLF_PATH = "/home/user/develop/hackatum25/ai/hawaii_frame_clf.joblib"
IMAGE_SIZE = (224, 224)

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_feature_extractor():
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modules = list(backbone.children())[:-1]
    fe = nn.Sequential(*modules)
    fe.eval()
    return fe


def get_embedding_from_frame(frame_bgr, feature_extractor):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = feature_extractor(x)
        feat = feat.view(1, -1)
    return feat.cpu().numpy()[0]


clf = joblib.load(CLF_PATH)
feature_extractor = load_feature_extractor()

last_hawaii_pred = False
last_hawaii_prob = 0.0
frame_count = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    person_count = 0
    luggage_count = 0
    frame_count += 1

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
    if frame_count % 30 == 0:
        emb = get_embedding_from_frame(img, feature_extractor)
        prob = clf.predict_proba([emb])[0]
        last_hawaii_prob = float(prob[1])
        buf.append((person_count, luggage_count, last_hawaii_prob))

    hawaii_text = f"Hawaii: ({last_hawaii_prob:.2f})"

    summary_text = f"Persons: {person_count}  Luggage: {luggage_count}  {hawaii_text}"
    cv2.putText(
        img, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
    )

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break

print("Median Values: ", extract_features_from_buf())

cap.release()
cv2.destroyAllWindows()
