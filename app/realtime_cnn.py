import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "app"

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small
from ultralytics.models import YOLO
from ultralytics.utils import LOGGER

from .features import extract_features_from_buf
from .state import buf

LOGGER.setLevel("CRITICAL")

# ---------------------- Config ----------------------

DEVICE = torch.device("cpu")
MODEL_WEIGHTS_PATH = "convnext_hawaii_binary.pth"

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

yolo_model = YOLO("yolo11n.pt", verbose=False)

TARGET_CLASSES = {"person", "backpack", "handbag", "suitcase"}
CONF_THRESHOLD = 0.5

classNames = yolo_model.names if hasattr(yolo_model, "names") else list(TARGET_CLASSES)

IMAGE_SIZE = (224, 224)

convnext_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_convnext_binary_model() -> nn.Module:
    """
    Load ConvNeXt-small with the binary classification head,
    then load your trained weights from disk.
    """
    model = convnext_small(weights="DEFAULT")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)

    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def get_hawaii_prob_from_frame(frame_bgr: np.ndarray, model: nn.Module) -> float:
    """
    Take a BGR frame (OpenCV), run it through ConvNeXt, and return P(hawaii==1).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    x = convnext_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].item()

    return float(prob)


def run_realtime_model_convnext():
    """Main loop for realtime object detection and ConvNeXt classification."""
    model = load_convnext_binary_model()

    last_hawaii_prob = 0.0
    frame_count = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        results = yolo_model(img, stream=True)

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

        if frame_count % 8 == 0:
            prob = get_hawaii_prob_from_frame(img, model)
            last_hawaii_prob = prob
            buf.append((person_count, luggage_count, last_hawaii_prob))

        hawaii_text = f"Hawaii: ({'True' if last_hawaii_prob >= 0.99 else 'False'} {last_hawaii_prob:.3f})"
        summary_text = (
            f"Persons: {person_count}  Luggage: {luggage_count}  {hawaii_text}"
        )
        cv2.putText(
            img,
            summary_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) == ord("q"):
            break

    print("Median Values: ", extract_features_from_buf())

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_model_convnext()
