import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "app"

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torchvision import transforms
from torchvision.models import convnext_small
from ultralytics.models import YOLO
from ultralytics.utils import LOGGER
import threading
import queue
import tkinter as tk

from .features import extract_features_from_buf
from .state import buf

LOGGER.setLevel("CRITICAL")

# ---------------------- Config ----------------------

DEVICE = torch.device("cpu")
MODEL_WEIGHTS_PATH = "convnext_hawaii_binary.pth"

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolo11n.pt", verbose=False)

TARGET_CLASSES = {"person", "backpack", "handbag", "suitcase"}
CONF_THRESHOLD = 0.5

classNames = model.names if hasattr(model, "names") else list(TARGET_CLASSES)

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


def convnext_worker(frame_queue, result_dict, convnext_model):
    """Background thread worker for ConvNeXt Hawaii shirt detection."""
    while True:
        try:
            item = frame_queue.get(timeout=1)
            if item is None:  # Poison pill to stop thread
                break

            frame, person_count, luggage_count = item

            # Run expensive ConvNeXt inference in background
            prob = get_hawaii_prob_from_frame(frame, convnext_model)

            # Update shared result
            result_dict['last_hawaii_prob'] = prob

            # Append to buffer
            buf.append((person_count, luggage_count, prob))
            # print(f"Buffer updated: persons={person_count}, luggage={luggage_count}, hawaii_prob={prob:.4f}, buffer_len={len(buf)}")

            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"ConvNeXt detection error: {e}")
            continue


class VideoApp:
    def __init__(self, root, convnext_model, frame_queue, result_dict):
        self.root = root
        self.root.title("Webcam - Object Detection")
        self.convnext_model = convnext_model
        self.frame_queue = frame_queue
        self.result_dict = result_dict

        # Frame counter and cache
        self.frame_count = 0
        self.person_count = 0
        self.luggage_count = 0
        self.cached_boxes = []

        # Create canvas for video display
        self.canvas = tk.Canvas(root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind resize event
        self.canvas.bind('<Configure>', self.on_resize)

        self.running = True
        self.update_frame()

    def on_resize(self, event):
        # Canvas was resized, next frame will adapt
        pass

    def update_frame(self):
        if not self.running:
            return

        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            self.root.quit()
            return

        self.frame_count += 1

        # Run YOLO only every 3 frames to reduce load
        if self.frame_count % 3 == 0:
            results = model(img, stream=True, verbose=False)

            self.person_count = 0
            self.luggage_count = 0
            self.cached_boxes = []

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
                        self.person_count += 1
                    elif label_lookup in ("backpack", "handbag", "suitcase"):
                        self.luggage_count += 1

                    # Cache box info for reuse
                    self.cached_boxes.append({
                        'coords': (x1, y1, x2, y2),
                        'label': label_lookup,
                        'confidence': confidence
                    })

        # Draw cached boxes on every frame
        for box_info in self.cached_boxes:
            x1, y1, x2, y2 = box_info['coords']
            label_lookup = box_info['label']
            confidence = box_info['confidence']

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
            org = (x1, max(y1 - 10, 0))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            color = (0, 255, 0)
            thickness = 2
            label_text = f"{label_lookup} {confidence:.2f}"
            cv2.putText(img, label_text, org, font, fontScale, color, thickness)

        # Queue frame for background ConvNeXt detection (non-blocking)
        if self.frame_count % 3 == 0:
            try:
                self.frame_queue.put_nowait((img.copy(), self.person_count, self.luggage_count))
            except queue.Full:
                pass  # Skip this frame if queue is full

        # Read last result from background thread
        last_hawaii_prob = self.result_dict['last_hawaii_prob']

        hawaii_text = f"Hawaii: ({'True' if last_hawaii_prob >= 0.99 else 'False'} {last_hawaii_prob:.3f})"
        summary_text = (
            f"Persons: {self.person_count}  Luggage: {self.luggage_count}  {hawaii_text}"
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

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling to fit frame in canvas while maintaining aspect ratio
            frame_h, frame_w = img.shape[:2]
            scale = min(canvas_width / frame_w, canvas_height / frame_h)

            # New frame size
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)

            # Resize frame
            resized = cv2.resize(img, (new_w, new_h))

            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Convert to PPM format (Tkinter's native format)
            h, w = rgb.shape[:2]
            ppm_header = f'P6 {w} {h} 255 '.encode()
            ppm_data = ppm_header + rgb.tobytes()

            photo = tk.PhotoImage(width=w, height=h, data=ppm_data, format='PPM')

            # Calculate position to center the image
            x_pos = (canvas_width - new_w) // 2
            y_pos = (canvas_height - new_h) // 2

            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(x_pos, y_pos, image=photo, anchor=tk.NW)
            self.canvas.image = photo  # Keep a reference

        # Schedule next frame update (30 FPS)
        self.root.after(33, self.update_frame)

    def on_closing(self):
        self.running = False
        self.root.quit()


def run_realtime_model_convnext():
    """Main loop for realtime object detection and ConvNeXt classification using Tkinter GUI."""
    # Clear old buffer data from previous runs
    print("Clearing old buffer data...")
    buf.clear()

    convnext_model = load_convnext_binary_model()

    # Setup background thread for ConvNeXt detection
    frame_queue = queue.Queue(maxsize=2)
    result_dict = {'last_hawaii_prob': 0.0}

    detection_thread = threading.Thread(
        target=convnext_worker,
        args=(frame_queue, result_dict, convnext_model),
        daemon=True
    )
    detection_thread.start()

    # Create Tkinter window
    root = tk.Tk()
    root.geometry("1280x720")  # Initial size, fully resizable

    app = VideoApp(root, convnext_model, frame_queue, result_dict)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    try:
        root.mainloop()
    finally:
        # Cleanup
        frame_queue.put(None)  # Signal thread to stop
        detection_thread.join(timeout=2)

        print("Median Values: ", extract_features_from_buf())

        cap.release()


if __name__ == "__main__":
    run_realtime_model_convnext()
