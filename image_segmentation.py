import torch
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from ultralytics import YOLO
import numpy as np


LUGGAGE_CLASSES = {"backpack", "handbag", "suitcase"}


def draw_detections(image: Image.Image, boxes, classes, scores, class_names, score_thresh: float = 0.25):
    """
    Draw bounding boxes and labels on the input image for the selected detections.

    image: PIL.Image
    boxes: (N, 4) xyxy tensor/array
    classes: (N,) class indices
    scores: (N,) confidence scores
    class_names: list of class names indexed by class id
    returns: annotated PIL.Image
    """
    image_np = np.array(image)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_np)
    ax.axis("off")

    for box, cls_id, score in zip(boxes, classes, scores):
        cls_id = int(cls_id)
        label = class_names[cls_id]

        if label not in LUGGAGE_CLASSES:
            continue

        if score < score_thresh:
            continue

        x1, y1, x2, y2 = box

        cmap = matplotlib.colormaps.get_cmap("tab10")
        color = cmap(cls_id % 10)

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        caption = f"{label} {score:.2f}"
        ax.text(
            x1,
            y1 - 2,
            caption,
            fontsize=10,
            color="white",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.7, pad=2),
        )

    fig.canvas.draw()
    annotated = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated = annotated.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(annotated)


def run_yolo(image_path: str, output_path: str):
    model = YOLO("yolo11n.pt")
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print("Running YOLO inference...")
    results = model(image)[0]

    boxes = results.boxes.xyxy.cpu().numpy()  
    classes = results.boxes.cls.cpu().numpy()  
    scores = results.boxes.conf.cpu().numpy()

    class_names = model.names  

    print(f"Total detections: {len(boxes)}")
    luggage_count = sum(
        1
        for cls_id, score in zip(classes, scores)
        if class_names[int(cls_id)] in LUGGAGE_CLASSES and score >= 0.25
    )
    print(f"Luggage-related detections (backpack/handbag/suitcase): {luggage_count}")

    annotated_image = draw_detections(image, boxes, classes, scores, class_names)

    annotated_image.save(output_path)
    print(f"Saved annotated image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO luggage detector with bounding boxes.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="output_yolo.png", help="Path to save output image")

    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    run_yolo(args.image_path, args.output)


if __name__ == "__main__":
    main()
