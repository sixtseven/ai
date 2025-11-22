#!/usr/bin/env python3
"""
Process a video: detect people and luggage (using Faster R-CNN), and refine masks with SAM 1.

Notes:
- Requires a SAM checkpoint passed via --sam-checkpoint (e.g., sam_vit_h_4b8939.pth).
- Uses a torchvision pretrained Faster R-CNN COCO model for initial detection.
- This version uses the public SAM 1 architecture for maximum accessibility.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
import torchvision
import torchvision.transforms.functional as TF

# --- SAM 1 Imports ---
try:
    # We use the original SAM 1 imports
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    raise ImportError(
        "Failed to import segment_anything. Please run: pip install segment-anything"
    ) from e
# ---------------------

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'child', 'man', 'woman', 'luggage'
    # so we stick to standard detection classes like 'person', 'backpack', 'handbag', 'suitcase'
]

TARGET_CLASSES = ('person', 'backpack', 'handbag', 'suitcase', 'child', 'man', 'woman', 'luggage')


def load_detector(device: torch.device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)
    model.eval()
    return model


def dominant_color_from_mask(image: np.ndarray, mask: np.ndarray, k: int = 2) -> List[int]:
    """Return dominant RGB color inside mask as [R,G,B]."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return [0, 0, 0]
    
    pixels = img_rgb[ys, xs]
    if len(pixels) < k:
        mean = pixels.mean(axis=0)
        return [int(x) for x in mean.tolist()]
        
    km = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(pixels)
    counts = np.bincount(km.labels_)
    dominant = km.cluster_centers_[counts.argmax()]
    return [int(x) for x in dominant.tolist()]


def process_video(video_path: str, sam_checkpoint: str, out_json: str, vis_out: str = None, frame_skip: int = 5, device: str = None, sam_model_name: str = "vit_h"):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load models
    print(f"Loading detector on {device}...")
    detector = load_detector(device)

    predictor = None
    if sam_checkpoint:
        print(f"Loading SAM model '{sam_model_name}' from checkpoint...")
        # SAM 1 Loading: Uses sam_model_registry
        try:
            sam = sam_model_registry[sam_model_name](checkpoint=sam_checkpoint)
            sam.to(device)
            predictor = SamPredictor(sam)
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            print("Proceeding with bbox-based masks (no SAM refinement).")
    else:
        print("No SAM checkpoint provided; proceeding with bbox-based masks (no SAM refinement).")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if vis_out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(vis_out, fourcc, fps / max(1, frame_skip), (width, height))

    results = {
        'video': str(video_path),
        'frames': []
    }

    frame_idx = 0
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        print(f"Processing frame {frame_idx}...")
        img = frame.copy()

        # Detector input preparation
        img_tensor = TF.to_tensor(Image.fromarray(img[:, :, ::-1])).to(device)

        with torch.no_grad():
            outputs = detector([img_tensor])[0]

        boxes = outputs['boxes'].detach().cpu().numpy()
        labels = outputs['labels'].detach().cpu().numpy()
        scores = outputs['scores'].detach().cpu().numpy()

        # If SAM is available, set SAM image once per frame
        if predictor is not None:
            # SAM expects RGB image
            predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

        frame_info = {'frame_index': frame_idx, 'people': [], 'bags': []}
        
        # Overlay image (for drawing)
        overlay_img = img.copy()

        for box, label, score in zip(boxes, labels, scores):
            if score < 0.5:
                continue
            
            # Ensure label index is valid
            if int(label) >= len(COCO_INSTANCE_CATEGORY_NAMES):
                continue
            
            cls_name = COCO_INSTANCE_CATEGORY_NAMES[int(label)]
            
            if cls_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = [int(x) for x in box]
            # Clip box coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            
            # --- SAM Refinement (Visual Prompt: BBox) ---
            mask = np.zeros((height, width), dtype=bool)
            if predictor is not None:
                # SAM expects box as [x1, y1, x2, y2]
                masks, scores_mask, logits = predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
                mask = masks[0]
            else:
                # Fallback: create a mask from the bounding box area
                mask[y1:y2, x1:x2] = True
            
            # --- Store Data ---
            
            if cls_name == 'person':
                color = dominant_color_from_mask(frame, mask)
                person_entry = {
                    'bbox': [x1, y1, x2, y2],
                    'score': float(score),
                    'mask_area': int(mask.sum()),
                    'dominant_color_rgb': color,
                    'accessories': []
                }
                frame_info['people'].append(person_entry)
                color_box = (0, 255, 0) # Green for Person
                color_mask = (0, 255, 0)
            else: # Luggage
                bag_entry = {
                    'type': cls_name,
                    'bbox': [x1, y1, x2, y2],
                    'score': float(score),
                    'mask_area': int(mask.sum())
                }
                frame_info['bags'].append(bag_entry)
                color_box = (255, 0, 0) # Blue for Luggage
                color_mask = (255, 0, 0)
            
            # --- Visualization Overlay ---
            if writer is not None:
                # Draw box
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color_box, 2)
                
                # Blend mask
                colored_mask = np.zeros_like(img)
                colored_mask[mask] = color_mask 
                alpha = 0.4
                cv2.addWeighted(colored_mask, alpha, overlay_img, 1 - alpha, 0, overlay_img)
                
                # Label
                cv2.putText(overlay_img, f"{cls_name} {score:.2f}", (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        # --- Association and Grouping Logic (Simplified) ---
        # Simple association: check if the bag box overlaps a person box (IoU > 0)
        from torchvision.ops import box_iou
        
        person_boxes = [p['bbox'] for p in frame_info['people']]
        if person_boxes and frame_info['bags']:
            person_boxes_tensor = torch.tensor(person_boxes, dtype=torch.float32)
            bag_boxes_tensor = torch.tensor([b['bbox'] for b in frame_info['bags']], dtype=torch.float32)

            iou_matrix = box_iou(person_boxes_tensor, bag_boxes_tensor) # Person (rows) vs Bag (cols)

            for i, p_entry in enumerate(frame_info['people']):
                for j, b_entry in enumerate(frame_info['bags']):
                    if iou_matrix[i, j] > 0.05: # Minimal overlap to count as associated
                        p_entry['accessories'].append({
                            'bag_type': b_entry['type'],
                            'bag_bbox': b_entry['bbox'],
                            'bag_score': b_entry['score']
                        })
        
        # Simple grouping heuristic (remains the same)
        people = frame_info['people']
        groups = []
        centroids = []
        for p in people:
            x1, y1, x2, y2 = p['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centroids.append((cx, cy))

        assigned = [False] * len(centroids)
        for i, c in enumerate(centroids):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            for j, d in enumerate(centroids):
                if assigned[j]:
                    continue
                dist = np.linalg.norm(np.array(c) - np.array(d))
                diag = np.sqrt(width * width + height * height)
                if dist < diag * 0.08:  # Heuristic
                    group.append(j)
                    assigned[j] = True
            groups.append(group)

        # attach groups
        for g in groups:
            members = [people[i] for i in g]
            for m in members:
                m['group_size'] = len(g)

        # Write the final frame image
        if writer is not None:
             writer.write(overlay_img)
             
        results['frames'].append(frame_info)
        processed_frames += 1
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    max_bags = max((len(f['bags']) for f in results['frames']), default=0)
    aggregate = {
        'max_bags_in_frame': int(max_bags),
        'frames_processed': processed_frames
    }

    out = {'aggregate': aggregate, 'frames': results['frames']}
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Done. Wrote report to {out_json}. Processed {processed_frames} frames.")


def main():
    parser = argparse.ArgumentParser(description='Process video with detector + SAM to segment people and luggage')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--sam-checkpoint', required=True, help='Path to SAM 1 checkpoint (.pth), e.g., sam_vit_h_4b8939.pth')
    parser.add_argument('--out', required=True, help='Output JSON report path')
    parser.add_argument('--vis', required=False, help='Visualization video output path (optional)')
    parser.add_argument('--frame-skip', type=int, default=5, help='Process every N-th frame to speed up')
    parser.add_argument('--device', default=None, help='Torch device string, e.g. cuda:0 or cpu')
    
    # You can change the model name if you use a different checkpoint (e.g., vit_l, vit_b)
    parser.add_argument('--sam-model-name', default="vit_h", help='SAM model name from the registry (e.g., vit_h, vit_l, vit_b)')
    
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not Path(args.sam_checkpoint).exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    process_video(args.video, args.sam_checkpoint, args.out, vis_out=args.vis, frame_skip=args.frame_skip, device=args.device, sam_model_name=args.sam_model_name)


if __name__ == '__main__':
    main()