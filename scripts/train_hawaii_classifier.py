import os
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_ROOT = "./data/train"
OUTPUT_CLF_PATH = "hawaii_frame_clf.joblib"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_images_and_labels(data_root: str):
    X, y = [], []
    root = Path(data_root)
    for label_str in ["0", "1"]:
        class_dir = root / label_str
        if not class_dir.is_dir():
            continue
        label = int(label_str)
        for img_path in class_dir.glob("*.*"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Could not open {img_path}: {e}")
                continue
            X.append(img)
            y.append(label)
    y = np.array(y)
    print(f"Loaded {len(X)} images, label counts:", np.bincount(y))
    return X, y

def build_feature_extractor():
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modules = list(backbone.children())[:-1]  
    fe = nn.Sequential(*modules)
    fe.eval()
    return fe

def extract_embeddings(images, fe, device="cpu"):
    fe.to(device)
    embs = []
    with torch.no_grad():
        for img in images:
            x = transform(img).unsqueeze(0).to(device)
            f = fe(x)               # (1, 512, 1, 1)
            f = f.view(1, -1)       # (1, 512)
            embs.append(f.cpu().numpy()[0])
    return np.stack(embs, axis=0)   # (N, 512)

def main():
    images, labels = load_images_and_labels(DATA_ROOT)
    fe = build_feature_extractor()
    X = extract_embeddings(images, fe, device="cpu")

    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, stratify=labels, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Validation report:")
    print(classification_report(y_val, y_pred, digits=3))

    joblib.dump(clf, OUTPUT_CLF_PATH)
    print(f"Saved classifier to {OUTPUT_CLF_PATH}")

if __name__ == "__main__":
    main()
