import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from torchvision.utils import save_image

DATA_ROOT = "./data/train"
OUTPUT_CLF_PATH = "hawaii_frame_clf.joblib"


def denormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return t * std + mean


def log_transformed_image(img, tfm, out_path):
    x = tfm(img)  # (C,H,W)
    x = denormalize(x)
    x = torch.clamp(x, 0, 1)
    save_image(x, out_path)


val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# transform used for TRAINING only
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

heavy_train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomErasing(p=0.5),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


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


def extract_embeddings(
    images, fe, device: str = "cpu", tfm: transforms.Compose | None = None
):
    """
    Extract embeddings for a list of PIL images using the given feature extractor and transform.
    `tfm` controls whether we use train-time augmentation or plain validation transforms.
    """
    if tfm is None:
        tfm = val_transform

    fe.to(device)
    embs = []
    with torch.no_grad():
        for img in images:
            x = tfm(img).unsqueeze(0).to(device)
            f = fe(x)  # (1, 512, 1, 1)
            f = f.view(1, -1)  # (1, 512)
            embs.append(f.cpu().numpy()[0])
    return np.stack(embs, axis=0)  # (N, 512)


def main():
    images, labels = load_images_and_labels(DATA_ROOT)
    fe = build_feature_extractor()

    X_train_imgs, X_val_imgs, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    X_train = extract_embeddings(
        X_train_imgs, fe, device="cpu", tfm=heavy_train_transform
    )
    X_val = extract_embeddings(X_val_imgs, fe, device="cpu", tfm=val_transform)

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    threshold = 0.6
    y_pred = (val_probs >= threshold).astype(int)

    os.makedirs("debug_augmented", exist_ok=True)
    print("Saving augmented samples to ./debug_augmented")

    for i, img in enumerate(X_train_imgs[:5]):
        log_transformed_image(
            img, heavy_train_transform, f"debug_augmented/aug_{i}.png"
        )

    print(f"Validation report (threshold = {threshold}):")
    print(classification_report(y_val, y_pred, digits=3))

    joblib.dump(clf, OUTPUT_CLF_PATH)
    print(f"Saved classifier to {OUTPUT_CLF_PATH}")


if __name__ == "__main__":
    main()
