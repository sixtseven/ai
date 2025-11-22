import argparse
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


# ---------- CONFIG ----------
CLF_PATH = "hawaii_frame_clf.joblib"
IMAGE_SIZE = (224, 224)
# ----------------------------

# Must match training transforms
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_feature_extractor():
    """ResNet18 backbone up to the avgpool (512-dim embedding)."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modules = list(backbone.children())[:-1]  # remove final FC layer
    fe = nn.Sequential(*modules)
    fe.eval()
    return fe


def get_embedding(image_path, feature_extractor):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)   # (1, 3, 224,224)

    with torch.no_grad():
        feat = feature_extractor(x)   # (1, 512,1,1)
        feat = feat.view(1, -1)       # (1, 512)

    return feat.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run Hawaii shirt classifier on an input image."
    )
    parser.add_argument("image_path", type=str, help="Path to test image")
    args = parser.parse_args()

    # Load model
    print("Loading classifier...")
    clf = joblib.load(CLF_PATH)

    # Load feature extractor
    feature_extractor = load_feature_extractor()

    # Extract embedding
    print(f"Processing image: {args.image_path}")
    embedding = get_embedding(args.image_path, feature_extractor)

    # Predict
    prob = clf.predict_proba([embedding])[0]   
    pred = int(np.argmax(prob))

    print("\n===== RESULT =====")
    print(f"Hawaii shirt: {bool(pred)}")
    print(f"Probability: {prob[1]:.3f}")
    print("==================\n")


if __name__ == "__main__":
    main()
