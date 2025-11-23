from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import convnext_small
from tqdm import tqdm

device = torch.device("cpu")

data_dir = "data/train"
batch_size = 16
num_epochs = 10
learning_rate = 1e-4
train_split = 0.8
seed = 42

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

base_dataset = datasets.ImageFolder(data_dir)

dataset_size = len(base_dataset)
train_size = int(train_split * dataset_size)
val_size = dataset_size - train_size

generator = torch.Generator().manual_seed(seed)
train_indices, val_indices = torch.utils.data.random_split(
    range(dataset_size),
    [train_size, val_size],
    generator=generator,
)


class TransformSubset(Dataset):
    """
    Wraps a base dataset + subset of indices + a specific transform.
    This lets train and val use different transforms on the same underlying files.
    """

    def __init__(self, base_dataset: Dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, label = self.base_dataset[base_idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


train_dataset = TransformSubset(
    base_dataset, train_indices, transform=heavy_train_transform
)
val_dataset = TransformSubset(base_dataset, val_indices, transform=val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)


model = convnext_small(weights="DEFAULT")
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)


model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


def train_one_epoch(epoch: int) -> float:
    model.train()
    epoch_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def validate():
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return val_loss / len(val_loader), accuracy


for epoch in range(num_epochs):
    train_loss = train_one_epoch(epoch)
    val_loss, val_acc = validate()

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_acc:.4f}")

torch.save(model.state_dict(), "convnext_hawaii_binary.pth")
print("Model saved as convnext_hawaii_binary.pth")
