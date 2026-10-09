from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from focal_loss import FocalLoss

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 4
EPOCHS = 10
LEARNING_RATE = 0.0001

CLASS_NAMES = ["BG", "Healthy", "WSSV", "WSSV_BG"]

CLASS_TO_INDEX = {
    "BG": 0,
    "Healthy": 1,
    "WSSV": 2,
    "WSSV_BG": 3
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ShrimpDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = row["path"]
        class_name = row["class"]

        label = CLASS_TO_INDEX[class_name]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
train_csv = PROJECT_ROOT / "data" / "splits" / "train_original_only.csv"
val_csv = PROJECT_ROOT / "data" / "splits" / "val.csv"
test_csv = PROJECT_ROOT / "data" / "splits" / "test.csv"

train_dataset = ShrimpDataset(
    train_csv,
    transform=train_transform
)

val_dataset = ShrimpDataset(
    val_csv,
    transform=eval_transform
)

test_dataset = ShrimpDataset(
    test_csv,
    transform=eval_transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images   :", len(train_dataset))
print("Validation images :", len(val_dataset))
print("Test images       :", len(test_dataset))

class EfficientNetB2Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.DEFAULT
        )

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = EfficientNetB2Model(
    num_classes=NUM_CLASSES
).to(device)

print("Device:", device)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
criterion = FocalLoss(
    alpha=1.0,
    gamma=2.0
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "efficientnet_b2_focal_best.pth"

best_val_accuracy = 0.0
for epoch in range(EPOCHS):

    # =========================
    # TRAINING
    # =========================
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = correct / total

    # =========================
    # VALIDATION
    # =========================
    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / val_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    # =========================
    # SAVE BEST MODEL
    # =========================
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

print()
print("Best Validation Accuracy:", f"{best_val_accuracy:.4f}")
print("Best model saved at:")
print(MODEL_PATH)
