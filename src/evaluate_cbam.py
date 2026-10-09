from pathlib import Path
import sys

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from torchvision import models

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import create_dataloaders
from cbam import CBAM


class ConvNeXtCBAM(torch.nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.backbone = models.convnext_tiny(
            weights=None
        )

        self.cbam = CBAM(
            channels=768
        )

        self.classifier = torch.nn.Linear(
            768,
            num_classes
        )

    def forward(self, x):

        x = self.backbone.features(x)

        x = self.cbam(x)

        x = self.backbone.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        return x


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader
) = create_dataloaders()

print("\nTest images:", len(test_dataset))

model = ConvNeXtCBAM(
    num_classes=4
)

model_path = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "convnext_tiny_cbam_focal_reproduced_best.pth"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Best CBAM + Focal Loss model loaded successfully.")

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


print("\n========== CBAM + FOCAL TEST RESULTS ==========\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


class_names = [
    "BG",
    "Healthy",
    "WSSV",
    "WSSV_BG"
]


print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n========== CONFUSION MATRIX ==========\n")

print(cm)