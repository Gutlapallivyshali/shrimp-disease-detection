
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from data_loader import create_dataloaders
from cbam import CBAM


PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "convnext_tiny_cbam_focal_class_weighted_best.pth"
)

CLASS_NAMES = ["BG", "Healthy", "WSSV", "WSSV_BG"]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class ConvNeXtCBAM(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = models.convnext_tiny(weights=None)
        self.cbam = CBAM(channels=768)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


_, _, test_dataset, _, _, test_loader = create_dataloaders()

model = ConvNeXtCBAM(num_classes=4).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(
    all_labels, all_predictions, average="weighted", zero_division=0
)
recall = recall_score(
    all_labels, all_predictions, average="weighted", zero_division=0
)
f1 = f1_score(
    all_labels, all_predictions, average="weighted", zero_division=0
)

print("\n========== CLASS-WEIGHTED MODEL TEST RESULTS ==========")
print("Test images:", len(test_dataset))
print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2, 3]))