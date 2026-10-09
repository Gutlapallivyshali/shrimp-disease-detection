from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from cbam import CBAM


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path.home()
    / "Documents"
    / "shrimp-disease-detection"
)

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 4

CLASS_NAMES = [
    "BG",
    "Healthy",
    "WSSV",
    "WSSV_BG"
]

CLASS_TO_INDEX = {
    "BG": 0,
    "Healthy": 1,
    "WSSV": 2,
    "WSSV_BG": 3
}

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# 3. TEST DATASET
# ============================================================

class ShrimpDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.data = pd.read_csv(
            csv_file
        )

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["path"]

        class_name = row["class"]

        label = CLASS_TO_INDEX[
            class_name
        ]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# 4. TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# ============================================================
# 5. LOAD TEST DATA
# ============================================================

test_csv = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

test_dataset = ShrimpDataset(
    test_csv,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# 6. MODEL
# ============================================================

class ConvNeXtCBAM(nn.Module):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()

        self.backbone = models.convnext_tiny(
            weights=None
        )

        self.cbam = CBAM(
            channels=768
        )

        self.classifier = nn.Linear(
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


# ============================================================
# 7. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# 8. LOAD BEST MODEL
# ============================================================

model = ConvNeXtCBAM(
    num_classes=NUM_CLASSES
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "convnext_tiny_cbam_focal_original_only_best.pth"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# 9. PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(
            images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# 10. OVERALL METRICS
# ============================================================

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


print("\n========== TEST RESULTS ==========\n")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print(
    "\n========== CLASSIFICATION REPORT ==========\n"
)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(
    "\n========== CONFUSION MATRIX ==========\n"
)

print(cm)

# ============================================================
# 13. SAVE FINAL EVALUATION RESULTS
# ============================================================

import json
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CM_DIR = OUTPUT_DIR / "confusion_matrices"
PLOTS_DIR = OUTPUT_DIR / "plots"

CM_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Save metrics and class-wise report
metrics = {
    "test_images": len(test_dataset),
    "accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1_score": float(f1),
}

with open(OUTPUT_DIR / "final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

# Save final confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - ConvNeXt-Tiny + CBAM + Focal Loss")
plt.tight_layout()

plt.savefig(
    CM_DIR / "convnext_tiny_cbam_focal_original_only.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nFinal metrics and confusion matrix saved successfully.")