from pathlib import Path
import sys

import torch
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# Import our DataLoader
from data_loader import create_dataloaders


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 3. LOAD DATA
# ============================================================

(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader
) = create_dataloaders()


print("\nTest images:", len(test_dataset))


# ============================================================
# 4. LOAD CONVNEXT-TINY
# ============================================================

model = models.convnext_tiny(
    weights=None
)


# Our model has 4 output classes
num_classes = 4

model.classifier[2] = torch.nn.Linear(
    model.classifier[2].in_features,
    num_classes
)


# ============================================================
# 5. LOAD BEST TRAINED MODEL
# ============================================================

model_path = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "convnext_tiny_baseline_best.pth"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)


model = model.to(device)

model.eval()


print("Best baseline model loaded successfully.")


# ============================================================
# 6. MAKE PREDICTIONS ON TEST SET
# ============================================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)


        # Move results back to CPU
        all_labels.extend(
            labels.tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )


# ============================================================
# 7. CALCULATE METRICS
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


# ============================================================
# 8. PRINT OVERALL RESULTS
# ============================================================

print("\n========== BASELINE TEST RESULTS ==========\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# 9. CLASS-WISE REPORT
# ============================================================

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


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("\n========== CONFUSION MATRIX ==========\n")

print(cm)