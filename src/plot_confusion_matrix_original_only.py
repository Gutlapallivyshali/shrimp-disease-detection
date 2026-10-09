from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.metrics import confusion_matrix

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
# 3. DATASET
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
# 7. LOAD BEST MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

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
# 8. GET PREDICTIONS
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
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:")
print(cm)


# ============================================================
# 10. SAVE FIGURE
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "confusion_matrices"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "convnext_tiny_cbam_focal_original_only.png"
)


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.title(
    "Confusion Matrix - ConvNeXt-Tiny + CBAM + Focal Loss"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved to:"
)

print(
    OUTPUT_PATH
)