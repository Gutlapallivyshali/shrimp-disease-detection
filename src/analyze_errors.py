from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models

from data_loader import create_dataloaders
from cbam import CBAM


# ==================================================
# 1. PROJECT PATH
# ==================================================

PROJECT_ROOT = (
    Path.home()
    / "Documents"
    / "shrimp-disease-detection"
)

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


# ==================================================
# 2. MODEL
# ==================================================

class ConvNeXtCBAM(nn.Module):

    def __init__(self, num_classes=4):

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


# ==================================================
# 3. CLASS NAMES
# ==================================================

CLASS_NAMES = [
    "BG",
    "Healthy",
    "WSSV",
    "WSSV_BG"
]


# ==================================================
# 4. DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==================================================
# 5. LOAD TEST DATA
# ==================================================

(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader
) = create_dataloaders()

print("Test images:", len(test_dataset))


# ==================================================
# 6. LOAD BEST MODEL
# ==================================================

model = ConvNeXtCBAM(
    num_classes=4
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "convnext_tiny_cbam_focal_reproduced_best.pth"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Best reproduced model loaded.")


# ==================================================
# 7. FIND WSSV ↔ WSSV_BG ERRORS
# ==================================================

print("\n========== WSSV / WSSV_BG ERRORS ==========\n")

error_count = 0

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        for sample_index in range(len(labels)):

            true_class = labels[sample_index].item()
            predicted_class = predictions[sample_index].item()

            # Only show WSSV <-> WSSV_BG mistakes
            if (
                true_class in [2, 3]
                and predicted_class in [2, 3]
                and true_class != predicted_class
            ):

                confidence = probabilities[
                    sample_index,
                    predicted_class
                ].item()

                global_index = (
                    batch_index * test_loader.batch_size
                    + sample_index
                )

                row = test_dataset.data.iloc[
                    global_index
                ]

                print(
                    "Image:",
                    row["path"]
                )

                print(
                    "True class:",
                    CLASS_NAMES[true_class]
                )

                print(
                    "Predicted class:",
                    CLASS_NAMES[predicted_class]
                )

                print(
                    "Confidence:",
                    f"{confidence:.2%}"
                )

                print("-" * 60)

                error_count += 1


print(
    "\nTotal WSSV/WSSV_BG errors:",
    error_count
)