from pathlib import Path
import sys

import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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

from cbam import CBAM


# ============================================================
# 2. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "BG",
    "Healthy",
    "WSSV",
    "WSSV_BG"
]


# ============================================================
# 3. MODEL
# ============================================================

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


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 5. LOAD MODEL
# ============================================================

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


# ============================================================
# 6. TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. LOAD TEST CSV
# ============================================================

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

test_data = pd.read_csv(
    TEST_CSV
)


# ============================================================
# 8. OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 9. GRAD-CAM++ TARGET LAYER
# ============================================================

target_layers = [
    model.backbone.features[-1]
]


# ============================================================
# 10. PROCESS ONLY WSSV ↔ WSSV_BG ERRORS
# ============================================================

error_count = 0

for _, row in test_data.iterrows():

    image_path = row["path"]

    true_class_name = row["class"]

    # We only care about WSSV and WSSV_BG
    if true_class_name not in [
        "WSSV",
        "WSSV_BG"
    ]:
        continue

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    predicted_class_name = CLASS_NAMES[
        predicted_class
    ]

    # --------------------------------------------------------
    # Keep only WSSV ↔ WSSV_BG mistakes
    # --------------------------------------------------------

    if (
        predicted_class_name in [
            "WSSV",
            "WSSV_BG"
        ]
        and predicted_class_name != true_class_name
    ):

        confidence = probabilities[
            0,
            predicted_class
        ].item()

        # ----------------------------------------------------
        # Grad-CAM++
        # ----------------------------------------------------

        cam = GradCAMPlusPlus(
            model=model,
            target_layers=target_layers
        )

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        # ----------------------------------------------------
        # Prepare original image for visualization
        # ----------------------------------------------------

        rgb_image = (
            torch.from_numpy(
                __import__("numpy").array(
                    original_image.resize(
                        (224, 224)
                    )
                )
            ).float()
            / 255.0
        ).numpy()

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        filename = Path(
            image_path
        ).stem

        output_name = (
            f"{filename}"
            f"_TRUE-{true_class_name}"
            f"_PRED-{predicted_class_name}"
            f".jpg"
        )

        output_path = (
            OUTPUT_DIR
            / output_name
        )

        Image.fromarray(
            visualization
        ).save(output_path)

        print(
            f"Saved: {output_path.name}"
        )

        print(
            f"True: {true_class_name} | "
            f"Predicted: {predicted_class_name} | "
            f"Confidence: {confidence:.2%}"
        )

        print("-" * 70)

        error_count += 1


print()
print(
    "Total Grad-CAM++ error images:",
    error_count
)

print(
    "Saved to:",
    OUTPUT_DIR
)