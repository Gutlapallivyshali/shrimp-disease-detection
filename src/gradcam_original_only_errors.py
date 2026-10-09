from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

        return image, label, image_path


# ============================================================
# 4. TRANSFORM
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
# 5. MODEL
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
# 6. LOAD MODEL
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
# 7. LOAD TEST DATA
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
# 8. FIND WSSV / WSSV_BG ERRORS
# ============================================================

error_samples = []

with torch.no_grad():

    for images, labels, paths in test_loader:

        images = images.to(device)

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(
            len(labels)
        ):

            true_label = labels[i].item()

            predicted_label = predictions[i].item()

            if (
                true_label in [2, 3]
                and predicted_label in [2, 3]
                and true_label != predicted_label
            ):

                confidence = probabilities[
                    i,
                    predicted_label
                ].item()

                error_samples.append({
                    "path": paths[i],
                    "true": true_label,
                    "predicted": predicted_label,
                    "confidence": confidence
                })


# ============================================================
# 9. PRINT ERRORS
# ============================================================

print(
    "\nWSSV / WSSV_BG errors:",
    len(error_samples)
)

for i, sample in enumerate(
    error_samples,
    start=1
):

    print(
        f"{i}. "
        f"True: {CLASS_NAMES[sample['true']]} | "
        f"Predicted: {CLASS_NAMES[sample['predicted']]} | "
        f"Confidence: {sample['confidence']:.2%} | "
        f"{Path(sample['path']).name}"
    )


# ============================================================
# 10. GRAD-CAM++ OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam"
    / "original_only_errors"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 11. TARGET LAYER
# ============================================================

target_layers = [
    model.backbone.features[-1]
]


# ============================================================
# 12. GENERATE GRAD-CAM++
# ============================================================

for index, sample in enumerate(
    error_samples,
    start=1
):

    image_path = sample["path"]

    original_image = Image.open(
        image_path
    ).convert("RGB")

    original_image = original_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_image = (
        torch.tensor(
            list(original_image.getdata()),
            dtype=torch.float32
        )
        .reshape(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
        / 255.0
    ).numpy()


    input_tensor = test_transform(
        Image.open(
            image_path
        ).convert("RGB")
    ).unsqueeze(0).to(device)


    predicted_class = sample[
        "predicted"
    ]


    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]


    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers
    )


    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]


    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    output_name = (
        f"{index:02d}_"
        f"true_{CLASS_NAMES[sample['true']]}_"
        f"pred_{CLASS_NAMES[sample['predicted']]}.jpg"
    )


    output_path = (
        OUTPUT_DIR
        / output_name
    )


    Image.fromarray(
        visualization
    ).save(
        output_path
    )


    print(
        "Saved:",
        output_path
    )


print(
    "\nGrad-CAM++ error analysis complete."
)