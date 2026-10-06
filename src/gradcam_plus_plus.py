from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from cbam import CBAM


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "convnext_tiny_cbam_focal_best.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Class names
# --------------------------------------------------

CLASS_NAMES = [
    "BG",
    "Healthy",
    "WSSV",
    "WSSV_BG"
]


# --------------------------------------------------
# 3. Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# --------------------------------------------------
# 4. Define our trained model
# --------------------------------------------------

class ConvNeXtCBAM(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.backbone = models.convnext_tiny(
            weights=None
        )

        self.cbam = CBAM(channels=768)

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


# --------------------------------------------------
# 5. Load trained model
# --------------------------------------------------

model = ConvNeXtCBAM(num_classes=4)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# 6. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 7. Select one test image
# --------------------------------------------------

image_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "BG"
)

image_files = list(image_path.glob("*.jpg"))

if len(image_files) == 0:

    raise FileNotFoundError(
        "No JPG images found in the BG test folder."
    )

image_path = image_files[0]

print("Image:", image_path)


# --------------------------------------------------
# 8. Load image
# --------------------------------------------------

original_image = Image.open(
    image_path
).convert("RGB")

input_tensor = transform(
    original_image
).unsqueeze(0)

input_tensor = input_tensor.to(device)


# --------------------------------------------------
# 9. Predict disease
# --------------------------------------------------

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

confidence = probabilities[
    0,
    predicted_class
].item()


print(
    "Predicted class:",
    CLASS_NAMES[predicted_class]
)

print(
    "Confidence:",
    f"{confidence:.2%}"
)


# --------------------------------------------------
# 10. Grad-CAM++ target layer
# --------------------------------------------------

target_layers = [
    model.backbone.features[-1]
]


# --------------------------------------------------
# 11. Generate Grad-CAM++
# --------------------------------------------------

cam = GradCAMPlusPlus(
    model=model,
    target_layers=target_layers
)

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

targets = [
    ClassifierOutputTarget(predicted_class)
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]


# --------------------------------------------------
# 12. Prepare original image
# --------------------------------------------------

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


# --------------------------------------------------
# 13. Create heatmap overlay
# --------------------------------------------------

visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# --------------------------------------------------
# 14. Save result
# --------------------------------------------------

output_path = (
    OUTPUT_DIR
    / "gradcam_plus_plus_result.jpg"
)

Image.fromarray(
    visualization
).save(output_path)


print(
    "Grad-CAM++ result saved to:"
)

print(output_path)