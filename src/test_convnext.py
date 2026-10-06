from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from data_loader import create_dataloaders


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# Load pretrained ConvNeXt-Tiny
weights = models.ConvNeXt_Tiny_Weights.DEFAULT

model = models.convnext_tiny(weights=weights)


# Change final layer to 4 classes
num_classes = 4

model.classifier[2] = nn.Linear(
    model.classifier[2].in_features,
    num_classes
)

model = model.to(device)

model.eval()


# Load one batch
_, _, _, train_loader, _, _ = create_dataloaders()

images, labels = next(iter(train_loader))

# Use only 4 images for this quick test
images = images[:4].to(device)
labels = labels[:4].to(device)


# Forward pass
with torch.no_grad():
    outputs = model(images)


# Get predictions
probabilities = torch.softmax(outputs, dim=1)
predictions = probabilities.argmax(dim=1)


# Calculate loss
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)


# Print results
print("\n--- ConvNeXt-Tiny Test ---")
print("Input shape        :", images.shape)
print("Output shape       :", outputs.shape)
print("Probabilities shape:", probabilities.shape)
print("True labels        :", labels.cpu().tolist())
print("Predicted labels   :", predictions.cpu().tolist())
print("Loss               :", loss.item())

print("\nConvNeXt-Tiny forward pass successful!")