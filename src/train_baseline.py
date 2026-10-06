from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# Import our CSV-based DataLoader
from data_loader import create_dataloaders


# ============================================================
# 2. SELECT DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 3. CREATE DATA LOADERS
# ============================================================

(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader
) = create_dataloaders()


print("\n========== DATASET INFORMATION ==========\n")

print("Training images   :", len(train_dataset))
print("Validation images :", len(val_dataset))
print("Test images       :", len(test_dataset))


# ============================================================
# 4. LOAD PRETRAINED CONVNEXT-TINY
# ============================================================

weights = models.ConvNeXt_Tiny_Weights.DEFAULT

model = models.convnext_tiny(
    weights=weights
)


# ============================================================
# 5. CHANGE FINAL LAYER FOR OUR 4 CLASSES
# ============================================================

num_classes = 4

model.classifier[2] = nn.Linear(
    model.classifier[2].in_features,
    num_classes
)


# Move model to CPU/GPU
model = model.to(device)


# ============================================================
# 6. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 7. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001
)


# ============================================================
# 8. TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 10


# Folder for saving trained models
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Store the best validation accuracy
best_val_accuracy = 0.0


# ============================================================
# 9. TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    # Put model into training mode
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    # --------------------------------------------------------
    # Training batches
    # --------------------------------------------------------

    for images, labels in train_loader:

        # Move images and labels to selected device
        images = images.to(device)
        labels = labels.to(device)


        # Clear old gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(outputs, labels)


        # Backpropagation
        loss.backward()


        # Update model weights
        optimizer.step()


        # Add batch loss
        running_loss += loss.item()


        # Get predicted class
        predictions = outputs.argmax(dim=1)


        # Count correct predictions
        correct += (
            predictions == labels
        ).sum().item()


        # Count total images
        total += labels.size(0)


    # Calculate average training loss
    epoch_loss = (
        running_loss / len(train_loader)
    )


    # Calculate training accuracy
    epoch_accuracy = (
        correct / total
    )


    # ========================================================
    # 10. VALIDATION
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0


    # We don't calculate gradients during validation
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            # Forward pass
            outputs = model(images)


            # Get predicted class
            predictions = outputs.argmax(dim=1)


            # Count correct predictions
            val_correct += (
                predictions == labels
            ).sum().item()


            # Count total validation images
            val_total += labels.size(0)


    # Calculate validation accuracy
    val_accuracy = (
        val_correct / val_total
    )


    # ========================================================
    # 11. PRINT EPOCH RESULTS
    # ========================================================

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Training Accuracy: {epoch_accuracy:.4f} "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


    # ========================================================
    # 12. SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        model_path = (
            MODEL_DIR / "convnext_tiny_baseline_best.pth"
        )


        torch.save(
            model.state_dict(),
            model_path
        )


        print(
            f"  ✓ Best model saved "
            f"(Validation Accuracy: {val_accuracy:.4f})"
        )


# ============================================================
# 13. TRAINING COMPLETE
# ============================================================

print("\n========== TRAINING COMPLETE ==========\n")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    "Best model saved at:"
)

print(
    MODEL_DIR / "convnext_tiny_baseline_best.pth"
)