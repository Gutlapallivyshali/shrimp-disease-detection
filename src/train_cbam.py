from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models

PROJECT_ROOT = (
    Path.home()
    / "Documents"
    / "shrimp-disease-detection"
)

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import create_dataloaders
from cbam import CBAM
from focal_loss import FocalLoss


# ==================================================
# MODEL
# ==================================================

class ConvNeXtCBAM(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        # Pretrained ConvNeXt-Tiny
        self.backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        # CBAM attention
        self.cbam = CBAM(
            channels=768
        )

        # Classifier for 4 shrimp classes
        self.classifier = nn.Linear(
            768,
            num_classes
        )

    def forward(self, x):

        # Feature extraction
        x = self.backbone.features(x)

        # Attention
        x = self.cbam(x)

        # Global average pooling
        x = self.backbone.avgpool(x)

        # Convert to vector
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Classification
        x = self.classifier(x)

        return x


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)


    # ==================================================
    # DATA
    # ==================================================

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()


    print("\n========== DATASET INFORMATION ==========\n")

    print(
        "Training images   :",
        len(train_dataset)
    )

    print(
        "Validation images :",
        len(val_dataset)
    )

    print(
        "Test images       :",
        len(test_dataset)
    )


    # ==================================================
    # MODEL
    # ==================================================

    model = ConvNeXtCBAM(
        num_classes=4
    )

    model = model.to(device)


    # ==================================================
    # LOSS
    # ==================================================

    criterion = FocalLoss(
        alpha=1.0,
        gamma=2.0
    )


    # ==================================================
    # OPTIMIZER
    # ==================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00001
    )


    # ==================================================
    # TRAINING SETTINGS
    # ==================================================

    NUM_EPOCHS = 10

    MODEL_DIR = (
        PROJECT_ROOT
        / "outputs"
        / "models"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_accuracy = 0.0


    # ==================================================
    # TRAINING
    # ==================================================

    print(
        "\n========== LOWER LEARNING RATE TRAINING ==========\n"
    )


    for epoch in range(NUM_EPOCHS):

        # -----------------------------
        # TRAINING PHASE
        # -----------------------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Focal Loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

            # Predictions
            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


        train_loss = (
            running_loss
            / len(train_loader)
        )

        train_accuracy = (
            correct
            / total
        )


        # -----------------------------
        # VALIDATION PHASE
        # -----------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                predictions = outputs.argmax(
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)


        val_loss = (
            val_loss
            / len(val_loader)
        )

        val_accuracy = (
            val_correct
            / val_total
        )


        # -----------------------------
        # PRINT RESULTS
        # -----------------------------

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Accuracy: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Accuracy: {val_accuracy:.4f}"
        )


        # -----------------------------
        # SAVE BEST MODEL
        # -----------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            model_path = (
                MODEL_DIR
                / "convnext_tiny_cbam_focal_reproduced_best.pth"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"  ✓ Best model saved "
                f"(Validation Accuracy: "
                f"{val_accuracy:.4f})"
            )


    # ==================================================
    # COMPLETE
    # ==================================================

    print(
        "\n========== TRAINING COMPLETE ==========\n"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        "Best model saved at:"
    )

    print(
        MODEL_DIR
        / "convnext_tiny_cbam_focal_reproduced_best.pth"
    )