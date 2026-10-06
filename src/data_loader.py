from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16


# Class names and their numerical labels
CLASS_NAMES = ["BG", "Healthy", "WSSV", "WSSV_BG"]

CLASS_TO_INDEX = {
    "BG": 0,
    "Healthy": 1,
    "WSSV": 2,
    "WSSV_BG": 3
}


# ============================================================
# 3. IMAGE NORMALIZATION
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# 4. TRAINING TRANSFORMATIONS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),
])

# ============================================================
# 5. VALIDATION / TEST TRANSFORMATIONS
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# 6. CUSTOM DATASET
# ============================================================

class ShrimpDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["path"]

        class_name = row["class"]

        label = CLASS_TO_INDEX[class_name]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:

            image = self.transform(image)

        return image, label


# ============================================================
# 7. CREATE DATALOADERS
# ============================================================

def create_dataloaders():

    train_csv = SPLIT_DIR / "train.csv"
    val_csv = SPLIT_DIR / "val.csv"
    test_csv = SPLIT_DIR / "test.csv"


    train_dataset = ShrimpDataset(
        train_csv,
        transform=train_transform
    )

    val_dataset = ShrimpDataset(
        val_csv,
        transform=eval_transform
    )

    test_dataset = ShrimpDataset(
        test_csv,
        transform=eval_transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# 8. TEST THE DATALOADER
# ============================================================

if __name__ == "__main__":

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()


    print("\n========== DATASET INFORMATION ==========\n")

    print("Classes:")
    print(CLASS_NAMES)

    print("\nClass-to-index mapping:")
    print(CLASS_TO_INDEX)

    print("\nDataset sizes:")

    print(f"Training   : {len(train_dataset)}")
    print(f"Validation : {len(val_dataset)}")
    print(f"Test       : {len(test_dataset)}")


    images, labels = next(iter(train_loader))


    print("\n========== FIRST BATCH ==========\n")

    print(f"Image tensor shape : {images.shape}")
    print(f"Label tensor shape : {labels.shape}")
    print(f"Labels             : {labels.tolist()}")


    print("\nDataLoader is working correctly.")