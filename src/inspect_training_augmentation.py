from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

AUGMENTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Augmented_shrimp"
)

CLASSES = ["BG", "Healthy", "WSSV", "WSSV_BG"]


def main():

    train_df = pd.read_csv(SPLIT_DIR / "train.csv")

    print("\n========== TRAINING AUGMENTATION CHECK ==========\n")

    total_augmented = 0

    for class_name in CLASSES:

        train_class = train_df[
            train_df["class"] == class_name
        ]

        training_originals = set(
            train_class["filename"]
        )

        augmented_dir = AUGMENTED_DIR / class_name

        count = 0

        for file in augmented_dir.iterdir():

            if not file.is_file():
                continue

            if not file.name.startswith("aug_"):
                continue

            parts = file.name.split("_", 2)

            if len(parts) != 3:
                continue

            original_name = parts[2]

            if original_name in training_originals:
                count += 1

        total_augmented += count

        print(f"{class_name}")
        print(f"  Training originals : {len(training_originals)}")
        print(f"  Training augmentations : {count}")
        print()

    print("----------------------------------------------")
    print(f"Total training augmentations : {total_augmented}")
    print("\nNo files were modified.")


if __name__ == "__main__":
    main()