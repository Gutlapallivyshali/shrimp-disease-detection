from pathlib import Path
import pandas as pd
import shutil


PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

ORIGINAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ShrimpImages"
    / "ShrimpImages"
)

AUGMENTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Augmented_shrimp"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLASSES = ["BG", "Healthy", "WSSV", "WSSV_BG"]


def create_folders():

    for split in ["train", "val", "test"]:

        for class_name in CLASSES:

            folder = PROCESSED_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)


def copy_original_images(df, split):

    for _, row in df.iterrows():

        class_name = row["class"]
        filename = row["filename"]

        source = ORIGINAL_DIR / class_name / filename
        destination = (
            PROCESSED_DIR
            / split
            / class_name
            / filename
        )

        shutil.copy2(source, destination)


def copy_training_augmentations(train_df):

    for class_name in CLASSES:

        training_originals = set(
            train_df[
                train_df["class"] == class_name
            ]["filename"]
        )

        augmented_dir = AUGMENTED_DIR / class_name

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

                destination = (
                    PROCESSED_DIR
                    / "train"
                    / class_name
                    / file.name
                )

                shutil.copy2(file, destination)


def main():

    print("\n========== PREPARING DATASET ==========\n")

    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    val_df = pd.read_csv(SPLIT_DIR / "val.csv")
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")

    print("Creating folders...")
    create_folders()

    print("Copying training originals...")
    copy_original_images(train_df, "train")

    print("Copying validation images...")
    copy_original_images(val_df, "val")

    print("Copying test images...")
    copy_original_images(test_df, "test")

    print("Copying training augmentations...")
    copy_training_augmentations(train_df)

    print("\n========== DATASET PREPARED ==========\n")

    for split in ["train", "val", "test"]:

        print(split.upper())

        for class_name in CLASSES:

            folder = (
                PROCESSED_DIR
                / split
                / class_name
            )

            count = len([
                f for f in folder.iterdir()
                if f.is_file()
            ])

            print(f"  {class_name}: {count}")

        print()

    print("Original dataset was NOT modified.")


if __name__ == "__main__":
    main()