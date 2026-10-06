from pathlib import Path
import re

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

ORIGINAL_DIR = PROJECT_ROOT / "data" / "raw" / "ShrimpImages" / "ShrimpImages"
AUGMENTED_DIR = PROJECT_ROOT / "data" / "raw" / "Augmented_shrimp"

CLASSES = ["BG", "Healthy", "WSSV", "WSSV_BG"]


def inspect_relationships():

    print("\n========== AUGMENTED IMAGE RELATIONSHIPS ==========\n")

    for class_name in CLASSES:

        original_dir = ORIGINAL_DIR / class_name
        augmented_dir = AUGMENTED_DIR / class_name

        original_files = {
            file.name
            for file in original_dir.iterdir()
            if file.is_file()
        }

        augmented_files = [
            file.name
            for file in augmented_dir.iterdir()
            if file.is_file() and file.name.startswith("aug_")
        ]

        matched = []
        unmatched = []

        for filename in augmented_files:

            # Remove augmentation prefix.
            # Example:
            # aug_0_Healthy-16-img-2.jpg
            # becomes:
            # Healthy-16-img-2.jpg

            parts = filename.split("_", 2)

            if len(parts) == 3:
                original_name = parts[2]

                if original_name in original_files:
                    matched.append(filename)
                else:
                    unmatched.append(filename)

        print(f"{class_name}")
        print(f"  Augmented images : {len(augmented_files)}")
        print(f"  Matched         : {len(matched)}")
        print(f"  Unmatched       : {len(unmatched)}")

        if unmatched:
            print("  Unmatched examples:")
            for filename in unmatched[:5]:
                print(f"    {filename}")

        print()


def main():
    print("ShrimpDiseaseDB - Augmentation Relationship Check")
    print("--------------------------------------------------")

    inspect_relationships()

    print("Inspection completed.")
    print("No files were modified.")


if __name__ == "__main__":
    main()