from pathlib import Path


PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLASSES = ["BG", "Healthy", "WSSV", "WSSV_BG"]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def main():

    print("\n========== PROCESSED DATASET VERIFICATION ==========\n")

    problems = []

    for split in ["train", "val", "test"]:

        print(split.upper())

        for class_name in CLASSES:

            folder = PROCESSED_DIR / split / class_name

            if not folder.exists():

                problems.append(
                    f"Missing folder: {folder}"
                )

                continue

            files = [
                file
                for file in folder.iterdir()
                if file.is_file()
            ]

            image_files = [
                file
                for file in files
                if file.suffix.lower() in IMAGE_EXTENSIONS
            ]

            print(
                f"  {class_name}: "
                f"{len(image_files)} images"
            )

            # Check for unexpected non-image files
            if len(files) != len(image_files):

                problems.append(
                    f"Non-image file found in {folder}"
                )

            # Validation/test must contain only original images
            if split in ["val", "test"]:

                augmented_files = [
                    file
                    for file in image_files
                    if file.name.startswith("aug_")
                ]

                if augmented_files:

                    problems.append(
                        f"Augmented images found in {split}/{class_name}"
                    )

        print()

    print("========== RESULT ==========\n")

    if problems:

        print("PROBLEMS FOUND:")

        for problem in problems:
            print(f"  - {problem}")

    else:

        print("✓ All checks passed.")
        print("✓ Validation contains no augmented images.")
        print("✓ Test contains no augmented images.")
        print("✓ Dataset structure is correct.")

    print("\nVerification completed.")


if __name__ == "__main__":
    main()