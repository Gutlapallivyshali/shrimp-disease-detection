from pathlib import Path
import re
from collections import Counter

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

ORIGINAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ShrimpImages"
    / "ShrimpImages"
)

CLASSES = ["BG", "Healthy", "WSSV", "WSSV_BG"]


def inspect_source_groups():

    print("\n========== SOURCE GROUP INSPECTION ==========\n")

    for class_name in CLASSES:

        class_dir = ORIGINAL_DIR / class_name

        files = [
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        groups = []

        pattern = re.compile(
            rf"^{re.escape(class_name)}-(\d+)-img-\d+\.",
            re.IGNORECASE
        )

        for file in files:

            match = pattern.match(file.name)

            if match:
                source_id = match.group(1)
                groups.append(source_id)

        counts = Counter(groups)

        print(f"{class_name}")
        print(f"  Total images       : {len(files)}")
        print(f"  Source groups      : {len(counts)}")

        if counts:
            print(f"  Images per group:")
            print(f"    Minimum          : {min(counts.values())}")
            print(f"    Maximum          : {max(counts.values())}")
            print(f"    Average          : {sum(counts.values()) / len(counts):.2f}")

        print()


def main():

    print("ShrimpDiseaseDB - Source Group Inspection")
    print("-----------------------------------------")

    inspect_source_groups()

    print("Inspection completed.")
    print("No files were modified.")


if __name__ == "__main__":
    main()