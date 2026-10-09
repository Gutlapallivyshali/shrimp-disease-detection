from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path.home()
    / "Documents"
    / "shrimp-disease-detection"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

train = pd.read_csv(
    SPLIT_DIR / "train.csv"
)


# ============================================================
# 3. IDENTIFY ORIGINAL / AUGMENTED IMAGES
# ============================================================

def is_augmented(path):

    filename = Path(path).stem

    return filename.startswith("aug_")


train["is_augmented"] = train["path"].apply(
    is_augmented
)


# ============================================================
# 4. COUNT ORIGINAL VS AUGMENTED
# ============================================================

print(
    "\n========== ORIGINAL VS AUGMENTED ==========\n"
)

for class_name in [
    "WSSV",
    "WSSV_BG"
]:

    class_data = train[
        train["class"] == class_name
    ]

    original_count = (
        ~class_data["is_augmented"]
    ).sum()

    augmented_count = (
        class_data["is_augmented"]
    ).sum()

    total = len(class_data)

    print(
        f"{class_name}:"
    )

    print(
        f"  Original  : {original_count}"
    )

    print(
        f"  Augmented : {augmented_count}"
    )

    print(
        f"  Total     : {total}"
    )

    print()


# ============================================================
# 5. AUGMENTATION RATIO
# ============================================================

print(
    "========== AUGMENTATION RATIO ==========\n"
)

for class_name in [
    "WSSV",
    "WSSV_BG"
]:

    class_data = train[
        train["class"] == class_name
    ]

    original_count = (
        ~class_data["is_augmented"]
    ).sum()

    augmented_count = (
        class_data["is_augmented"]
    ).sum()

    ratio = (
        augmented_count / original_count
        if original_count > 0
        else 0
    )

    print(
        f"{class_name}: "
        f"{ratio:.2f} augmented images "
        f"per original image"
    )


print(
    "\nAnalysis complete."
)