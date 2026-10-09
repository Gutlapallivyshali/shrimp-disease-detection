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
# 2. LOAD CURRENT TRAINING CSV
# ============================================================

train_csv = SPLIT_DIR / "train.csv"

train = pd.read_csv(train_csv)


# ============================================================
# 3. IDENTIFY ORIGINAL IMAGES
# ============================================================

def is_original(path):

    filename = Path(path).stem

    return not filename.startswith("aug_")


train_original = train[
    train["path"].apply(is_original)
].copy()


# ============================================================
# 4. SAVE NEW CSV
# ============================================================

output_csv = (
    SPLIT_DIR
    / "train_original_only.csv"
)

train_original.to_csv(
    output_csv,
    index=False
)


# ============================================================
# 5. PRINT RESULTS
# ============================================================

print("\n========== ORIGINAL TRAINING SET ==========\n")

print(
    "Original training images:",
    len(train_original)
)

print("\nClass distribution:")

print(
    train_original["class"].value_counts()
)

print(
    "\nSaved to:"
)

print(output_csv)