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
# 2. LOAD SPLIT FILES
# ============================================================

train = pd.read_csv(
    SPLIT_DIR / "train.csv"
)

val = pd.read_csv(
    SPLIT_DIR / "val.csv"
)

test = pd.read_csv(
    SPLIT_DIR / "test.csv"
)


# ============================================================
# 3. CREATE SOURCE GROUP FROM FILE NAME
# ============================================================

def get_source_id(path):

    filename = Path(path).stem

    # Remove augmentation prefix if present
    # Example:
    # aug_0_WSSV-106-img-3
    # becomes:
    # WSSV-106-img-3

    if filename.startswith("aug_"):

        parts = filename.split("_", 2)

        if len(parts) == 3:
            filename = parts[2]

    # Example:
    # WSSV-106-img-3
    # becomes:
    # WSSV-106

    source_id = filename.split("-img-")[0]

    return source_id


# Create source_id for each split

train["source_id"] = train["path"].apply(
    get_source_id
)

val["source_id"] = val["path"].apply(
    get_source_id
)

test["source_id"] = test["path"].apply(
    get_source_id
)


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print("\n========== CLASS DISTRIBUTION ==========\n")

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test)
]:

    print(f"{name}:")

    print(
        df["class"].value_counts()
    )

    print()


# ============================================================
# 5. SOURCE GROUP DIVERSITY
# ============================================================

print("\n========== SOURCE GROUPS ==========\n")

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test)
]:

    print(f"{name}:")

    selected = df[
        df["class"].isin(
            ["WSSV", "WSSV_BG"]
        )
    ]

    source_counts = (
        selected
        .groupby("class")["source_id"]
        .nunique()
    )

    print(
        "Unique source groups:"
    )

    print(source_counts)

    print()


# ============================================================
# 6. IMAGES PER SOURCE GROUP
# ============================================================

print(
    "\n========== IMAGES PER SOURCE GROUP ==========\n"
)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test)
]:

    print(f"\n{name}:")

    selected = df[
        df["class"].isin(
            ["WSSV", "WSSV_BG"]
        )
    ]

    group_counts = (
        selected
        .groupby(
            ["class", "source_id"]
        )
        .size()
    )

    for class_name in [
        "WSSV",
        "WSSV_BG"
    ]:

        values = group_counts[
            class_name
        ]

        print(
            f"{class_name}: "
            f"min={values.min()}, "
            f"max={values.max()}, "
            f"average={values.mean():.2f}"
        )


print("\nAnalysis complete.")