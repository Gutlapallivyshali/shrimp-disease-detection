from pathlib import Path
import random
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

RAW_DIR = PROJECT_ROOT / "data" / "raw"

ORIGINAL_DIR = RAW_DIR / "ShrimpImages" / "ShrimpImages"

AUGMENTED_DIR = RAW_DIR / "Augmented_shrimp"

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


# --------------------------------------------------
# 2. SETTINGS
# --------------------------------------------------

CLASSES = ["BG", "Healthy", "WSSV", "WSSV_BG"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# --------------------------------------------------
# 3. CREATE OUTPUT FOLDER
# --------------------------------------------------

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 4. FIND ORIGINAL IMAGES
# --------------------------------------------------

original_records = []

for class_name in CLASSES:

    class_folder = ORIGINAL_DIR / class_name

    for image_path in class_folder.glob("*.jpg"):

        filename = image_path.name

        # Example:
        # BG-4-img-1.jpg
        # split → ["BG", "4", "img", "1"]

        parts = image_path.stem.split("-")

        source_group = "-".join(parts[:2])

        original_records.append({
            "path": str(image_path),
            "class": class_name,
            "source_group": source_group,
            "is_augmented": False
        })


# --------------------------------------------------
# 5. GROUP ORIGINAL IMAGES
# --------------------------------------------------

groups = {}

for record in original_records:

    key = (record["class"], record["source_group"])

    if key not in groups:
        groups[key] = []

    groups[key].append(record)


# --------------------------------------------------
# 6. SPLIT SOURCE GROUPS
# --------------------------------------------------

random.seed(RANDOM_SEED)

train_records = []
val_records = []
test_records = []


for class_name in CLASSES:

    class_groups = [
        key for key in groups
        if key[0] == class_name
    ]

    random.shuffle(class_groups)

    total_groups = len(class_groups)

    train_end = int(total_groups * TRAIN_RATIO)

    val_end = train_end + int(total_groups * VAL_RATIO)

    train_groups = class_groups[:train_end]

    val_groups = class_groups[train_end:val_end]

    test_groups = class_groups[val_end:]


    for group in train_groups:
        train_records.extend(groups[group])

    for group in val_groups:
        val_records.extend(groups[group])

    for group in test_groups:
        test_records.extend(groups[group])


# --------------------------------------------------
# 7. ADD AUGMENTED IMAGES ONLY TO TRAINING
# --------------------------------------------------

train_source_groups = {
    (record["class"], record["source_group"])
    for record in train_records
}


for class_name in CLASSES:

    augmented_folder = AUGMENTED_DIR / class_name

    for image_path in augmented_folder.glob("*.jpg"):

        filename = image_path.name

        # Skip original images.
        # Augmented filenames start with "aug_"

        if not filename.startswith("aug_"):
            continue

        original_name = filename.split("_", 2)[2]

        parts = Path(original_name).stem.split("-")

        source_group = "-".join(parts[:2])

        key = (class_name, source_group)

        if key in train_source_groups:

            train_records.append({
                "path": str(image_path),
                "class": class_name,
                "source_group": source_group,
                "is_augmented": True
            })


# --------------------------------------------------
# 8. CONVERT TO DATAFRAMES
# --------------------------------------------------

train_df = pd.DataFrame(train_records)

val_df = pd.DataFrame(val_records)

test_df = pd.DataFrame(test_records)


# --------------------------------------------------
# 9. SHUFFLE ROWS
# --------------------------------------------------

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

val_df = val_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# --------------------------------------------------
# 10. SAVE CSV FILES
# --------------------------------------------------

train_df.to_csv(
    SPLIT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    SPLIT_DIR / "val.csv",
    index=False
)

test_df.to_csv(
    SPLIT_DIR / "test.csv",
    index=False
)


# --------------------------------------------------
# 11. PRINT RESULTS
# --------------------------------------------------

print("\n========== SPLIT RESULTS ==========\n")

print("Training images   :", len(train_df))
print("Validation images :", len(val_df))
print("Test images       :", len(test_df))

print("\nAugmented images:")

print(
    "Training   :",
    train_df["is_augmented"].sum()
)

print(
    "Validation :",
    val_df["is_augmented"].sum()
)

print(
    "Test       :",
    test_df["is_augmented"].sum()
)

print("\nCSV files created successfully.")