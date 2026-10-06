from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


def main():

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")

    print("\n========== SPLIT VERIFICATION ==========\n")

    train_groups = set(
        zip(train["class"], train["source_id"])
    )

    val_groups = set(
        zip(val["class"], val["source_id"])
    )

    test_groups = set(
        zip(test["class"], test["source_id"])
    )

    train_val = train_groups & val_groups
    train_test = train_groups & test_groups
    val_test = val_groups & test_groups

    print(f"Train images      : {len(train)}")
    print(f"Validation images : {len(val)}")
    print(f"Test images       : {len(test)}")

    print()

    print(f"Train groups      : {len(train_groups)}")
    print(f"Validation groups : {len(val_groups)}")
    print(f"Test groups       : {len(test_groups)}")

    print("\n========== LEAKAGE CHECK ==========\n")

    print(f"Train ∩ Validation : {len(train_val)}")
    print(f"Train ∩ Test       : {len(train_test)}")
    print(f"Validation ∩ Test  : {len(val_test)}")

    if len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0:
        print("\n✓ NO SOURCE-GROUP LEAKAGE DETECTED")
    else:
        print("\n✗ LEAKAGE DETECTED")

    print("\nVerification completed.")


if __name__ == "__main__":
    main()