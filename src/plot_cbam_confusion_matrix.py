from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "confusion_matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


confusion_matrix = [
    [26, 2, 0, 8],
    [3, 50, 9, 0],
    [2, 4, 35, 15],
    [0, 0, 8, 27]
]


class_names = [
    "BG",
    "Healthy",
    "WSSV",
    "WSSV_BG"
]


plt.figure(figsize=(7, 6))

sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("ConvNeXt-Tiny + CBAM + Focal Loss")

plt.tight_layout()


output_path = (
    OUTPUT_DIR
    / "convnext_tiny_cbam_focal_confusion_matrix.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.show()

print("Confusion matrix saved to:")
print(output_path)