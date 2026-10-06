import numpy as np

class_names = [
    "BG",
    "Healthy",
    "WSSV",
    "WSSV_BG"
]

# Baseline confusion matrix
baseline_cm = np.array([
    [32, 1, 1, 2],
    [10, 41, 11, 0],
    [5, 1, 42, 8],
    [1, 0, 8, 26]
])

# CBAM + Focal Loss confusion matrix
cbam_cm = np.array([
    [26, 2, 0, 8],
    [3, 50, 9, 0],
    [2, 4, 35, 15],
    [0, 0, 8, 27]
])


print("========== CORRECT PREDICTIONS ==========\n")

for i, class_name in enumerate(class_names):

    baseline_correct = baseline_cm[i, i]
    cbam_correct = cbam_cm[i, i]

    print(
        f"{class_name:10} "
        f"Baseline: {baseline_correct:2d}   "
        f"CBAM: {cbam_correct:2d}"
    )


print("\n========== RECALL COMPARISON ==========\n")

for i, class_name in enumerate(class_names):

    baseline_recall = (
        baseline_cm[i, i] / baseline_cm[i].sum()
    )

    cbam_recall = (
        cbam_cm[i, i] / cbam_cm[i].sum()
    )

    difference = cbam_recall - baseline_recall

    print(
        f"{class_name:10} "
        f"Baseline: {baseline_recall:.2%}   "
        f"CBAM: {cbam_recall:.2%}   "
        f"Change: {difference:+.2%}"
    )


print("\n========== CONFUSION MATRIX DIFFERENCE ==========\n")

difference_matrix = cbam_cm - baseline_cm

print(difference_matrix)