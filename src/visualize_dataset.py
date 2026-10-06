from pathlib import Path
import random

import matplotlib.pyplot as plt
from torchvision import datasets, transforms


PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "train"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)


class_names = dataset.classes


print("Classes:", class_names)


# Get 2 random images from each class
selected_indices = []

for class_index in range(len(class_names)):

    indices = [
        i
        for i, (_, label) in enumerate(dataset.samples)
        if label == class_index
    ]

    selected = random.sample(indices, 2)

    selected_indices.extend(selected)


plt.figure(figsize=(12, 8))


for position, index in enumerate(selected_indices):

    image, label = dataset[index]

    image = image.permute(1, 2, 0)

    plt.subplot(2, 4, position + 1)

    plt.imshow(image)

    plt.title(class_names[label])

    plt.axis("off")


plt.tight_layout()

plt.show()