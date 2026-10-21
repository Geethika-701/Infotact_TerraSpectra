import os
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "raw"
)

labels_file = os.path.join(
    RAW_DIR,
    "labels.npy"
)

labels = np.load(
    labels_file
)

print("=" * 60)
print("Week 2 Labels")
print("=" * 60)

print(
    "Label shape:",
    labels.shape
)

print(
    "Healthy pixels:",
    np.sum(labels == 0)
)

print(
    "Chemically stressed pixels:",
    np.sum(labels == 1)
)