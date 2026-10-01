import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import Hybrid3DCNN


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results" / "week2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PATCH_SIZE = 5

BATCH_SIZE = 64

EPOCHS = 15

LEARNING_RATE = 0.001

NUM_CLASSES = 2

CLASS_NAMES = {
    0: "Healthy",
    1: "Chemically Stressed"
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("TerraSpectra - Week 2 Hybrid 3D-CNN Training")
print("=" * 60)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

cube_path = PROCESSED_DIR / "normalized_hyperspectral.npy"
labels_path = RAW_DIR / "labels.npy"

if not cube_path.exists():
    raise FileNotFoundError(
        f"Normalized hyperspectral cube not found:\n{cube_path}"
    )

if not labels_path.exists():
    raise FileNotFoundError(
        f"Labels file not found:\n{labels_path}"
    )


cube = np.load(cube_path)

labels = np.load(labels_path)


print("\nDataset information")
print("-" * 60)

print("Cube shape   :", cube.shape)
print("Labels shape :", labels.shape)

print("Minimum value:", cube.min())
print("Maximum value:", cube.max())

print("\nClass distribution:")

unique_classes, class_counts = np.unique(
    labels,
    return_counts=True
)

for class_id, count in zip(
    unique_classes,
    class_counts
):
    print(
        f"{class_id} - "
        f"{CLASS_NAMES.get(int(class_id), 'Unknown')}: "
        f"{count}"
    )


# ============================================================
# VERIFY DATA
# ============================================================

if cube.ndim != 3:
    raise ValueError(
        "Expected hyperspectral cube with shape "
        "(height, width, spectral_bands)"
    )

if labels.ndim != 2:
    raise ValueError(
        "Expected labels with shape (height, width)"
    )

height, width, spectral_bands = cube.shape

if labels.shape != (height, width):
    raise ValueError(
        "Cube and labels spatial dimensions do not match."
    )

print("\nSpectral bands:", spectral_bands)

if spectral_bands < 200:
    raise ValueError(
        "Week 2 requires the 200+ band hyperspectral data."
    )


# ============================================================
# PAD HYPERSPECTRAL CUBE
# ============================================================

padding = PATCH_SIZE // 2

padded_cube = np.pad(
    cube,
    (
        (padding, padding),
        (padding, padding),
        (0, 0)
    ),
    mode="reflect"
)


# ============================================================
# PYTORCH DATASET
# ============================================================

class HyperspectralDataset(Dataset):

    def __init__(
        self,
        cube,
        labels,
        coordinates
    ):

        self.cube = cube

        self.labels = labels

        self.coordinates = coordinates

        self.patch_size = PATCH_SIZE

        self.padding = PATCH_SIZE // 2

    def __len__(self):

        return len(self.coordinates)

    def __getitem__(self, index):

        y, x = self.coordinates[index]

        y_p = y + self.padding

        x_p = x + self.padding

        half = self.patch_size // 2

        patch = self.cube[
            y_p - half:y_p + half + 1,
            x_p - half:x_p + half + 1,
            :
        ]

        # Current patch:
        # [height, width, spectral_bands]

        # Convert to:
        # [spectral_bands, height, width]

        patch = np.transpose(
            patch,
            (2, 0, 1)
        )

        # Convert to:
        # [1, spectral_bands, height, width]

        patch = np.expand_dims(
            patch,
            axis=0
        )

        patch = torch.tensor(
            patch,
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[y, x],
            dtype=torch.long
        )

        return patch, label


# ============================================================
# CREATE COORDINATES
# ============================================================

coordinates = []

targets = []

for y in range(height):

    for x in range(width):

        coordinates.append(
            (y, x)
        )

        targets.append(
            int(labels[y, x])
        )


coordinates = np.array(
    coordinates
)

targets = np.array(
    targets
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_coordinates, val_coordinates, train_targets, val_targets = (
    train_test_split(
        coordinates,
        targets,
        test_size=0.2,
        random_state=SEED,
        stratify=targets
    )
)


print("\nData split")
print("-" * 60)

print(
    "Training samples  :",
    len(train_coordinates)
)

print(
    "Validation samples:",
    len(val_coordinates)
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = HyperspectralDataset(
    padded_cube,
    labels,
    train_coordinates
)

val_dataset = HyperspectralDataset(
    padded_cube,
    labels,
    val_coordinates
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = Hybrid3DCNN(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_targets,
    minlength=NUM_CLASSES
)

class_weights = (
    len(train_targets)
    /
    (
        NUM_CLASSES * class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


print("\nClass weights:")
print(class_weights)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_validation_accuracy = 0.0

best_model_path = (
    MODEL_DIR /
    "hybrid_3dcnn.pth"
)

training_history = []


print("\nStarting training...")
print("=" * 60)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for inputs, targets_batch in train_loader:

        inputs = inputs.to(DEVICE)

        targets_batch = targets_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            inputs.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions ==
            targets_batch
        ).sum().item()

        total += inputs.size(0)


    train_loss = (
        running_loss /
        total
    )

    train_accuracy = (
        correct /
        total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0

    validation_correct = 0

    validation_total = 0


    with torch.no_grad():

        for inputs, targets_batch in val_loader:

            inputs = inputs.to(DEVICE)

            targets_batch = targets_batch.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets_batch
            )

            validation_loss += (
                loss.item() *
                inputs.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            validation_correct += (
                predictions ==
                targets_batch
            ).sum().item()

            validation_total += (
                inputs.size(0)
            )


    validation_loss = (
        validation_loss /
        validation_total
    )

    validation_accuracy = (
        validation_correct /
        validation_total
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "spectral_bands": spectral_bands,
                "patch_size": PATCH_SIZE,
                "class_names": CLASS_NAMES
            },
            best_model_path
        )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    epoch_record = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy
    }

    training_history.append(
        epoch_record
    )


    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {validation_loss:.4f} "
        f"| Val Acc: {validation_accuracy:.4f}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = (
    RESULTS_DIR /
    "training_history.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        training_history,
        file,
        indent=4
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)

print("Training completed.")

print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.4f}"
)

print(
    f"Model saved to:\n{best_model_path}"
)

print(
    f"Training history saved to:\n{history_path}"
)

print("=" * 60)