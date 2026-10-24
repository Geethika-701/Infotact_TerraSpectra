import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from backend.model import TerraSpectra3DCNN



# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

CUBE_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "raw",
    "mock_hyperspectral_cube.npy"
)

LABEL_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "raw",
    "labels.npy"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "terraspectra_3dcnn.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PATCH_SIZE = 7
BANDS = 224

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class BalancedSpectralDataset(Dataset):

    def __init__(
        self,
        cube,
        labels
    ):

        self.cube = cube
        self.labels = labels

        half = PATCH_SIZE // 2

        healthy_positions = []
        stressed_positions = []

        height = labels.shape[0]
        width = labels.shape[1]

        y = half

        while y < height - half:

            x = half

            while x < width - half:

                label = int(
                    labels[y, x]
                )

                if label == 0:

                    healthy_positions.append(
                        (y, x)
                    )

                else:

                    stressed_positions.append(
                        (y, x)
                    )

                x += 1

            y += 1

        print(
            "Original healthy centers:",
            len(healthy_positions)
        )

        print(
            "Original stressed centers:",
            len(stressed_positions)
        )

        # ----------------------------------------------------
        # Balance the classes
        # ----------------------------------------------------

        random_generator = np.random.default_rng(
            SEED
        )

        number_per_class = min(
            len(healthy_positions),
            len(stressed_positions)
        )

        healthy_indices = random_generator.choice(
            len(healthy_positions),
            size=number_per_class,
            replace=False
        )

        stressed_indices = random_generator.choice(
            len(stressed_positions),
            size=number_per_class,
            replace=False
        )

        self.samples = []

        for index in healthy_indices:

            y_coord, x_coord = healthy_positions[
                index
            ]

            self.samples.append(
                (
                    y_coord,
                    x_coord,
                    0
                )
            )

        for index in stressed_indices:

            y_coord, x_coord = stressed_positions[
                index
            ]

            self.samples.append(
                (
                    y_coord,
                    x_coord,
                    1
                )
            )

        # Shuffle the balanced samples

        random_generator.shuffle(
            self.samples
        )

        print(
            "Balanced training samples:",
            len(self.samples)
        )

        print(
            "Healthy training samples:",
            number_per_class
        )

        print(
            "Stressed training samples:",
            number_per_class
        )

    def __len__(
        self
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):

        y_coord, x_coord, label = self.samples[
            index
        ]

        half = PATCH_SIZE // 2

        patch = self.cube[
            y_coord - half:
            y_coord + half + 1,
            x_coord - half:
            x_coord + half + 1,
            :
        ]

        patch = patch.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Per-patch spectral normalization
        # ----------------------------------------------------

        patch_min = patch.min()
        patch_max = patch.max()

        if patch_max - patch_min > 1e-8:

            patch = (
                patch - patch_min
            ) / (
                patch_max - patch_min
            )

        else:

            patch = np.zeros_like(
                patch,
                dtype=np.float32
            )

        tensor = torch.from_numpy(
            patch
        )

        # [7, 7, 224]
        #
        # CNN expects:
        # [1, 7, 7, 224]

        tensor = tensor.unsqueeze(
            0
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return (
            tensor,
            label_tensor
        )


# ============================================================
# LOAD DATA
# ============================================================

print("")
print("========================================")
print("TerraSpectra 3D-CNN Training")
print("========================================")
print("")

print(
    "Device:",
    DEVICE
)

print(
    "Cube:",
    CUBE_PATH
)

print(
    "Labels:",
    LABEL_PATH
)

cube = np.load(
    CUBE_PATH
).astype(
    np.float32
)

labels = np.load(
    LABEL_PATH
)

print(
    "Cube shape:",
    cube.shape
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Spectral bands:",
    cube.shape[2]
)

print(
    "Healthy pixels:",
    int(
        np.sum(labels == 0)
    )
)

print(
    "Stressed pixels:",
    int(
        np.sum(labels == 1)
    )
)


# ============================================================
# DATASET
# ============================================================

dataset = BalancedSpectralDataset(
    cube,
    labels
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = TerraSpectra3DCNN()

model = model.to(
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    predicted_healthy = 0

    predicted_stressed = 0

    actual_healthy = 0

    actual_stressed = 0

    batch_number = 0

    for inputs, targets in loader:

        inputs = inputs.to(
            DEVICE
        )

        targets = targets.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            inputs
        )

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += int(
            (
                predictions == targets
            ).sum().item()
        )

        total += targets.size(
            0
        )

        predicted_healthy += int(
            (
                predictions == 0
            ).sum().item()
        )

        predicted_stressed += int(
            (
                predictions == 1
            ).sum().item()
        )

        actual_healthy += int(
            (
                targets == 0
            ).sum().item()
        )

        actual_stressed += int(
            (
                targets == 1
            ).sum().item()
        )

        batch_number += 1

    epoch_loss = (
        total_loss /
        batch_number
    )

    epoch_accuracy = (
        correct /
        total
        *
        100.0
    )

    print("")
    print(
        "Epoch "
        + str(epoch + 1)
        + "/"
        + str(EPOCHS)
    )

    print(
        "Loss:",
        round(
            epoch_loss,
            6
        )
    )

    print(
        "Accuracy:",
        round(
            epoch_accuracy,
            2
        ),
        "%"
    )

    print(
        "Predicted Healthy:",
        predicted_healthy
    )

    print(
        "Predicted Stressed:",
        predicted_stressed
    )

    print(
        "Actual Healthy:",
        actual_healthy
    )

    print(
        "Actual Stressed:",
        actual_stressed
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("")
print("========================================")
print("3D-CNN training completed")
print("========================================")

print(
    "Model saved to:",
    MODEL_PATH
)

print("")