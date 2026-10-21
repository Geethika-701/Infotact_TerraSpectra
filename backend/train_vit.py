import os

import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from vit_model import TerraSpectraViT


CUBE_PATH = "dataset/raw/mock_hyperspectral_cube.npy"
LABEL_PATH = "dataset/raw/labels.npy"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "terraspectra_vit.pth"
)


class HyperspectralViTDataset(Dataset):

    def __init__(
        self,
        cube,
        labels,
        patch_size=7
    ):

        self.cube = cube
        self.labels = labels
        self.patch_size = patch_size

        margin = patch_size // 2

        self.samples = []

        height, width, _ = cube.shape

        for row in range(
            margin,
            height - margin
        ):

            for col in range(
                margin,
                width - margin
            ):

                label = labels[
                    row,
                    col
                ]

                self.samples.append(
                    (
                        row,
                        col,
                        label
                    )
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        row, col, label = self.samples[index]

        margin = self.patch_size // 2

        patch = self.cube[
            row - margin:row + margin + 1,
            col - margin:col + margin + 1,
            :
        ]

        # H, W, Bands
        # →
        # Bands, H, W

        patch = np.transpose(
            patch,
            (2, 0, 1)
        )

        patch = torch.tensor(
            patch,
            dtype=torch.float32
        )

        # [Bands, H, W]
        # →
        # [1, Bands, H, W]

        patch = patch.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return patch, label


def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    print("Loading hyperspectral dataset...")

    cube = np.load(
        CUBE_PATH
    ).astype(np.float32)

    labels = np.load(
        LABEL_PATH
    )

    print("Cube shape:", cube.shape)
    print("Labels shape:", labels.shape)

    # Global normalization
    mean = cube.mean()
    std = cube.std()

    cube = (
        cube - mean
    ) / (
        std + 1e-8
    )

    dataset = HyperspectralViTDataset(
        cube,
        labels,
        patch_size=7
    )

    print(
        "Training samples:",
        len(dataset)
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Training device:",
        device
    )

    model = TerraSpectraViT(
        bands=224,
        patch_size=7,
        spectral_group_size=16,
        embedding_dim=128,
        num_heads=8,
        num_layers=4,
        num_classes=2
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.01
    )

    epochs = 5

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for patches, targets in loader:

            patches = patches.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(
                patches
            )

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

        accuracy = (
            correct / total
        ) * 100

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {total_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print(
        "Vision Transformer model saved:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()