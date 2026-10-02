import os

import numpy as np
import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)

from torch import nn

from model import Hybrid3DCNN
from week2_preprocessing import (
    prepare_week2_data
)


MODEL_PATH = (
    "results/training/"
    "hybrid_3dcnn.pth"
)

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


def train_model():

    print("=" * 60)
    print("TerraSpectra Week 2")
    print("Hybrid 3D-CNN Training")
    print("=" * 60)

    (
        hyperspectral,
        pca,
        labels,
        coordinates
    ) = prepare_week2_data()

    # --------------------------------------------------
    # Convert numpy arrays to tensors
    # --------------------------------------------------

    hs_tensor = torch.tensor(
        hyperspectral,
        dtype=torch.float32
    )

    pca_tensor = torch.tensor(
        pca,
        dtype=torch.float32
    )

    label_tensor = torch.tensor(
        labels,
        dtype=torch.long
    )

    dataset = TensorDataset(
        hs_tensor,
        pca_tensor,
        label_tensor
    )

    # --------------------------------------------------
    # Train / validation split
    # --------------------------------------------------

    total_size = len(dataset)

    train_size = int(
        total_size * 0.8
    )

    validation_size = (
        total_size - train_size
    )

    train_dataset, validation_dataset = (
        random_split(
            dataset,
            [
                train_size,
                validation_size
            ],
            generator=torch.Generator().manual_seed(42)
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nTraining device:",
        device
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = Hybrid3DCNN(
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_accuracy = 0.0

    os.makedirs(
        "results/training",
        exist_ok=True
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0

        correct = 0
        total = 0

        for (
            hs_batch,
            pca_batch,
            labels_batch
        ) in train_loader:

            hs_batch = hs_batch.to(
                device
            )

            pca_batch = pca_batch.to(
                device
            )

            labels_batch = labels_batch.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                hs_batch,
                pca_batch
            )

            loss = criterion(
                outputs,
                labels_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * labels_batch.size(0)
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            correct += (
                predictions == labels_batch
            ).sum().item()

            total += (
                labels_batch.size(0)
            )

        train_loss = (
            total_loss / total
        )

        train_accuracy = (
            correct / total
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        model.eval()

        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for (
                hs_batch,
                pca_batch,
                labels_batch
            ) in validation_loader:

                hs_batch = hs_batch.to(
                    device
                )

                pca_batch = pca_batch.to(
                    device
                )

                labels_batch = labels_batch.to(
                    device
                )

                outputs = model(
                    hs_batch,
                    pca_batch
                )

                predictions = (
                    torch.argmax(
                        outputs,
                        dim=1
                    )
                )

                validation_correct += (
                    predictions == labels_batch
                ).sum().item()

                validation_total += (
                    labels_batch.size(0)
                )

        validation_accuracy = (
            validation_correct
            / validation_total
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Training Loss: {train_loss:.4f}"
        )

        print(
            f"Training Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        # Save best model

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "validation_accuracy":
                        validation_accuracy,

                    "classes": [
                        "healthy",
                        "chemically_stressed"
                    ]
                },
                MODEL_PATH
            )

            print(
                "Best model saved."
            )

    print("\n" + "=" * 60)

    print(
        "Training completed."
    )

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy:.4f}"
    )

    print(
        "Model saved:",
        MODEL_PATH
    )

    print("=" * 60)


if __name__ == "__main__":
    train_model()