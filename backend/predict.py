import json
import os

import numpy as np
import torch

from model import Hybrid3DCNN
from week2_preprocessing import (
    prepare_week2_data
)


MODEL_PATH = (
    "results/training/"
    "hybrid_3dcnn.pth"
)

OUTPUT_PATH = (
    "results/predictions/"
    "predictions.json"
)

BASE_LONGITUDE = 83.2185
BASE_LATITUDE = 17.6868

PIXEL_STEP = 0.00005


def predict():

    print("=" * 60)
    print("TerraSpectra Week 2")
    print("Hybrid 3D-CNN Prediction")
    print("=" * 60)

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "Trained model not found.\n"
            "Run train.py first."
        )

    (
        hyperspectral,
        pca,
        labels,
        coordinates
    ) = prepare_week2_data()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nPrediction device:",
        device
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = Hybrid3DCNN(
        num_classes=2
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------
    # Convert data
    # --------------------------------------------------

    hs_tensor = torch.tensor(
        hyperspectral,
        dtype=torch.float32
    )

    pca_tensor = torch.tensor(
        pca,
        dtype=torch.float32
    )

    predictions = []

    # --------------------------------------------------
    # Batch inference
    # --------------------------------------------------

    batch_size = 32

    total_samples = len(
        hs_tensor
    )

    with torch.no_grad():

        for start in range(
            0,
            total_samples,
            batch_size
        ):

            end = min(
                start + batch_size,
                total_samples
            )

            hs_batch = (
                hs_tensor[start:end]
                .to(device)
            )

            pca_batch = (
                pca_tensor[start:end]
                .to(device)
            )

            outputs = model(
                hs_batch,
                pca_batch
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_classes = torch.argmax(
                probabilities,
                dim=1
            )

            for index in range(
                end - start
            ):

                row, col = coordinates[
                    start + index
                ]

                class_id = int(
                    predicted_classes[index]
                    .item()
                )

                confidence = float(
                    probabilities[
                        index,
                        class_id
                    ].item()
                )

                if class_id == 0:

                    status = "healthy"

                else:

                    status = (
                        "chemically_stressed"
                    )

                longitude = (
                    BASE_LONGITUDE
                    + (
                        col - 32
                    )
                    * PIXEL_STEP
                )

                latitude = (
                    BASE_LATITUDE
                    + (
                        row - 32
                    )
                    * PIXEL_STEP
                )

                predictions.append(
                    {
                        "row": int(row),

                        "column": int(col),

                        "longitude": float(
                            longitude
                        ),

                        "latitude": float(
                            latitude
                        ),

                        "status": status,

                        "class_id": class_id,

                        "probability": confidence
                    }
                )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    healthy_count = sum(
        item["status"] == "healthy"
        for item in predictions
    )

    stressed_count = sum(
        item["status"]
        == "chemically_stressed"
        for item in predictions
    )

    # --------------------------------------------------
    # Output JSON
    # --------------------------------------------------

    output = {
        "model": "Hybrid 3D-CNN",

        "classes": [
            "healthy",
            "chemically_stressed"
        ],

        "source": (
            "mock hyperspectral "
            "GeoTIFF dataset"
        ),

        "total_predictions":
            len(predictions),

        "healthy_count":
            healthy_count,

        "chemically_stressed_count":
            stressed_count,

        "validation_accuracy":
            checkpoint.get(
                "validation_accuracy"
            ),

        "predictions":
            predictions
    }

    os.makedirs(
        "results/predictions",
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print("\nPrediction completed")
    print("---------------------")

    print(
        "Total predictions:",
        len(predictions)
    )

    print(
        "Healthy:",
        healthy_count
    )

    print(
        "Chemically stressed:",
        stressed_count
    )

    print(
        "Validation accuracy:",
        checkpoint.get(
            "validation_accuracy"
        )
    )

    print(
        "Saved:",
        OUTPUT_PATH
    )

    print("=" * 60)


if __name__ == "__main__":
    predict()