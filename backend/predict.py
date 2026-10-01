import json
from pathlib import Path

import numpy as np
import torch

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

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILES
# ============================================================

CUBE_PATH = (
    PROCESSED_DIR /
    "normalized_hyperspectral.npy"
)

LABELS_PATH = (
    RAW_DIR /
    "labels.npy"
)

MODEL_PATH = (
    MODEL_DIR /
    "hybrid_3dcnn.pth"
)

OUTPUT_PATH = (
    RESULTS_DIR /
    "predictions.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = 5

BATCH_SIZE = 64

CLASS_NAMES = {
    0: "Healthy",
    1: "Chemically Stressed"
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("TerraSpectra - Week 2 Pixel Prediction")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# VERIFY FILES
# ============================================================

if not CUBE_PATH.exists():

    raise FileNotFoundError(
        f"Normalized cube not found:\n{CUBE_PATH}"
    )


if not LABELS_PATH.exists():

    raise FileNotFoundError(
        f"Labels not found:\n{LABELS_PATH}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}\n\n"
        "Run train.py first."
    )


# ============================================================
# LOAD DATA
# ============================================================

cube = np.load(
    CUBE_PATH
)

labels = np.load(
    LABELS_PATH
)


height, width, spectral_bands = cube.shape


print("\nInput cube:")
print("Height:", height)
print("Width :", width)
print("Bands :", spectral_bands)


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model = Hybrid3DCNN(
    num_classes=2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# PAD CUBE
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
# CREATE ALL PIXEL COORDINATES
# ============================================================

coordinates = []

for y in range(height):

    for x in range(width):

        coordinates.append(
            (y, x)
        )


# ============================================================
# PREDICT IN BATCHES
# ============================================================

all_predictions = []

all_confidences = []


print("\nPredicting pixels...")


for start in range(
    0,
    len(coordinates),
    BATCH_SIZE
):

    batch_coordinates = coordinates[
        start:start + BATCH_SIZE
    ]

    patches = []


    for y, x in batch_coordinates:

        y_p = y + padding

        x_p = x + padding

        half = PATCH_SIZE // 2

        patch = padded_cube[
            y_p - half:y_p + half + 1,
            x_p - half:x_p + half + 1,
            :
        ]

        # [height, width, bands]
        # →
        # [bands, height, width]

        patch = np.transpose(
            patch,
            (2, 0, 1)
        )

        # [bands, height, width]
        # →
        # [1, bands, height, width]

        patch = np.expand_dims(
            patch,
            axis=0
        )

        patches.append(
            patch
        )


    batch = np.stack(
        patches
    )


    batch = torch.tensor(
        batch,
        dtype=torch.float32
    ).to(DEVICE)


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            batch
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )


    all_predictions.extend(
        predictions.cpu().numpy().tolist()
    )

    all_confidences.extend(
        confidences.cpu().numpy().tolist()
    )


# ============================================================
# BUILD JSON
# ============================================================

results = []


for index, (y, x) in enumerate(
    coordinates
):

    prediction = int(
        all_predictions[index]
    )

    confidence = float(
        all_confidences[index]
    )


    result = {
        "x": int(x),
        "y": int(y),
        "prediction": prediction,
        "class": CLASS_NAMES[prediction],
        "confidence": round(
            confidence,
            4
        )
    }


    results.append(
        result
    )


# ============================================================
# SAVE JSON
# ============================================================

output_data = {

    "project": "TerraSpectra",

    "task": "Hyperspectral Crop Disease Forecasting",

    "model": "Hybrid 3D-CNN",

    "input": {
        "height": int(height),
        "width": int(width),
        "spectral_bands": int(spectral_bands),
        "patch_size": PATCH_SIZE
    },

    "classes": CLASS_NAMES,

    "pixels": results
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output_data,
        file,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

prediction_array = np.array(
    all_predictions
)


unique, counts = np.unique(
    prediction_array,
    return_counts=True
)


print("\nPrediction summary")
print("-" * 60)


for class_id, count in zip(
    unique,
    counts
):

    print(
        f"{CLASS_NAMES[int(class_id)]}: "
        f"{int(count)} pixels"
    )


print("\nPrediction completed.")

print(
    f"JSON saved to:\n{OUTPUT_PATH}"
)

print("=" * 60)