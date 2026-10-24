import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from backend.model import TerraSpectra3DCNN
from backend.vit_model import TerraSpectraViT


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

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

CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "terraspectra_3dcnn.pth"
)

VIT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "terraspectra_vit.pth"
)


def normalize_cube(cube):
    cube = cube.astype(np.float32)

    minimum = float(cube.min())
    maximum = float(cube.max())

    if maximum - minimum < 1e-8:
        return np.zeros_like(
            cube,
            dtype=np.float32
        )

    cube = (
        cube - minimum
    ) / (
        maximum - minimum
    )

    return cube.astype(np.float32)


def create_patch(cube, row, col, patch_size=7):
    height = cube.shape[0]
    width = cube.shape[1]

    radius = patch_size // 2

    padded = np.pad(
        cube,
        (
            (radius, radius),
            (radius, radius),
            (0, 0)
        ),
        mode="reflect"
    )

    center_row = row + radius
    center_col = col + radius

    patch = padded[
        center_row - radius:
        center_row + radius + 1,
        center_col - radius:
        center_col + radius + 1,
        :
    ]

    return patch.astype(np.float32)


def load_models():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    cnn = TerraSpectra3DCNN()
    vit = TerraSpectraViT()

    cnn_checkpoint = torch.load(
        CNN_MODEL_PATH,
        map_location=device
    )

    vit_checkpoint = torch.load(
        VIT_MODEL_PATH,
        map_location=device
    )

    if (
        isinstance(cnn_checkpoint, dict)
        and "model_state_dict" in cnn_checkpoint
    ):
        cnn.load_state_dict(
            cnn_checkpoint["model_state_dict"]
        )
    else:
        cnn.load_state_dict(
            cnn_checkpoint
        )

    if (
        isinstance(vit_checkpoint, dict)
        and "model_state_dict" in vit_checkpoint
    ):
        vit.load_state_dict(
            vit_checkpoint["model_state_dict"]
        )
    else:
        vit.load_state_dict(
            vit_checkpoint
        )

    cnn.to(device)
    vit.to(device)

    cnn.eval()
    vit.eval()

    return cnn, vit, device


def prepare_cnn_input(patches):
    tensor = torch.tensor(
        np.asarray(patches),
        dtype=torch.float32
    )

    # 3D-CNN input:
    #
    # [batch, 7, 7, 224]
    #
    # becomes:
    #
    # [batch, 1, 7, 7, 224]

    tensor = tensor.unsqueeze(1)

    return tensor


def prepare_vit_input(patches):
    tensor = torch.tensor(
        np.asarray(patches),
        dtype=torch.float32
    )

    # ViT input:
    #
    # [batch, 7, 7, 224]

    return tensor


def calculate_metrics(
    true_labels,
    predictions
):
    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    true_labels = np.asarray(
        true_labels
    )

    predictions = np.asarray(
        predictions
    )

    tp = int(
        np.sum(
            (true_labels == 1)
            &
            (predictions == 1)
        )
    )

    tn = int(
        np.sum(
            (true_labels == 0)
            &
            (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (true_labels == 0)
            &
            (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (true_labels == 1)
            &
            (predictions == 0)
        )
    )

    return {
        "accuracy": round(
            float(accuracy),
            4
        ),
        "precision": round(
            float(precision),
            4
        ),
        "recall": round(
            float(recall),
            4
        ),
        "f1_score": round(
            float(f1),
            4
        ),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }


def select_balanced_samples(
    labels,
    sample_limit=1000,
    seed=42
):
    """
    Creates a class-balanced validation set.

    For a 1000-sample validation:
        500 Healthy
        500 Chemically Stressed

    If the dataset contains fewer samples of either class,
    the maximum possible balanced subset is used.
    """

    rng = np.random.default_rng(
        seed
    )

    healthy_coordinates = []
    stressed_coordinates = []

    height = labels.shape[0]
    width = labels.shape[1]

    for row in range(height):
        for col in range(width):

            label = int(
                labels[row, col]
            )

            if label == 0:
                healthy_coordinates.append(
                    (row, col)
                )

            elif label == 1:
                stressed_coordinates.append(
                    (row, col)
                )

    if len(healthy_coordinates) == 0:
        raise ValueError(
            "No healthy samples were found."
        )

    if len(stressed_coordinates) == 0:
        raise ValueError(
            "No chemically stressed samples were found."
        )

    half = sample_limit // 2

    healthy_count = min(
        half,
        len(healthy_coordinates)
    )

    stressed_count = min(
        half,
        len(stressed_coordinates)
    )

    # Keep both classes equal.
    balanced_count = min(
        healthy_count,
        stressed_count
    )

    healthy_indices = rng.choice(
        len(healthy_coordinates),
        size=balanced_count,
        replace=False
    )

    stressed_indices = rng.choice(
        len(stressed_coordinates),
        size=balanced_count,
        replace=False
    )

    selected_coordinates = []

    for index in healthy_indices:
        selected_coordinates.append(
            healthy_coordinates[int(index)]
        )

    for index in stressed_indices:
        selected_coordinates.append(
            stressed_coordinates[int(index)]
        )

    # Shuffle the final balanced set.
    rng.shuffle(
        selected_coordinates
    )

    return (
        selected_coordinates,
        balanced_count,
        len(healthy_coordinates),
        len(stressed_coordinates)
    )


def validate_models(
    sample_limit=1000,
    batch_size=32
):
    if not os.path.exists(
        CUBE_PATH
    ):
        raise FileNotFoundError(
            f"Hyperspectral cube not found: "
            f"{CUBE_PATH}"
        )

    if not os.path.exists(
        LABEL_PATH
    ):
        raise FileNotFoundError(
            f"Labels not found: "
            f"{LABEL_PATH}"
        )

    if not os.path.exists(
        CNN_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"CNN model not found: "
            f"{CNN_MODEL_PATH}"
        )

    if not os.path.exists(
        VIT_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"ViT model not found: "
            f"{VIT_MODEL_PATH}"
        )

    cube = np.load(
        CUBE_PATH
    )

    labels = np.load(
        LABEL_PATH
    )

    cube = normalize_cube(
        cube
    )

    (
        selected_coordinates,
        balanced_count,
        total_healthy,
        total_stressed
    ) = select_balanced_samples(
        labels,
        sample_limit=sample_limit,
        seed=42
    )

    patches = []
    true_labels = []

    for row, col in selected_coordinates:

        patch = create_patch(
            cube,
            row,
            col,
            patch_size=7
        )

        patches.append(
            patch
        )

        true_labels.append(
            int(labels[row, col])
        )

    cnn, vit, device = load_models()

    cnn_predictions = []
    vit_predictions = []
    fusion_predictions = []

    total_samples = len(
        patches
    )

    with torch.no_grad():

        start = 0

        while start < total_samples:

            end = min(
                start + batch_size,
                total_samples
            )

            batch_patches = patches[
                start:end
            ]

            cnn_input = prepare_cnn_input(
                batch_patches
            )

            vit_input = prepare_vit_input(
                batch_patches
            )

            cnn_input = cnn_input.to(
                device
            )

            vit_input = vit_input.to(
                device
            )

            cnn_output = cnn(
                cnn_input
            )

            vit_output = vit(
                vit_input
            )

            cnn_probability = torch.softmax(
                cnn_output,
                dim=1
            )[:, 1]

            vit_probability = torch.softmax(
                vit_output,
                dim=1
            )[:, 1]

            fused_probability = (
                0.60 * cnn_probability
                +
                0.40 * vit_probability
            )

            cnn_batch_predictions = (
                cnn_probability >= 0.5
            ).long().cpu().numpy()

            vit_batch_predictions = (
                vit_probability >= 0.5
            ).long().cpu().numpy()

            fusion_batch_predictions = (
                fused_probability >= 0.5
            ).long().cpu().numpy()

            for value in (
                cnn_batch_predictions
            ):
                cnn_predictions.append(
                    int(value)
                )

            for value in (
                vit_batch_predictions
            ):
                vit_predictions.append(
                    int(value)
                )

            for value in (
                fusion_batch_predictions
            ):
                fusion_predictions.append(
                    int(value)
                )

            start = end

    cnn_metrics = calculate_metrics(
        true_labels,
        cnn_predictions
    )

    vit_metrics = calculate_metrics(
        true_labels,
        vit_predictions
    )

    fusion_metrics = calculate_metrics(
        true_labels,
        fusion_predictions
    )

    cnn_accuracy = (
        cnn_metrics["accuracy"]
    )

    vit_accuracy = (
        vit_metrics["accuracy"]
    )

    fusion_accuracy = (
        fusion_metrics["accuracy"]
    )

    improvement_vs_cnn = (
        fusion_accuracy
        -
        cnn_accuracy
    ) * 100

    improvement_vs_vit = (
        fusion_accuracy
        -
        vit_accuracy
    ) * 100

    return {
        "status": "success",

        "validation": {
            "samples_evaluated": total_samples,
            "healthy_samples": balanced_count,
            "stressed_samples": balanced_count,
            "sample_strategy": (
                "Class-balanced deterministic "
                "sampling with seed 42"
            ),
            "original_dataset_distribution": {
                "healthy_pixels": total_healthy,
                "stressed_pixels": total_stressed
            },
            "patch_size": "7x7",
            "spectral_bands": 224,
            "batch_size": batch_size,
            "device": str(device)
        },

        "cnn": cnn_metrics,

        "vision_transformer": vit_metrics,

        "fusion": fusion_metrics,

        "fusion_improvement": {
            "percentage_points_vs_cnn": round(
                float(improvement_vs_cnn),
                2
            ),
            "percentage_points_vs_vit": round(
                float(improvement_vs_vit),
                2
            )
        },

        "model_configuration": {
            "cnn_weight": 0.60,
            "vit_weight": 0.40,
            "fusion_method": (
                "Weighted probability fusion"
            ),
            "classes": [
                "Healthy",
                "Chemically Stressed"
            ]
        },

        "interpretation": {
            "status": (
                "Class-balanced validation completed"
            ),
            "note": (
                "Metrics are measured on the "
                "synthetic TerraSpectra dataset. "
                "Class balancing prevents the "
                "majority healthy class from "
                "artificially inflating accuracy."
            ),
            "important": (
                "The validation samples come "
                "from the same synthetic dataset "
                "used during model development. "
                "Therefore these results should "
                "not be described as an independent "
                "held-out generalization benchmark."
            )
        }
    }