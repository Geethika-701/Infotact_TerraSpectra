import os
import inspect
import numpy as np
import torch

from backend.model import TerraSpectra3DCNN
from backend.vit_model import TerraSpectraViT


# =========================================================
# PATHS
# =========================================================

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


# =========================================================
# MODEL CONFIGURATION
# =========================================================

PATCH_SIZE = 7
BANDS = 224

CNN_WEIGHT = 0.60
VIT_WEIGHT = 0.40


# =========================================================
# LOAD HYPERSPECTRAL CUBE
# =========================================================

def load_cube():

    if not os.path.exists(
        CUBE_PATH
    ):
        raise FileNotFoundError(
            f"Hyperspectral cube not found: {CUBE_PATH}"
        )

    cube = np.load(
        CUBE_PATH
    ).astype(
        np.float32
    )

    if cube.ndim != 3:

        raise ValueError(
            f"Expected a 3D hyperspectral cube, "
            f"got shape {cube.shape}"
        )

    if cube.shape[2] != BANDS:

        raise ValueError(
            f"Expected {BANDS} spectral bands, "
            f"got {cube.shape[2]}"
        )

    return cube


# =========================================================
# NORMALIZE COMPLETE CUBE
# =========================================================

def normalize_cube(
    cube
):

    cube = np.asarray(
        cube,
        dtype=np.float32
    )

    cube_min = float(
        cube.min()
    )

    cube_max = float(
        cube.max()
    )

    if cube_max - cube_min < 1e-8:

        return np.zeros_like(
            cube,
            dtype=np.float32
        )

    normalized = (
        cube - cube_min
    ) / (
        cube_max - cube_min
    )

    return normalized.astype(
        np.float32
    )


# =========================================================
# NORMALIZE INDIVIDUAL PATCH
# =========================================================

def normalize_patch(
    patch
):

    patch = np.asarray(
        patch,
        dtype=np.float32
    )

    patch_min = float(
        patch.min()
    )

    patch_max = float(
        patch.max()
    )

    if patch_max - patch_min < 1e-8:

        return np.zeros_like(
            patch,
            dtype=np.float32
        )

    normalized = (
        patch - patch_min
    ) / (
        patch_max - patch_min
    )

    return normalized.astype(
        np.float32
    )


# =========================================================
# CREATE VISION TRANSFORMER
# =========================================================

def create_vit_model():

    signature = inspect.signature(
        TerraSpectraViT.__init__
    )

    parameters = signature.parameters

    possible_arguments = {

        "bands":
            224,

        "patch_size":
            7,

        "embed_dim":
            128,

        "num_heads":
            8,

        "num_layers":
            4,

        "num_classes":
            2,

        "input_features":
            784,

        "spectral_group_size":
            16
    }

    kwargs = {}

    for name in parameters:

        if name == "self":
            continue

        if name in possible_arguments:

            kwargs[name] = (
                possible_arguments[name]
            )

    print(
        "TerraSpectraViT constructor arguments:",
        kwargs
    )

    model = TerraSpectraViT(
        **kwargs
    )

    return model


# =========================================================
# LOAD CHECKPOINT SAFELY
# =========================================================

def load_state_dict_safely(
    model,
    model_path,
    device,
    model_name
):

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"{model_name} model not found: "
            f"{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    print(
        f"{model_name} checkpoint loaded successfully."
    )

    return model


# =========================================================
# LOAD BOTH MODELS
# =========================================================

def load_models():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Inference device:",
        device
    )

    # -----------------------------------------------------
    # 3D CNN
    # -----------------------------------------------------

    cnn = TerraSpectra3DCNN()

    cnn = load_state_dict_safely(
        cnn,
        CNN_MODEL_PATH,
        device,
        "3D-CNN"
    )

    cnn.to(
        device
    )

    cnn.eval()

    # -----------------------------------------------------
    # Vision Transformer
    # -----------------------------------------------------

    vit = create_vit_model()

    vit = load_state_dict_safely(
        vit,
        VIT_MODEL_PATH,
        device,
        "Vision Transformer"
    )

    vit.to(
        device
    )

    vit.eval()

    return (
        cnn,
        vit,
        device
    )


# =========================================================
# CREATE 7x7 HYPERSPECTRAL PATCH
# =========================================================

def create_patch(
    cube,
    center_y,
    center_x,
    patch_size=7
):

    cube = np.asarray(
        cube,
        dtype=np.float32
    )

    if cube.ndim != 3:

        raise ValueError(
            f"Expected cube with 3 dimensions, "
            f"got {cube.shape}"
        )

    height = cube.shape[0]
    width = cube.shape[1]
    bands = cube.shape[2]

    if bands != BANDS:

        raise ValueError(
            f"Expected {BANDS} spectral bands, "
            f"got {bands}"
        )

    if patch_size <= 0:

        raise ValueError(
            "patch_size must be greater than zero."
        )

    if patch_size % 2 == 0:

        raise ValueError(
            "patch_size must be an odd number."
        )

    half = patch_size // 2

    center_y = int(
        center_y
    )

    center_x = int(
        center_x
    )

    # -----------------------------------------------------
    # Edge-safe padding
    #
    # This allows Week 4 tiled inference to process
    # pixels along all four image boundaries.
    # -----------------------------------------------------

    padded_cube = np.pad(
        cube,
        (
            (half, half),
            (half, half),
            (0, 0)
        ),
        mode="edge"
    )

    padded_y = (
        center_y + half
    )

    padded_x = (
        center_x + half
    )

    y_start = (
        padded_y - half
    )

    y_end = (
        padded_y + half + 1
    )

    x_start = (
        padded_x - half
    )

    x_end = (
        padded_x + half + 1
    )

    patch = padded_cube[
        y_start:y_end,
        x_start:x_end,
        :
    ]

    expected_shape = (
        patch_size,
        patch_size,
        BANDS
    )

    if patch.shape != expected_shape:

        raise ValueError(
            f"Unexpected patch shape {patch.shape}. "
            f"Expected {expected_shape}."
        )

    return patch.astype(
        np.float32
    )


# =========================================================
# EXTRACT PATCH FROM NORMALIZED CUBE
# =========================================================

def extract_patch(
    cube,
    center_y,
    center_x
):

    return create_patch(
        cube,
        center_y,
        center_x,
        PATCH_SIZE
    )


# =========================================================
# PREPARE CNN BATCH
# =========================================================

def prepare_cnn_batch(
    patches
):

    array = np.asarray(
        patches,
        dtype=np.float32
    )

    if array.ndim != 4:

        raise ValueError(
            "CNN patches must have shape "
            "[batch, height, width, bands]."
        )

    tensor = torch.from_numpy(
        array
    )

    # [B, 7, 7, 224]
    #
    # becomes
    #
    # [B, 1, 7, 7, 224]

    tensor = tensor.unsqueeze(
        1
    )

    return tensor


# =========================================================
# PREPARE VIT BATCH
# =========================================================

def prepare_vit_batch(
    patches
):

    array = np.asarray(
        patches,
        dtype=np.float32
    )

    if array.ndim != 4:

        raise ValueError(
            "ViT patches must have shape "
            "[batch, height, width, bands]."
        )

    tensor = torch.from_numpy(
        array
    )

    return tensor


# =========================================================
# GET CHEMICAL-STRESS PROBABILITY
# =========================================================

def get_stressed_probability(
    logits
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    return probabilities[
        :,
        1
    ]


# =========================================================
# CLASSIFY RISK
# =========================================================

def classify_risk(
    probability
):

    probability = float(
        probability
    )

    if probability >= 0.70:

        return "High Risk"

    if probability >= 0.40:

        return "Moderate Risk"

    return "Healthy"


# =========================================================
# BATCH PREDICTION
# =========================================================

def predict_batch(
    cnn,
    vit,
    device,
    patches
):
    """
    Run a batch through both models.

    Returns:

        cnn_probabilities
        vit_probabilities
        fused_probabilities
    """

    if len(patches) == 0:

        return (
            np.asarray(
                [],
                dtype=np.float32
            ),
            np.asarray(
                [],
                dtype=np.float32
            ),
            np.asarray(
                [],
                dtype=np.float32
            )
        )

    normalized_patches = []

    for patch in patches:

        patch = np.asarray(
            patch,
            dtype=np.float32
        )

        expected_shape = (
            PATCH_SIZE,
            PATCH_SIZE,
            BANDS
        )

        if patch.shape != expected_shape:

            raise ValueError(
                f"Expected patch shape "
                f"{expected_shape}, "
                f"got {patch.shape}"
            )

        normalized_patch = normalize_patch(
            patch
        )

        normalized_patches.append(
            normalized_patch
        )

    cnn_input = prepare_cnn_batch(
        normalized_patches
    ).to(
        device
    )

    vit_input = prepare_vit_batch(
        normalized_patches
    ).to(
        device
    )

    with torch.no_grad():

        cnn_logits = cnn(
            cnn_input
        )

        vit_logits = vit(
            vit_input
        )

        cnn_risk = get_stressed_probability(
            cnn_logits
        )

        vit_risk = get_stressed_probability(
            vit_logits
        )

    cnn_risk = (
        cnn_risk
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    vit_risk = (
        vit_risk
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    fused_risk = (
        CNN_WEIGHT * cnn_risk
        +
        VIT_WEIGHT * vit_risk
    )

    fused_risk = fused_risk.astype(
        np.float32
    )

    return (
        cnn_risk,
        vit_risk,
        fused_risk
    )


# =========================================================
# SINGLE PATCH PREDICTION
# =========================================================

def predict_single_patch(
    patch,
    cnn=None,
    vit=None,
    device=None
):

    patch = np.asarray(
        patch,
        dtype=np.float32
    )

    expected_shape = (
        PATCH_SIZE,
        PATCH_SIZE,
        BANDS
    )

    if patch.shape != expected_shape:

        raise ValueError(
            f"Expected patch shape "
            f"{expected_shape}, "
            f"got {patch.shape}"
        )

    if (
        cnn is None
        or
        vit is None
        or
        device is None
    ):

        cnn, vit, device = load_models()

    cnn_risk, vit_risk, fused_risk = predict_batch(
        cnn,
        vit,
        device,
        [patch]
    )

    cnn_probability = float(
        cnn_risk[0]
    )

    vit_probability = float(
        vit_risk[0]
    )

    fused_probability = float(
        fused_risk[0]
    )

    return {

        "cnn_risk":
            round(
                cnn_probability,
                6
            ),

        "vit_risk":
            round(
                vit_probability,
                6
            ),

        "risk":
            round(
                fused_probability,
                6
            ),

        "label":
            classify_risk(
                fused_probability
            ),

        "cnn_weight":
            CNN_WEIGHT,

        "vit_weight":
            VIT_WEIGHT
    }


# =========================================================
# FUSED GRID PREDICTION
# =========================================================

def predict_fused_grid(
    stride=2,
    batch_size=16
):

    cube = load_cube()

    cube = normalize_cube(
        cube
    )

    height = cube.shape[0]
    width = cube.shape[1]

    cnn, vit, device = load_models()

    half = PATCH_SIZE // 2

    centers = []

    y = half

    while y < height - half:

        x = half

        while x < width - half:

            centers.append(
                (
                    y,
                    x
                )
            )

            x += stride

        y += stride

    results = []

    total_centers = len(
        centers
    )

    start_index = 0

    while start_index < total_centers:

        end_index = min(
            start_index + batch_size,
            total_centers
        )

        batch_centers = centers[
            start_index:end_index
        ]

        patches = []

        for center in batch_centers:

            patch = extract_patch(
                cube,
                center[0],
                center[1]
            )

            patches.append(
                patch
            )

        cnn_risk, vit_risk, fused_risk = predict_batch(
            cnn,
            vit,
            device,
            patches
        )

        i = 0

        while i < len(
            batch_centers
        ):

            y_coord = (
                batch_centers[i][0]
            )

            x_coord = (
                batch_centers[i][1]
            )

            cnn_probability = float(
                cnn_risk[i]
            )

            vit_probability = float(
                vit_risk[i]
            )

            fused_probability = float(
                fused_risk[i]
            )

            label = classify_risk(
                fused_probability
            )

            polygon = [

                [
                    float(
                        x_coord - half
                    ),
                    float(
                        y_coord - half
                    )
                ],

                [
                    float(
                        x_coord + half
                    ),
                    float(
                        y_coord - half
                    )
                ],

                [
                    float(
                        x_coord + half
                    ),
                    float(
                        y_coord + half
                    )
                ],

                [
                    float(
                        x_coord - half
                    ),
                    float(
                        y_coord + half
                    )
                ]
            ]

            results.append({

                "row":
                    int(
                        y_coord
                    ),

                "col":
                    int(
                        x_coord
                    ),

                "cnn_risk":
                    round(
                        cnn_probability,
                        6
                    ),

                "vit_risk":
                    round(
                        vit_probability,
                        6
                    ),

                "risk":
                    round(
                        fused_probability,
                        6
                    ),

                "label":
                    label,

                "polygon":
                    polygon
            })

            i += 1

        start_index = end_index

    return results


# =========================================================
# PREDICTION SUMMARY
# =========================================================

def prediction_summary(
    predictions
):

    total = len(
        predictions
    )

    high = 0
    moderate = 0
    healthy = 0

    max_risk = 0.0

    cnn_sum = 0.0
    vit_sum = 0.0
    fused_sum = 0.0

    for item in predictions:

        risk = float(
            item["risk"]
        )

        cnn_risk = float(
            item["cnn_risk"]
        )

        vit_risk = float(
            item["vit_risk"]
        )

        if risk >= 0.70:

            high += 1

        elif risk >= 0.40:

            moderate += 1

        else:

            healthy += 1

        if risk > max_risk:

            max_risk = risk

        cnn_sum += cnn_risk
        vit_sum += vit_risk
        fused_sum += risk

    if total > 0:

        cnn_mean = (
            cnn_sum /
            total
        )

        vit_mean = (
            vit_sum /
            total
        )

        fused_mean = (
            fused_sum /
            total
        )

    else:

        cnn_mean = 0.0
        vit_mean = 0.0
        fused_mean = 0.0

    return {

        "total_predictions":
            total,

        "high_risk":
            high,

        "moderate_risk":
            moderate,

        "healthy":
            healthy,

        "max_risk":
            round(
                max_risk,
                6
            ),

        "mean_cnn_risk":
            round(
                cnn_mean,
                6
            ),

        "mean_vit_risk":
            round(
                vit_mean,
                6
            ),

        "mean_fused_risk":
            round(
                fused_mean,
                6
            )
    }


# =========================================================
# PUBLIC FUSED INFERENCE
# =========================================================

def run_fused_inference(
    stride=2,
    batch_size=16
):

    predictions = predict_fused_grid(
        stride=stride,
        batch_size=batch_size
    )

    summary = prediction_summary(
        predictions
    )

    return {

        "status":
            "success",

        "predictions":
            predictions,

        "summary":
            summary,

        "model": {

            "cnn_weight":
                CNN_WEIGHT,

            "vit_weight":
                VIT_WEIGHT,

            "fusion_method":
                "Weighted probability fusion",

            "classes": [

                "Healthy",

                "Chemically Stressed"
            ]
        }
    }


# =========================================================
# COMPATIBILITY WRAPPER
# =========================================================

def predict_fused(
    stride=2,
    batch_size=16
):

    return run_fused_inference(
        stride=stride,
        batch_size=batch_size
    )


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print(
        "Running TerraSpectra fused inference..."
    )

    output = run_fused_inference(
        stride=2,
        batch_size=16
    )

    print(
        "Inference completed."
    )

    print(
        "Prediction summary:"
    )

    print(
        output["summary"]
    )