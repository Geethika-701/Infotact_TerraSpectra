import os
import numpy as np
import torch

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

PATCH_SIZE = 7

CNN_WEIGHT = 0.60
VIT_WEIGHT = 0.40


LONGITUDE = 83.2185
LATITUDE = 17.6868
PIXEL_SIZE = 0.00008


def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_models():

    device = get_device()

    cnn = TerraSpectra3DCNN()

    cnn.load_state_dict(
        torch.load(
            CNN_MODEL_PATH,
            map_location=device
        )
    )

    cnn = cnn.to(device)
    cnn.eval()

    vit = TerraSpectraViT(
        bands=224,
        patch_size=7,
        spectral_group_size=16,
        embedding_dim=128,
        num_heads=8,
        num_layers=4,
        num_classes=2
    )

    vit.load_state_dict(
        torch.load(
            VIT_MODEL_PATH,
            map_location=device
        )
    )

    vit = vit.to(device)
    vit.eval()

    return cnn, vit, device


def normalize_cube(cube):

    cube = cube.astype(
        np.float32
    )

    mean = cube.mean()
    std = cube.std()

    cube = (
        cube - mean
    ) / (
        std + 1e-8
    )

    return cube


def create_patch(
    cube,
    row,
    col,
    patch_size=7
):

    margin = patch_size // 2

    height, width, _ = cube.shape

    padded = np.pad(
        cube,
        (
            (margin, margin),
            (margin, margin),
            (0, 0)
        ),
        mode="edge"
    )

    padded_row = row + margin
    padded_col = col + margin

    patch = padded[
        padded_row - margin:
        padded_row + margin + 1,

        padded_col - margin:
        padded_col + margin + 1,

        :
    ]

    return patch


def prepare_batch(
    patches
):

    batch = np.stack(
        patches
    )

    # H,W,B
    # ->
    # B,H,W

    batch = np.transpose(
        batch,
        (0, 3, 1, 2)
    )

    tensor = torch.tensor(
        batch,
        dtype=torch.float32
    )

    # CNN / ViT expect:
    # batch, 1, bands, H, W

    tensor = tensor.unsqueeze(1)

    return tensor


def predict_batch(
    cnn,
    vit,
    device,
    patches
):

    tensor = prepare_batch(
        patches
    )

    tensor = tensor.to(device)

    with torch.no_grad():

        cnn_output = cnn(
            tensor
        )

        vit_output = vit(
            tensor
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
            CNN_WEIGHT * cnn_probability
            +
            VIT_WEIGHT * vit_probability
        )

    return (
        cnn_probability.cpu().numpy(),
        vit_probability.cpu().numpy(),
        fused_probability.cpu().numpy()
    )


def classify_risk(
    risk
):

    if risk >= 0.7:
        return "High Risk"

    if risk >= 0.4:
        return "Moderate"

    return "Healthy"


def coordinates_for_pixel(
    row,
    col,
    height,
    width
):

    longitude = (
        LONGITUDE
        +
        (col - width / 2)
        * PIXEL_SIZE
    )

    latitude = (
        LATITUDE
        +
        (row - height / 2)
        * PIXEL_SIZE
    )

    half = PIXEL_SIZE

    polygon = [

        [
            longitude - half,
            latitude - half
        ],

        [
            longitude + half,
            latitude - half
        ],

        [
            longitude + half,
            latitude + half
        ],

        [
            longitude - half,
            latitude + half
        ]
    ]

    return (
        longitude,
        latitude,
        polygon
    )


def predict_fused_grid(
    stride=2,
    batch_size=16
):

    cube = np.load(
        CUBE_PATH
    )

    cube = normalize_cube(
        cube
    )

    height, width, bands = cube.shape

    cnn, vit, device = load_models()

    results = []

    patch_size = PATCH_SIZE

    rows = range(
        0,
        height,
        stride
    )

    cols = range(
        0,
        width,
        stride
    )

    coordinates = []

    patches = []

    for row in rows:

        for col in cols:

            patch = create_patch(
                cube,
                row,
                col,
                patch_size
            )

            patches.append(
                patch
            )

            coordinates.append(
                (row, col)
            )

            if len(patches) >= batch_size:

                cnn_probs, vit_probs, fused_probs = predict_batch(
                    cnn,
                    vit,
                    device,
                    patches
                )

                for index in range(
                    len(patches)
                ):

                    current_row, current_col = (
                        coordinates[index]
                    )

                    cnn_risk = float(
                        cnn_probs[index]
                    )

                    vit_risk = float(
                        vit_probs[index]
                    )

                    fused_risk = float(
                        fused_probs[index]
                    )

                    longitude, latitude, polygon = (
                        coordinates_for_pixel(
                            current_row,
                            current_col,
                            height,
                            width
                        )
                    )

                    results.append({

                        "row":
                            current_row,

                        "col":
                            current_col,

                        "longitude":
                            longitude,

                        "latitude":
                            latitude,

                        "cnn_risk":
                            round(
                                cnn_risk,
                                4
                            ),

                        "vit_risk":
                            round(
                                vit_risk,
                                4
                            ),

                        "risk":
                            round(
                                fused_risk,
                                4
                            ),

                        "label":
                            classify_risk(
                                fused_risk
                            ),

                        "model":
                            "3D-CNN + Vision Transformer",

                        "polygon":
                            polygon
                    })

                patches = []
                coordinates = []

    # Remaining patches

    if patches:

        cnn_probs, vit_probs, fused_probs = predict_batch(
            cnn,
            vit,
            device,
            patches
        )

        for index in range(
            len(patches)
        ):

            current_row, current_col = (
                coordinates[index]
            )

            cnn_risk = float(
                cnn_probs[index]
            )

            vit_risk = float(
                vit_probs[index]
            )

            fused_risk = float(
                fused_probs[index]
            )

            longitude, latitude, polygon = (
                coordinates_for_pixel(
                    current_row,
                    current_col,
                    height,
                    width
                )
            )

            results.append({

                "row":
                    current_row,

                "col":
                    current_col,

                "longitude":
                    longitude,

                "latitude":
                    latitude,

                "cnn_risk":
                    round(
                        cnn_risk,
                        4
                    ),

                "vit_risk":
                    round(
                        vit_risk,
                        4
                    ),

                "risk":
                    round(
                        fused_risk,
                        4
                    ),

                "label":
                    classify_risk(
                        fused_risk
                    ),

                "model":
                    "3D-CNN + Vision Transformer",

                "polygon":
                    polygon
            })

    return results


def predict_single_patch(
    patch
):

    cnn, vit, device = load_models()

    cnn_probs, vit_probs, fused_probs = predict_batch(
        cnn,
        vit,
        device,
        [patch]
    )

    return {
        "cnn_risk":
            float(cnn_probs[0]),

        "vit_risk":
            float(vit_probs[0]),

        "fused_risk":
            float(fused_probs[0])
    }