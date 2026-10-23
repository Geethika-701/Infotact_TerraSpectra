import numpy as np
import torch

#from model import TerraSpectra3DCNN
from backend.model import TerraSpectra3DCNN

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CUBE_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "raw",
    "mock_hyperspectral_cube.npy"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "terraspectra_3dcnn.pth"
)

LONGITUDE = 83.2185
LATITUDE = 17.6868

PIXEL_SIZE = 0.00008


def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = TerraSpectra3DCNN()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model, device


def predict_grid():

    cube = np.load(
        CUBE_PATH
    ).astype(
        np.float32
    )

    mean = cube.mean()

    std = cube.std()

    cube = (
        cube - mean
    ) / (
        std + 1e-8
    )

    model, device = load_model()

    results = []

    patch_size = 7

    margin = patch_size // 2

    height, width, _ = cube.shape

    with torch.no_grad():

        for row in range(
            margin,
            height - margin,
            2
        ):

            for col in range(
                margin,
                width - margin,
                2
            ):

                patch = cube[
                    row - margin:
                    row + margin + 1,

                    col - margin:
                    col + margin + 1,

                    :
                ]

                patch = np.transpose(
                    patch,
                    (2, 0, 1)
                )

                tensor = torch.tensor(
                    patch,
                    dtype=torch.float32
                )

                tensor = tensor.unsqueeze(
                    0
                )

                tensor = tensor.unsqueeze(
                    0
                )

                tensor = tensor.to(
                    device
                )

                output = model(
                    tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                risk = probabilities[
                    0,
                    1
                ].item()

                lon = (
                    LONGITUDE
                    +
                    (
                        col - width / 2
                    )
                    *
                    PIXEL_SIZE
                )

                lat = (
                    LATITUDE
                    +
                    (
                        row - height / 2
                    )
                    *
                    PIXEL_SIZE
                )

                half = PIXEL_SIZE

                polygon = [

                    [
                        lon - half,
                        lat - half
                    ],

                    [
                        lon + half,
                        lat - half
                    ],

                    [
                        lon + half,
                        lat + half
                    ],

                    [
                        lon - half,
                        lat + half
                    ]

                ]

                if risk >= 0.7:

                    label = "High Risk"

                elif risk >= 0.4:

                    label = "Moderate"

                else:

                    label = "Healthy"

                results.append({

                    "row": row,

                    "col": col,

                    "longitude": lon,

                    "latitude": lat,

                    "risk": round(
                        risk,
                        4
                    ),

                    "label": label,

                    "polygon": polygon

                })

    return results