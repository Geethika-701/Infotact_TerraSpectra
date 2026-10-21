import os
import numpy as np

from read_hyperspectral import read_hyperspectral
from preprocessing import normalize_cube


PATCH_SIZE = 11


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "mock_hyperspectral.tif"
)

LABEL_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "labels.npy"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def create_patches():

    cube, _, _, _, _ = \
        read_hyperspectral(
            INPUT_FILE
        )

    labels = np.load(
        LABEL_FILE
    )

    cube = normalize_cube(
        cube
    )

    height, width, bands = \
        cube.shape

    half = PATCH_SIZE // 2

    X = []

    y = []


    # --------------------------------------------------------
    # Extract patches
    # --------------------------------------------------------

    for row in range(
        half,
        height - half
    ):

        for col in range(
            half,
            width - half
        ):

            patch = cube[
                row-half:row+half+1,
                col-half:col+half+1,
                :
            ]

            X.append(
                patch
            )

            y.append(
                labels[row, col]
            )


    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    print("=" * 60)
    print("Week 2 preprocessing")
    print("=" * 60)

    print(
        "Patch dataset:",
        X.shape
    )

    print(
        "Labels:",
        y.shape
    )

    print(
        "Healthy:",
        np.sum(y == 0)
    )

    print(
        "Stressed:",
        np.sum(y == 1)
    )

    return X, y


if __name__ == "__main__":

    X, y = create_patches()

    X_FILE = os.path.join(
        OUTPUT_DIR,
        "week2_X.npy"
    )

    Y_FILE = os.path.join(
        OUTPUT_DIR,
        "week2_y.npy"
    )

    np.save(
        X_FILE,
        X
    )

    np.save(
        Y_FILE,
        y
    )

    print(
        "X saved:",
        X_FILE
    )

    print(
        "Y saved:",
        Y_FILE
    )