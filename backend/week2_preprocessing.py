import os
import numpy as np
import rasterio
from sklearn.decomposition import PCA


CUBE_PATHS = [
    "dataset/processed/normalized_hyperspectral.npy",
    "dataset/processed/mock_hyperspectral.tif",
    "dataset/raw/mock_hyperspectral.tif",
]

PCA_PATH = (
    "dataset/processed/pca_hyperspectral.npy"
)

LABEL_PATH = (
    "results/training/labels.npy"
)

PATCH_SIZE = 5


def load_hyperspectral_cube():

    for path in CUBE_PATHS:

        if os.path.exists(path):

            print(
                f"Loading hyperspectral data: {path}"
            )

            if path.endswith(".npy"):

                cube = np.load(path)

            else:

                with rasterio.open(path) as src:

                    cube = src.read()

                # Rasterio:
                # bands, height, width
                #
                # Convert to:
                # height, width, bands

                cube = np.transpose(
                    cube,
                    (1, 2, 0)
                )

            cube = cube.astype(
                np.float32
            )

            return cube

    raise FileNotFoundError(
        "No hyperspectral cube found."
    )


def load_pca_cube(
    cube
):

    if os.path.exists(PCA_PATH):

        print(
            f"Loading PCA cube: {PCA_PATH}"
        )

        return np.load(
            PCA_PATH
        ).astype(
            np.float32
        )

    print(
        "PCA file not found."
    )

    print(
        "Calculating PCA from hyperspectral cube..."
    )

    height, width, bands = cube.shape

    pixels = cube.reshape(
        height * width,
        bands
    )

    pca = PCA(
        n_components=10,
        random_state=42
    )

    reduced = pca.fit_transform(
        pixels
    )

    return reduced.reshape(
        height,
        width,
        10
    ).astype(
        np.float32
    )


def load_labels():

    if not os.path.exists(
        LABEL_PATH
    ):

        raise FileNotFoundError(
            "Week 2 labels not found. "
            "Run create_week2_labels.py first."
        )

    return np.load(
        LABEL_PATH
    ).astype(
        np.int64
    )


def extract_patches(
    cube,
    pca_cube,
    labels,
    patch_size=5
):

    height, width, bands = cube.shape

    radius = patch_size // 2

    hyperspectral_patches = []
    pca_patches = []
    patch_labels = []
    coordinates = []

    for row in range(
        radius,
        height - radius
    ):

        for col in range(
            radius,
            width - radius
        ):

            hs_patch = cube[
                row - radius:
                row + radius + 1,

                col - radius:
                col + radius + 1,

                :
            ]

            pca_patch = pca_cube[
                row - radius:
                row + radius + 1,

                col - radius:
                col + radius + 1,

                :
            ]

            label = labels[
                row,
                col
            ]

            hyperspectral_patches.append(
                hs_patch
            )

            pca_patches.append(
                pca_patch
            )

            patch_labels.append(
                label
            )

            coordinates.append(
                (
                    row,
                    col
                )
            )

    hyperspectral_patches = np.asarray(
        hyperspectral_patches,
        dtype=np.float32
    )

    pca_patches = np.asarray(
        pca_patches,
        dtype=np.float32
    )

    patch_labels = np.asarray(
        patch_labels,
        dtype=np.int64
    )

    # Convert:
    #
    # N, H, W, Bands
    #
    # to:
    #
    # N, 1, H, W, Bands

    hyperspectral_patches = (
        hyperspectral_patches
        [:, np.newaxis, :, :, :]
    )

    # Convert PCA:
    #
    # N, H, W, Components
    #
    # to:
    #
    # N, Components, H, W

    pca_patches = np.transpose(
        pca_patches,
        (0, 3, 1, 2)
    )

    return (
        hyperspectral_patches,
        pca_patches,
        patch_labels,
        coordinates
    )


def prepare_week2_data():

    cube = load_hyperspectral_cube()

    pca_cube = load_pca_cube(
        cube
    )

    labels = load_labels()

    print("\nData Shapes")
    print("-----------")
    print("Hyperspectral:", cube.shape)
    print("PCA:", pca_cube.shape)
    print("Labels:", labels.shape)

    (
        hyperspectral,
        pca,
        patch_labels,
        coordinates
    ) = extract_patches(
        cube,
        pca_cube,
        labels,
        PATCH_SIZE
    )

    print("\nPatch Data")
    print("----------")
    print(
        "Hyperspectral patches:",
        hyperspectral.shape
    )

    print(
        "PCA patches:",
        pca.shape
    )

    print(
        "Labels:",
        patch_labels.shape
    )

    return (
        hyperspectral,
        pca,
        patch_labels,
        coordinates
    )