import os
import numpy as np
from sklearn.decomposition import PCA

from read_hyperspectral import read_hyperspectral
from preprocessing import normalize_cube


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def perform_pca(
    cube,
    components=3
):

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cube = normalize_cube(cube)

    height, width, bands = cube.shape

    # --------------------------------------------------------
    # Convert:
    #
    # Height × Width × Bands
    #
    # to:
    #
    # Pixels × Bands
    # --------------------------------------------------------

    pixels = cube.reshape(
        height * width,
        bands
    )

    print(
        "Pixel matrix:",
        pixels.shape
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=components,
        random_state=42
    )

    transformed = pca.fit_transform(
        pixels
    )

    print(
        "PCA transformed:",
        transformed.shape
    )

    print(
        "Explained variance:",
        pca.explained_variance_ratio_
    )

    print(
        "Total explained variance:",
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    # --------------------------------------------------------
    # Convert back:
    #
    # Pixels × Components
    #
    # to:
    #
    # Height × Width × Components
    # --------------------------------------------------------

    pca_cube = transformed.reshape(
        height,
        width,
        components
    )

    return pca_cube, pca


if __name__ == "__main__":

    cube, profile, transform, crs, bounds = \
        read_hyperspectral(
            INPUT_FILE
        )

    pca_cube, pca = perform_pca(
        cube,
        components=3
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "pca_cube.npy"
    )

    np.save(
        output_file,
        pca_cube
    )

    print("=" * 60)
    print("PCA completed")
    print("=" * 60)

    print(
        "Original shape:",
        cube.shape
    )

    print(
        "PCA shape:",
        pca_cube.shape
    )

    print(
        "Saved:",
        output_file
    )