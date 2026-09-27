import os
import numpy as np
from sklearn.decomposition import PCA

INPUT_PATH = "dataset/processed/normalized_hyperspectral.npy"
OUTPUT_PATH = "dataset/processed/pca_hyperspectral.npy"

N_COMPONENTS = 10


def perform_pca():

    # Load normalized hyperspectral cube
    cube = np.load(INPUT_PATH)

    height, width, bands = cube.shape

    print("Input Cube")
    print("----------")
    print("Height:", height)
    print("Width :", width)
    print("Bands :", bands)

    # Convert pixels into rows
    # Shape:
    # (height * width, bands)

    pixels = cube.reshape(
        height * width,
        bands
    )

    print("\nPCA Input Shape:", pixels.shape)

    # PCA
    pca = PCA(
        n_components=N_COMPONENTS,
        random_state=42
    )

    reduced_pixels = pca.fit_transform(
        pixels
    )

    # Restore spatial dimensions
    reduced_cube = reduced_pixels.reshape(
        height,
        width,
        N_COMPONENTS
    )

    os.makedirs(
        "dataset/processed",
        exist_ok=True
    )

    np.save(
        OUTPUT_PATH,
        reduced_cube.astype(np.float32)
    )

    explained_variance = (
        pca.explained_variance_ratio_
    )

    print("\nPCA completed!")
    print("----------------")
    print("Original Bands:", bands)
    print("Reduced Components:", N_COMPONENTS)
    print("Output Shape:", reduced_cube.shape)

    print("\nExplained Variance:")
    for index, value in enumerate(
        explained_variance,
        start=1
    ):
        print(
            f"Component {index}: "
            f"{value:.4f}"
        )

    print(
        "\nTotal Explained Variance:",
        f"{explained_variance.sum():.4f}"
    )

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    perform_pca()