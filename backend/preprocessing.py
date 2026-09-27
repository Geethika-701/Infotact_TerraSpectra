import os
import numpy as np
import rasterio

INPUT_PATH = "dataset/raw/mock_hyperspectral.tif"
OUTPUT_PATH = "dataset/processed/normalized_hyperspectral.npy"


def load_hyperspectral_data():

    with rasterio.open(INPUT_PATH) as src:

        data = src.read()

    # Convert:
    # (bands, height, width)
    # to
    # (height, width, bands)

    cube = np.transpose(
        data,
        (1, 2, 0)
    )

    return cube


def normalize_cube(cube):

    minimum = cube.min()
    maximum = cube.max()

    normalized = (
        cube - minimum
    ) / (
        maximum - minimum
    )

    return normalized.astype(np.float32)


def preprocess():

    cube = load_hyperspectral_data()

    print("Original Shape:", cube.shape)
    print("Original Min:", cube.min())
    print("Original Max:", cube.max())

    normalized_cube = normalize_cube(cube)

    os.makedirs(
        "dataset/processed",
        exist_ok=True
    )

    np.save(
        OUTPUT_PATH,
        normalized_cube
    )

    print("\nNormalization completed!")
    print("Normalized Shape:", normalized_cube.shape)
    print("Normalized Min:", normalized_cube.min())
    print("Normalized Max:", normalized_cube.max())
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess()