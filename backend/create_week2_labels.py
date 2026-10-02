import os
import numpy as np


INPUT_PATH = "dataset/processed/normalized_hyperspectral.npy"
OUTPUT_PATH = "results/training/labels.npy"


def create_labels():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    cube = np.load(INPUT_PATH)

    height, width, bands = cube.shape

    print("Hyperspectral Cube")
    print("------------------")
    print("Height:", height)
    print("Width :", width)
    print("Bands :", bands)

    # --------------------------------------------------
    # Create a synthetic spectral stress score.
    #
    # This is for the MOCK dataset only.
    # It is not a real disease annotation.
    # --------------------------------------------------

    low_band_mean = cube[:, :, :bands // 3].mean(
        axis=2
    )

    high_band_mean = cube[
        :,
        :,
        2 * bands // 3:
    ].mean(axis=2)

    spectral_score = (
        high_band_mean - low_band_mean
    )

    # Normalize score
    minimum = spectral_score.min()
    maximum = spectral_score.max()

    normalized_score = (
        spectral_score - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    # Median threshold gives approximately
    # balanced healthy/stressed classes.
    threshold = np.median(
        normalized_score
    )

    labels = (
        normalized_score >= threshold
    ).astype(np.int64)

    os.makedirs(
        "results/training",
        exist_ok=True
    )

    np.save(
        OUTPUT_PATH,
        labels
    )

    healthy = int(
        np.sum(labels == 0)
    )

    stressed = int(
        np.sum(labels == 1)
    )

    print("\nWeek 2 Labels Created")
    print("---------------------")
    print("Threshold:", float(threshold))
    print("Healthy:", healthy)
    print("Chemically Stressed:", stressed)
    print("Total:", height * width)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    create_labels()