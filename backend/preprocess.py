import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


RAW_CUBE = "dataset/raw/mock_hyperspectral_cube.npy"

PROCESSED_DIR = "dataset/processed"

PCA_CUBE = os.path.join(
    PROCESSED_DIR,
    "pca_cube.npy"
)

PCA_MODEL = os.path.join(
    PROCESSED_DIR,
    "pca_model.pkl"
)

SCALER_MODEL = os.path.join(
    PROCESSED_DIR,
    "scaler.pkl"
)


def main():

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    print("\nLoading hyperspectral cube...")

    cube = np.load(
        RAW_CUBE
    )

    print(
        "Original cube shape:",
        cube.shape
    )

    height, width, bands = cube.shape

    # Convert spatial pixels into rows
    pixels = cube.reshape(
        height * width,
        bands
    )

    print(
        "Pixel matrix:",
        pixels.shape
    )

    # -----------------------------
    # NORMALIZATION
    # -----------------------------

    print("\nNormalizing spectral bands...")

    scaler = StandardScaler()

    normalized = scaler.fit_transform(
        pixels
    )

    print(
        "Normalized shape:",
        normalized.shape
    )

    # -----------------------------
    # PCA
    # -----------------------------

    print("\nRunning PCA...")

    pca = PCA(
        n_components=3,
        random_state=42
    )

    reduced = pca.fit_transform(
        normalized
    )

    print(
        "PCA output:",
        reduced.shape
    )

    print("\nExplained variance:")

    for i, value in enumerate(
        pca.explained_variance_ratio_
    ):
        print(
            f"PC{i + 1}: {value * 100:.2f}%"
        )

    # -----------------------------
    # Reshape PCA cube
    # -----------------------------

    pca_cube = reduced.reshape(
        height,
        width,
        3
    ).astype(
        np.float32
    )

    print(
        "\nPCA cube shape:",
        pca_cube.shape
    )

    # -----------------------------
    # SAVE
    # -----------------------------

    np.save(
        PCA_CUBE,
        pca_cube
    )

    joblib.dump(
        pca,
        PCA_MODEL
    )

    joblib.dump(
        scaler,
        SCALER_MODEL
    )

    print("\nWeek 1 preprocessing completed.")

    print(
        "Saved:",
        PCA_CUBE
    )

    print(
        "Saved:",
        PCA_MODEL
    )

    print(
        "Saved:",
        SCALER_MODEL
    )


if __name__ == "__main__":
    main()