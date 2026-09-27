import os
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = "dataset/processed/pca_hyperspectral.npy"
OUTPUT_DIR = "results/pca"


def visualize_pca():

    # Load PCA-reduced cube
    pca_cube = np.load(INPUT_PATH)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("PCA Cube Shape:", pca_cube.shape)

    # Visualize first 3 components
    for component in range(3):

        image = pca_cube[:, :, component]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"pca_component_{component + 1}.png"
        )

        plt.figure(
            figsize=(7, 6)
        )

        plt.imshow(
            image,
            cmap="viridis"
        )

        plt.colorbar(
            label="Component Value"
        )

        plt.title(
            f"PCA Component {component + 1}"
        )

        plt.axis("off")

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", output_path)


if __name__ == "__main__":
    visualize_pca()