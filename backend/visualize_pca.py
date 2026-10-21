import os
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "pca_cube.npy"
)


pca_cube = np.load(FILE)

print(
    "PCA cube shape:",
    pca_cube.shape
)


# PC1

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    pca_cube[:, :, 0],
    cmap="viridis"
)

plt.colorbar(
    label="PC1"
)

plt.title(
    "TerraSpectra - Principal Component 1"
)

plt.xlabel(
    "Pixel Column"
)

plt.ylabel(
    "Pixel Row"
)

plt.tight_layout()

plt.show()


# PC2

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    pca_cube[:, :, 1],
    cmap="viridis"
)

plt.colorbar(
    label="PC2"
)

plt.title(
    "TerraSpectra - Principal Component 2"
)

plt.tight_layout()

plt.show()


# PC3

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    pca_cube[:, :, 2],
    cmap="viridis"
)

plt.colorbar(
    label="PC3"
)

plt.title(
    "TerraSpectra - Principal Component 3"
)

plt.tight_layout()

plt.show()