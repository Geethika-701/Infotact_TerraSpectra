import os
import numpy as np

np.random.seed(42)

HEIGHT = 64
WIDTH = 64
BANDS = 224

os.makedirs("dataset/raw", exist_ok=True)

cube = np.random.rand(
    HEIGHT,
    WIDTH,
    BANDS
).astype(np.float32)

labels = np.zeros(
    (HEIGHT, WIDTH),
    dtype=np.uint8
)

# Chemically stressed region
labels[20:44, 25:50] = 1

stressed_region = labels == 1

cube[stressed_region] *= 0.75

spectral_pattern = np.linspace(
    0.8,
    1.2,
    BANDS
).astype(np.float32)

cube[stressed_region] *= spectral_pattern

np.save(
    "dataset/raw/mock_hyperspectral_cube.npy",
    cube
)

np.save(
    "dataset/raw/labels.npy",
    labels
)

print("Mock hyperspectral dataset created successfully!")

print(f"Spatial dimensions : {HEIGHT} x {WIDTH}")
print(f"Spectral bands     : {BANDS}")
print(f"Cube shape         : {cube.shape}")
print(f"Labels shape       : {labels.shape}")

print(
    "Healthy pixels     :",
    np.sum(labels == 0)
)

print(
    "Stressed pixels    :",
    np.sum(labels == 1)
)