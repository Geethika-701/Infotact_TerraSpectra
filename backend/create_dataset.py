import os
import numpy as np

# ============================================================
# TerraSpectra - Mock Hyperspectral Dataset
# ============================================================

np.random.seed(42)

# Dataset dimensions
HEIGHT = 64
WIDTH = 64
BANDS = 224

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "raw"
)

os.makedirs(RAW_DIR, exist_ok=True)

# ------------------------------------------------------------
# Generate hyperspectral cube
#
# Shape:
# Height × Width × Bands
# 64 × 64 × 224
# ------------------------------------------------------------

cube = np.random.rand(
    HEIGHT,
    WIDTH,
    BANDS
).astype(np.float32)

# ------------------------------------------------------------
# Create labels
#
# 0 = Healthy
# 1 = Chemically Stressed
# ------------------------------------------------------------

labels = np.zeros(
    (HEIGHT, WIDTH),
    dtype=np.uint8
)

# Simulated stressed region
labels[20:44, 25:50] = 1

# ------------------------------------------------------------
# Modify spectral signature of stressed pixels
# ------------------------------------------------------------

stressed_region = labels == 1

# Overall reduction in reflectance
cube[stressed_region] *= 0.75

# Spectral response pattern
spectral_pattern = np.linspace(
    0.8,
    1.2,
    BANDS
).astype(np.float32)

cube[stressed_region] *= spectral_pattern

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

cube_file = os.path.join(
    RAW_DIR,
    "mock_hyperspectral_cube.npy"
)

labels_file = os.path.join(
    RAW_DIR,
    "labels.npy"
)

np.save(cube_file, cube)
np.save(labels_file, labels)

# ------------------------------------------------------------
# Information
# ------------------------------------------------------------

print("=" * 60)
print("TerraSpectra Mock Hyperspectral Dataset")
print("=" * 60)

print("Dataset created successfully!")

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

print("Cube saved to:")
print(cube_file)

print("Labels saved to:")
print(labels_file)

print("=" * 60)