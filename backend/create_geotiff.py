import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Input and output paths
input_path = "dataset/raw/mock_hyperspectral_cube.npy"
output_path = "dataset/raw/mock_hyperspectral.tif"

# Load hyperspectral cube
cube = np.load(input_path)

height, width, bands = cube.shape

# Create output directory
os.makedirs("dataset/raw", exist_ok=True)

# Geographic information for the mock farm area
transform = from_origin(
    83.30,      # longitude
    17.72,      # latitude
    0.0001,     # pixel width
    0.0001      # pixel height
)

# Create multi-band GeoTIFF
with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=bands,
    dtype="float32",
    crs="EPSG:4326",
    transform=transform
) as dst:

    for band in range(bands):
        dst.write(cube[:, :, band], band + 1)

print("GeoTIFF created successfully!")
print(f"Output file : {output_path}")
print(f"Dimensions  : {height} x {width}")
print(f"Bands       : {bands}")
print("CRS         : EPSG:4326")