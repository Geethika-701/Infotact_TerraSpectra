import os
import numpy as np
import rasterio
from rasterio.transform import from_origin


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "mock_hyperspectral_cube.npy"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "mock_hyperspectral.tif"
)


# ------------------------------------------------------------
# Load cube
# ------------------------------------------------------------

cube = np.load(INPUT_FILE)

height, width, bands = cube.shape

print("Input cube shape:", cube.shape)

# ------------------------------------------------------------
# Rasterio expects:
#
# Band × Height × Width
#
# Convert from:
#
# Height × Width × Band
# ------------------------------------------------------------

cube_raster = np.transpose(
    cube,
    (2, 0, 1)
)

# ------------------------------------------------------------
# Geographic information
# ------------------------------------------------------------

transform = from_origin(
    80.0000,
    16.0000,
    0.0001,
    0.0001
)

# ------------------------------------------------------------
# Create GeoTIFF
# ------------------------------------------------------------

with rasterio.open(

    OUTPUT_FILE,

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

        dst.write(
            cube_raster[band],
            band + 1
        )


print("=" * 60)
print("GeoTIFF created successfully")
print("=" * 60)

print("File:", OUTPUT_FILE)
print("Height:", height)
print("Width:", width)
print("Bands:", bands)
print("CRS: EPSG:4326")