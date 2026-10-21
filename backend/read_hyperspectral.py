import os
import rasterio
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DEFAULT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "mock_hyperspectral.tif"
)


def read_hyperspectral(
    file_path=DEFAULT_FILE
):

    with rasterio.open(file_path) as src:

        # Rasterio returns:
        # Bands × Height × Width

        raster_cube = src.read()

        profile = src.profile

        transform = src.transform

        crs = src.crs

        bounds = src.bounds

        width = src.width

        height = src.height

        count = src.count


    # Convert:
    #
    # Bands × Height × Width
    #
    # to:
    #
    # Height × Width × Bands

    cube = np.transpose(
        raster_cube,
        (1, 2, 0)
    )

    print("=" * 60)
    print("Hyperspectral Data")
    print("=" * 60)

    print("Cube shape:", cube.shape)
    print("Height:", height)
    print("Width:", width)
    print("Bands:", count)
    print("CRS:", crs)
    print("Bounds:", bounds)

    return (
        cube,
        profile,
        transform,
        crs,
        bounds
    )


if __name__ == "__main__":

    read_hyperspectral()