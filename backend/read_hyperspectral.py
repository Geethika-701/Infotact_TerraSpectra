import rasterio
import numpy as np

GEOTIFF_PATH = "dataset/raw/mock_hyperspectral.tif"


def read_geotiff():
    with rasterio.open(GEOTIFF_PATH) as src:

        print("GeoTIFF Information")
        print("--------------------")
        print("Width       :", src.width)
        print("Height      :", src.height)
        print("Bands       :", src.count)
        print("CRS         :", src.crs)
        print("Data Type   :", src.dtypes[0])

        # Rasterio returns:
        # (bands, height, width)
        data = src.read()

        # Convert to:
        # (height, width, bands)
        cube = np.transpose(
            data,
            (1, 2, 0)
        )

        profile = src.profile

    return cube, profile


if __name__ == "__main__":

    cube, profile = read_geotiff()

    print("\nLoaded Cube")
    print("-----------")
    print("Shape:", cube.shape)
    print("Minimum:", cube.min())
    print("Maximum:", cube.max())