import os
import numpy as np

from sklearn.decomposition import PCA

from backend.rasterio_ingestion import (
    read_hyperspectral_geotiff
)


def run_pca(
    cube,
    components=3
):
    """
    Performs PCA on the spectral dimension.

    Input:
        height x width x spectral_bands

    Output:
        height x width x components
    """

    if cube.ndim != 3:
        raise ValueError(
            "Hyperspectral cube must have "
            "three dimensions."
        )

    height = cube.shape[0]
    width = cube.shape[1]
    bands = cube.shape[2]

    pixels = cube.reshape(
        height * width,
        bands
    )

    pca = PCA(
        n_components=components
    )

    reduced_pixels = pca.fit_transform(
        pixels
    )

    reduced_cube = reduced_pixels.reshape(
        height,
        width,
        components
    )

    explained_variance = (
        pca.explained_variance_ratio_
        * 100
    )

    return (
        reduced_cube,
        pca,
        explained_variance
    )


def pca_summary(
    components=3
):
    """
    Reads the GeoTIFF through Rasterio,
    then performs real PCA on all spectral bands.
    """

    cube, raster_metadata = (
        read_hyperspectral_geotiff()
    )

    original_shape = cube.shape

    reduced_cube, pca, explained_variance = (
        run_pca(
            cube,
            components=components
        )
    )

    original_memory = cube.nbytes
    reduced_memory = reduced_cube.nbytes

    total_variance = float(
        explained_variance.sum()
    )

    result = {
        "status": "success",
        "method": "Principal Component Analysis",
        "input": {
            "height": original_shape[0],
            "width": original_shape[1],
            "spectral_bands": original_shape[2],
            "shape": list(
                original_shape
            ),
            "dtype": str(
                cube.dtype
            ),
            "memory_mb": round(
                original_memory /
                (1024 * 1024),
                4
            )
        },
        "output": {
            "height": reduced_cube.shape[0],
            "width": reduced_cube.shape[1],
            "components": reduced_cube.shape[2],
            "shape": list(
                reduced_cube.shape
            ),
            "dtype": str(
                reduced_cube.dtype
            ),
            "memory_mb": round(
                reduced_memory /
                (1024 * 1024),
                4
            )
        },
        "explained_variance_percent": [
            round(
                float(value),
                4
            )
            for value in explained_variance
        ],
        "total_explained_variance_percent": round(
            total_variance,
            4
        ),
        "variance_components": {
            "PC1": round(
                float(explained_variance[0]),
                4
            ),
            "PC2": round(
                float(explained_variance[1]),
                4
            ),
            "PC3": round(
                float(explained_variance[2]),
                4
            )
        },
        "compression": {
            "original_bands": original_shape[2],
            "reduced_components": components,
            "reduction_ratio": round(
                original_shape[2] /
                components,
                2
            )
        },
        "geospatial": {
            "crs": raster_metadata["crs"],
            "bounds": raster_metadata["bounds"]
        }
    }

    return result