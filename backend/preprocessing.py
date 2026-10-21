import numpy as np


def normalize_cube(cube):

    cube = cube.astype(
        np.float32
    )

    minimum = np.min(cube)

    maximum = np.max(cube)

    normalized = (
        cube - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    return normalized


def standardize_cube(cube):

    mean = np.mean(cube)

    std = np.std(cube)

    standardized = (
        cube - mean
    ) / (
        std + 1e-8
    )

    return standardized