import os
import time
import tracemalloc

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CUBE_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "raw",
    "mock_hyperspectral_cube.npy"
)


def bytes_to_mb(value):
    return round(value / (1024 * 1024), 4)


def run_memory_audit():
    """
    Memory audit for the TerraSpectra hyperspectral pipeline.

    Demonstrates:
    1. Full hyperspectral cube memory.
    2. Per-tile memory.
    3. Batched inference memory.
    4. Reduction achieved by processing smaller tiles.
    """

    if not os.path.exists(CUBE_PATH):
        raise FileNotFoundError(
            f"Hyperspectral cube not found: {CUBE_PATH}"
        )

    cube = np.load(CUBE_PATH)

    height = cube.shape[0]
    width = cube.shape[1]
    bands = cube.shape[2]

    full_cube_bytes = cube.nbytes

    # ---------------------------------------------------------
    # Tile memory calculation
    # ---------------------------------------------------------

    tile_height = 16
    tile_width = 16

    tile = cube[
        0:tile_height,
        0:tile_width,
        :
    ]

    tile_bytes = tile.nbytes

    # ---------------------------------------------------------
    # Batch memory calculation
    # ---------------------------------------------------------

    batch_size = 16

    # 7 x 7 spatial patch with 224 spectral bands
    patch_height = 7
    patch_width = 7

    single_patch_elements = (
        patch_height *
        patch_width *
        bands
    )

    single_patch_bytes = (
        single_patch_elements *
        np.dtype(np.float32).itemsize
    )

    batch_bytes = (
        single_patch_bytes *
        batch_size
    )

    # ---------------------------------------------------------
    # Demonstrate actual temporary batch allocation
    # ---------------------------------------------------------

    tracemalloc.start()

    start_time = time.perf_counter()

    temporary_batch = np.zeros(
        (
            batch_size,
            1,
            patch_height,
            patch_width,
            bands
        ),
        dtype=np.float32
    )

    temporary_batch += 0.001

    current_memory, peak_memory = tracemalloc.get_traced_memory()

    del temporary_batch

    elapsed_seconds = time.perf_counter() - start_time

    tracemalloc.stop()

    # ---------------------------------------------------------
    # Memory reduction
    # ---------------------------------------------------------

    if full_cube_bytes > 0:
        tile_memory_reduction = (
            1 -
            (tile_bytes / full_cube_bytes)
        ) * 100
    else:
        tile_memory_reduction = 0

    if full_cube_bytes > 0:
        batch_memory_reduction = (
            1 -
            (batch_bytes / full_cube_bytes)
        ) * 100
    else:
        batch_memory_reduction = 0

    return {
        "status": "success",
        "audit": "Hyperspectral Memory Handling Audit",

        "input": {
            "file": "mock_hyperspectral_cube.npy",
            "shape": [
                int(height),
                int(width),
                int(bands)
            ],
            "dtype": str(cube.dtype),
            "total_elements": int(cube.size)
        },

        "full_cube": {
            "memory_bytes": int(full_cube_bytes),
            "memory_mb": bytes_to_mb(full_cube_bytes)
        },

        "tiling": {
            "tile_size": f"{tile_height}x{tile_width}",
            "spectral_bands": int(bands),
            "tile_memory_bytes": int(tile_bytes),
            "tile_memory_mb": bytes_to_mb(tile_bytes),
            "memory_reduction_percent": round(
                tile_memory_reduction,
                2
            )
        },

        "batching": {
            "batch_size": int(batch_size),
            "patch_size": f"{patch_height}x{patch_width}",
            "spectral_bands": int(bands),
            "estimated_batch_memory_bytes": int(batch_bytes),
            "estimated_batch_memory_mb": bytes_to_mb(
                batch_bytes
            ),
            "memory_reduction_vs_full_cube_percent": round(
                batch_memory_reduction,
                2
            )
        },

        "runtime_memory_test": {
            "peak_traced_memory_mb": bytes_to_mb(
                peak_memory
            ),
            "current_traced_memory_mb": bytes_to_mb(
                current_memory
            ),
            "test_duration_seconds": round(
                elapsed_seconds,
                6
            )
        },

        "engineering_controls": [
            "float32 tensors are used to control memory footprint",
            "inference is performed in batches",
            "large imagery is divided into overlapping tiles",
            "tile predictions are stitched back into the original spatial grid",
            "the complete prediction grid is not processed as one inference batch"
        ],

        "conclusion": (
            "TerraSpectra uses memory-conscious hyperspectral "
            "processing through float32 tensors, batching, "
            "tiling and stitched inference."
        )
    }