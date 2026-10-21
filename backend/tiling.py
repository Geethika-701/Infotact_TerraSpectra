import numpy as np


def generate_tiles(
    cube,
    tile_height=16,
    tile_width=16,
    overlap=4
):
    """
    Split a hyperspectral cube into overlapping tiles.

    Input:
        cube -> [height, width, bands]

    Example:
        [64, 64, 224]

    Output:
        List of overlapping tiles.
    """

    height, width, bands = cube.shape

    if overlap >= tile_height:
        raise ValueError(
            "Overlap must be smaller than tile height."
        )

    if overlap >= tile_width:
        raise ValueError(
            "Overlap must be smaller than tile width."
        )

    step_y = tile_height - overlap
    step_x = tile_width - overlap

    tiles = []

    tile_id = 0

    y = 0

    while y < height:

        x = 0

        y_end = min(
            y + tile_height,
            height
        )

        y_start = max(
            0,
            y_end - tile_height
        )

        while x < width:

            x_end = min(
                x + tile_width,
                width
            )

            x_start = max(
                0,
                x_end - tile_width
            )

            tile = cube[
                y_start:y_end,
                x_start:x_end,
                :
            ]

            tiles.append({
                "tile_id": tile_id,
                "row_start": y_start,
                "row_end": y_end,
                "col_start": x_start,
                "col_end": x_end,
                "height": tile.shape[0],
                "width": tile.shape[1],
                "bands": bands,
                "data": tile
            })

            tile_id += 1

            if x_end >= width:
                break

            x += step_x

        if y_end >= height:
            break

        y += step_y

    return tiles


def stitch_predictions(
    tile_predictions,
    height,
    width
):
    """
    Stitch overlapping tile predictions.

    When multiple tiles cover the same pixel,
    their predictions are averaged.
    """

    prediction_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for tile in tile_predictions:

        y1 = tile["row_start"]
        y2 = tile["row_end"]

        x1 = tile["col_start"]
        x2 = tile["col_end"]

        prediction = np.asarray(
            tile["prediction"],
            dtype=np.float32
        )

        prediction_sum[
            y1:y2,
            x1:x2
        ] += prediction

        prediction_count[
            y1:y2,
            x1:x2
        ] += 1

    prediction_count[
        prediction_count == 0
    ] = 1

    final_prediction = (
        prediction_sum /
        prediction_count
    )

    return final_prediction


def tile_info(
    cube,
    tile_height=16,
    tile_width=16,
    overlap=4
):
    """
    Return metadata about the tiling configuration.
    """

    height, width, bands = cube.shape

    tiles = generate_tiles(
        cube,
        tile_height,
        tile_width,
        overlap
    )

    return {
        "image_height": height,
        "image_width": width,
        "spectral_bands": bands,
        "tile_height": tile_height,
        "tile_width": tile_width,
        "overlap": overlap,
        "step_height": tile_height - overlap,
        "step_width": tile_width - overlap,
        "number_of_tiles": len(tiles)
    }