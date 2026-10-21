from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np

from predict import predict_grid

from analytics import calculate_analytics

from history import (
    load_history,
    save_snapshot,
    create_snapshot
)

from tiling import (
    tile_info,
    generate_tiles,
    stitch_predictions
)


CUBE_PATH = (
    "dataset/raw/mock_hyperspectral_cube.npy"
)


app = FastAPI(
    title="TerraSpectra API",
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():

    return {

        "project": "TerraSpectra",

        "status": "online",

        "version": "2.0.0",

        "week_1": [
            "Hyperspectral Processing",
            "PCA Dimensionality Reduction"
        ],

        "week_2": [
            "3D-CNN Classification",
            "Risk Mapping"
        ],

        "week_3": [
            "Vision Transformer",
            "Self Attention",
            "Analytics"
        ],

        "week_4": [
            "FastAPI",
            "Image Tiling",
            "Prediction Stitching",
            "Historical Timeline"
        ]
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "api": "online"
    }


# --------------------------------------------------
# EXISTING WEEK 2 PREDICTIONS
# --------------------------------------------------

@app.get("/predictions")
def predictions():

    results = predict_grid()

    return {
        "count": len(results),
        "predictions": results
    }


# --------------------------------------------------
# WEEK 3 ANALYTICS
# --------------------------------------------------

@app.get("/analytics")
def analytics():

    results = predict_grid()

    data = calculate_analytics(
        results
    )

    return data


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

@app.get("/models")
def models():

    return {

        "primary_model":
            "3D-CNN",

        "advanced_model":
            "Vision Transformer",

        "attention":
            "Multi-Head Self-Attention",

        "spectral_bands":
            224,

        "classification": [
            "Healthy",
            "Chemically Stressed"
        ],

        "fusion_ready":
            True
    }


# --------------------------------------------------
# SPECTRAL / CHEMICAL ANOMALIES
# --------------------------------------------------

@app.get("/anomalies")
def anomalies():

    results = predict_grid()

    data = calculate_analytics(
        results
    )

    return {

        "chemical_stress_zones":
            data[
                "chemical_anomalies"
            ][
                "chemical_stress_zones"
            ],

        "spectral_anomaly_zones":
            data[
                "chemical_anomalies"
            ][
                "spectral_anomaly_zones"
            ],

        "stress_class":
            data[
                "chemical_anomalies"
            ][
                "stress_class"
            ],

        "note": (
            "Current dataset supports "
            "chemical stress classification. "
            "Specific chemical identities "
            "require chemical-specific "
            "training labels."
        )
    }


# --------------------------------------------------
# WEEK 4 TILING INFORMATION
# --------------------------------------------------

@app.get("/tiling")
def tiling():

    cube = np.load(
        CUBE_PATH
    )

    info = tile_info(
        cube,
        tile_height=16,
        tile_width=16,
        overlap=4
    )

    return info


# --------------------------------------------------
# WEEK 4 TILE PROCESSING DEMO
# --------------------------------------------------

@app.get("/tiling/process")
def process_tiles():

    cube = np.load(
        CUBE_PATH
    )

    tiles = generate_tiles(
        cube,
        tile_height=16,
        tile_width=16,
        overlap=4
    )

    processed_tiles = []

    for tile in tiles:

        tile_data = tile["data"]

        # Simple spectral statistic for each tile.
        # This represents the type of processing
        # that can later be replaced by model inference.

        spectral_mean = float(
            tile_data.mean()
        )

        processed_tiles.append({

            "tile_id":
                tile["tile_id"],

            "row_start":
                tile["row_start"],

            "row_end":
                tile["row_end"],

            "col_start":
                tile["col_start"],

            "col_end":
                tile["col_end"],

            "spectral_mean":
                round(
                    spectral_mean,
                    6
                )
        })

    return {

        "image_height":
            cube.shape[0],

        "image_width":
            cube.shape[1],

        "bands":
            cube.shape[2],

        "tiles_processed":
            len(processed_tiles),

        "tile_size":
            "16x16",

        "overlap":
            4,

        "tiles":
            processed_tiles
    }


# --------------------------------------------------
# WEEK 4 STITCHING DEMO
# --------------------------------------------------

@app.get("/tiling/stitch")
def stitch_demo():

    cube = np.load(
        CUBE_PATH
    )

    height = cube.shape[0]
    width = cube.shape[1]

    tiles = generate_tiles(
        cube,
        tile_height=16,
        tile_width=16,
        overlap=4
    )

    tile_predictions = []

    for tile in tiles:

        tile_data = tile["data"]

        # Demo prediction based on spectral mean.
        # The real model can replace this later.

        spectral_mean = float(
            tile_data.mean()
        )

        prediction = np.full(
            (
                tile["height"],
                tile["width"]
            ),
            spectral_mean,
            dtype=np.float32
        )

        tile_predictions.append({

            "row_start":
                tile["row_start"],

            "row_end":
                tile["row_end"],

            "col_start":
                tile["col_start"],

            "col_end":
                tile["col_end"],

            "prediction":
                prediction
        })

    stitched = stitch_predictions(
        tile_predictions,
        height,
        width
    )

    return {

        "image_height":
            height,

        "image_width":
            width,

        "tiles":
            len(tiles),

        "stitched_shape":
            list(
                stitched.shape
            ),

        "min_prediction":
            round(
                float(
                    stitched.min()
                ),
                6
            ),

        "max_prediction":
            round(
                float(
                    stitched.max()
                ),
                6
            ),

        "mean_prediction":
            round(
                float(
                    stitched.mean()
                ),
                6
            )
    }


# --------------------------------------------------
# WEEK 4 HISTORY
# --------------------------------------------------

@app.get("/history")
def history():

    records = load_history()

    return {

        "count":
            len(records),

        "history":
            records
    }


# --------------------------------------------------
# SAVE CURRENT SNAPSHOT
# --------------------------------------------------

@app.post("/history/snapshot")
def history_snapshot():

    results = predict_grid()

    analytics_data = calculate_analytics(
        results
    )

    snapshot = create_snapshot(
        analytics_data
    )

    saved = save_snapshot(
        snapshot
    )

    return {

        "status":
            "saved",

        "snapshot":
            saved
    }