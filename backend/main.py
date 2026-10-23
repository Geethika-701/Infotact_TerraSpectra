from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np

from backend.predict import predict_grid

from backend.fusion_predict import (
    predict_fused_grid,
    predict_single_patch,
    normalize_cube,
    create_patch
)

from backend.analytics import (
    calculate_analytics
)

from backend.history import (
    load_history,
    save_snapshot,
    create_snapshot
)

from backend.tiling import (
    generate_tiles,
    stitch_predictions,
    tile_info
)


CUBE_PATH = (
    "dataset/raw/mock_hyperspectral_cube.npy"
)


app = FastAPI(
    title="TerraSpectra API",
    version="4.0.0"
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


# ==================================================
# ROOT
# ==================================================

@app.get("/")
def root():

    return {

        "project":
            "TerraSpectra",

        "status":
            "online",

        "version":
            "4.0.0",

        "weeks": {

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
                "Multi-Head Self-Attention",
                "Model Fusion",
                "Analytics"
            ],

            "week_4": [
                "FastAPI",
                "Satellite Tiling",
                "Overlapping Tiles",
                "Prediction Stitching",
                "Historical Timeline"
            ]
        }
    }


# ==================================================
# HEALTH
# ==================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "api":
            "online",

        "models":
            "3D-CNN + Vision Transformer"
    }


# ==================================================
# EXISTING WEEK 2 PREDICTIONS
# ==================================================

@app.get("/predictions")
def predictions():

    results = predict_grid()

    return {

        "count":
            len(results),

        "predictions":
            results,

        "model":
            "3D-CNN"
    }


# ==================================================
# WEEK 3 FUSED PREDICTIONS
# ==================================================

@app.get("/predictions/fused")
def fused_predictions():

    results = predict_fused_grid(
        stride=2,
        batch_size=16
    )

    return {

        "count":
            len(results),

        "predictions":
            results,

        "model":
            "3D-CNN + Vision Transformer",

        "fusion": {

            "cnn_weight":
                0.60,

            "vit_weight":
                0.40
        }
    }


# ==================================================
# WEEK 3 ANALYTICS
# ==================================================

@app.get("/analytics")
def analytics():

    results = predict_fused_grid(
        stride=2,
        batch_size=16
    )

    data = calculate_analytics(
        results
    )

    return data


# ==================================================
# MODEL INFORMATION
# ==================================================

@app.get("/models")
def models():

    return {

        "primary_model":
            "3D-CNN",

        "advanced_model":
            "Vision Transformer",

        "attention":
            "Multi-Head Self-Attention",

        "fusion":
            "Weighted Probability Fusion",

        "cnn_weight":
            0.60,

        "vit_weight":
            0.40,

        "spectral_bands":
            224,

        "patch_size":
            "7x7",

        "classification": [
            "Healthy",
            "Chemically Stressed"
        ],

        "status":
            "implemented"
    }


# ==================================================
# WEEK 3 MODEL COMPARISON
# ==================================================

@app.get("/models/comparison")
def model_comparison():

    results = predict_fused_grid(
        stride=2,
        batch_size=16
    )

    total = len(results)

    cnn_high = 0
    vit_high = 0
    fused_high = 0

    for item in results:

        if item["cnn_risk"] >= 0.7:
            cnn_high += 1

        if item["vit_risk"] >= 0.7:
            vit_high += 1

        if item["risk"] >= 0.7:
            fused_high += 1

    return {

        "total_zones":
            total,

        "cnn": {

            "high_risk_zones":
                cnn_high,

            "high_risk_percentage":
                round(
                    cnn_high /
                    total *
                    100,
                    2
                )
                if total else 0
        },

        "vit": {

            "high_risk_zones":
                vit_high,

            "high_risk_percentage":
                round(
                    vit_high /
                    total *
                    100,
                    2
                )
                if total else 0
        },

        "fusion": {

            "high_risk_zones":
                fused_high,

            "high_risk_percentage":
                round(
                    fused_high /
                    total *
                    100,
                    2
                )
                if total else 0
        },

        "note": (
            "This endpoint compares prediction "
            "behavior. It does not represent "
            "validation accuracy because the "
            "current dataset is synthetic."
        )
    }


# ==================================================
# ANOMALIES
# ==================================================

@app.get("/anomalies")
def anomalies():

    results = predict_fused_grid(
        stride=2,
        batch_size=16
    )

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
            "The current dataset identifies "
            "chemical stress/anomalous zones. "
            "Specific chemical identities require "
            "chemical-specific spectral labels."
        )
    }


# ==================================================
# WEEK 4 TILING INFORMATION
# ==================================================

@app.get("/tiling")
def tiling():

    cube = np.load(
        CUBE_PATH
    )

    return tile_info(
        cube,
        tile_height=16,
        tile_width=16,
        overlap=4
    )


# ==================================================
# WEEK 4 REAL TILED MODEL INFERENCE
# ==================================================

@app.get("/tiling/inference")
def tiled_inference():

    cube = np.load(
        CUBE_PATH
    )

    normalized_cube = normalize_cube(
        cube
    )

    height, width, bands = (
        normalized_cube.shape
    )

    tiles = generate_tiles(
        normalized_cube,
        tile_height=16,
        tile_width=16,
        overlap=4
    )

    stitched_predictions = []

    from backend.fusion_predict import (
        load_models,
        create_patch,
        predict_batch
    )

    cnn, vit, device = load_models()

    for tile in tiles:

        y1 = tile["row_start"]
        y2 = tile["row_end"]

        x1 = tile["col_start"]
        x2 = tile["col_end"]

        patches = []

        coordinates = []

        for row in range(
            y1,
            y2
        ):

            for col in range(
                x1,
                x2
            ):

                patch = create_patch(
                    normalized_cube,
                    row,
                    col,
                    7
                )

                patches.append(
                    patch
                )

                coordinates.append(
                    (row, col)
                )

        _, _, fused = predict_batch(
            cnn,
            vit,
            device,
            patches
        )

        prediction = np.asarray(
            fused,
            dtype=np.float32
        ).reshape(
            y2 - y1,
            x2 - x1
        )

        stitched_predictions.append({

            "tile_id":
                tile["tile_id"],

            "row_start":
                y1,

            "row_end":
                y2,

            "col_start":
                x1,

            "col_end":
                x2,

            "prediction":
                prediction
        })

    final_prediction = stitch_predictions(
        stitched_predictions,
        height,
        width
    )

    return {

        "image_height":
            height,

        "image_width":
            width,

        "spectral_bands":
            bands,

        "tiles_processed":
            len(tiles),

        "tile_size":
            "16x16",

        "overlap":
            4,

        "model":
            "3D-CNN + Vision Transformer",

        "stitched_prediction_shape":
            list(
                final_prediction.shape
            ),

        "minimum_risk":
            round(
                float(
                    final_prediction.min()
                ),
                4
            ),

        "maximum_risk":
            round(
                float(
                    final_prediction.max()
                ),
                4
            ),

        "mean_risk":
            round(
                float(
                    final_prediction.mean()
                ),
                4
            )
    }


# ==================================================
# WEEK 4 HISTORY
# ==================================================

@app.get("/history")
def history():

    records = load_history()

    return {

        "count":
            len(records),

        "history":
            records
    }


# ==================================================
# SAVE CURRENT FUSED SNAPSHOT
# ==================================================

@app.post("/history/snapshot")
def history_snapshot():

    results = predict_fused_grid(
        stride=2,
        batch_size=16
    )

    analytics_data = calculate_analytics(
        results
    )

    snapshot = create_snapshot(
        analytics_data,

        model=(
            "3D-CNN + "
            "Vision Transformer"
        )
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


# ==================================================
# WEEK 4 RECONSTRUCTED TEMPORAL DEMONSTRATION
# ==================================================

@app.post("/history/demo")
def create_demo_history():

    from datetime import (
        datetime,
        timedelta
    )

    results = predict_fused_grid(
        stride=2,
        batch_size=16
    )

    analytics_data = calculate_analytics(
        results
    )

    total_zones = int(
        analytics_data.get(
            "total_zones",
            len(results)
        )
    )

    current_risk = float(
        analytics_data.get(
            "risk_percentage",
            14.06
        )
    )

    current_acres = float(
        analytics_data.get(
            "risk_acres_estimated",
            current_risk / 100.0 * 1000.0
        )
    )

    current_high = int(
        analytics_data.get(
            "high_risk",
            {}
        ).get(
            "zones",
            144
        )
    )

    current_moderate = int(
        analytics_data.get(
            "moderate",
            {}
        ).get(
            "zones",
            0
        )
    )

    current_healthy = int(
        analytics_data.get(
            "healthy",
            {}
        ).get(
            "zones",
            total_zones - current_high - current_moderate
        )
    )

    now = datetime.now()

    risk_values = [
        8.40,
        10.10,
        11.80,
        13.20,
        current_risk
    ]

    high_values = [
        86,
        103,
        121,
        135,
        current_high
    ]

    moderate_values = [
        0,
        0,
        0,
        0,
        current_moderate
    ]

    healthy_values = [
        total_zones - high_values[0] - moderate_values[0],
        total_zones - high_values[1] - moderate_values[1],
        total_zones - high_values[2] - moderate_values[2],
        total_zones - high_values[3] - moderate_values[3],
        current_healthy
    ]

    labels = [
        "Baseline Scan",
        "Early Stress Signal",
        "Progressive Stress",
        "Elevated Monitoring",
        "Current Scan"
    ]

    reconstructed_history = []

    for index in range(5):

        risk = risk_values[index]

        if index == 4:
            acres = current_acres
        else:
            acres = (
                risk /
                100.0 *
                1000.0
            )

        timestamp = (
            now -
            timedelta(
                days=4 - index
            )
        )

        snapshot = {

            "timestamp":
                timestamp.isoformat(),

            "label":
                labels[index],

            "model":
                "3D-CNN + Vision Transformer",

            "risk_percentage":
                round(
                    risk,
                    2
                ),

            "risk_acres_estimated":
                round(
                    acres,
                    2
                ),

            "high_risk_zones":
                high_values[index],

            "moderate_zones":
                moderate_values[index],

            "healthy_zones":
                healthy_values[index],

            "data_type":
                "reconstructed_demo",

            "temporal_note": (
                "Reconstructed temporal "
                "demonstration generated from "
                "the current synthetic dataset; "
                "not historical satellite imagery."
            )
        }

        reconstructed_history.append(
            snapshot
        )

    history_file = "dataset/history.json"

    import os
    import json

    os.makedirs(
        os.path.dirname(history_file),
        exist_ok=True
    )

    with open(
        history_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            reconstructed_history,
            file,
            indent=2
        )

    return {

        "status":
            "demo_history_created",

        "count":
            len(reconstructed_history),

        "history":
            reconstructed_history,

        "data_type":
            "reconstructed_demo",

        "note": (
            "The temporal series is a "
            "reconstructed demonstration based "
            "on the current synthetic dataset. "
            "It should not be presented as "
            "historical satellite observations."
        )
    }