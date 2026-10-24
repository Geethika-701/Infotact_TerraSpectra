
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os
import json
from datetime import datetime, timedelta

import numpy as np

from backend.predict import predict_grid

from backend.fusion_predict import (
    predict_fused_grid,
    predict_single_patch,
    normalize_cube,
    create_patch,
    load_models,
    predict_batch
)

from backend.analytics import (
    calculate_analytics
)

from backend.history import (
    load_history,
    save_snapshot,
    create_snapshot,
    get_progression
)

from backend.tiling import (
    generate_tiles,
    stitch_predictions,
    tile_info
)

from backend.rasterio_ingestion import (
    rasterio_summary,
    create_mock_geotiff
)

from backend.pca_processing import (
    pca_summary
)

from backend.memory_audit import (
    run_memory_audit
)

from backend.validation import (
    validate_models
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

CUBE_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "raw",
    "mock_hyperspectral_cube.npy"
)

HISTORY_FILE = os.path.join(
    PROJECT_DIR,
    "dataset",
    "history.json"
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="TerraSpectra API",
    version="4.1.0"
)


# =========================================================
# CORS
# =========================================================

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


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "project":
            "TerraSpectra",

        "status":
            "online",

        "version":
            "4.1.0",

        "weeks": {

            "week_1": [
                "Hyperspectral Processing",
                "Rasterio GeoTIFF Ingestion",
                "PCA Dimensionality Reduction"
            ],

            "week_2": [
                "3D-CNN Classification",
                "Risk Mapping",
                "Memory Conscious Processing",
                "GIS Validation"
            ],

            "week_3": [
                "Vision Transformer",
                "Multi-Head Self-Attention",
                "CNN + ViT Model Fusion",
                "Analytics",
                "Model Validation"
            ],

            "week_4": [
                "FastAPI",
                "Satellite Tiling",
                "Overlapping Tiles",
                "Prediction Stitching",
                "Historical Timeline",
                "Temporal Progression Analysis"
            ]
        }
    }


# =========================================================
# HEALTH
# =========================================================

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


# =========================================================
# WEEK 1
# RASTERIO HYPERSPECTRAL INGESTION
# =========================================================

@app.get("/rasterio")
def rasterio_endpoint():

    return rasterio_summary()


# =========================================================
# CREATE MOCK HYPERSPECTRAL GEOTIFF
# =========================================================

@app.post("/rasterio/create-geotiff")
def create_geotiff_endpoint():

    path = create_mock_geotiff()

    return {

        "status":
            "created",

        "file":
            os.path.basename(path),

        "message": (
            "Synthetic hyperspectral "
            "GeoTIFF created successfully."
        )
    }


# =========================================================
# WEEK 1
# PCA DIMENSIONALITY REDUCTION
# =========================================================

@app.get("/pca")
def pca_endpoint():

    return pca_summary(
        components=3
    )


# =========================================================
# WEEK 2
# ORIGINAL 3D-CNN PREDICTIONS
# =========================================================

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


# =========================================================
# WEEK 3
# FUSED 3D-CNN + VISION TRANSFORMER PREDICTIONS
# =========================================================

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

            "enabled":
                True,

            "cnn_weight":
                0.60,

            "vit_weight":
                0.40
        }
    }


# =========================================================
# WEEK 3
# ANALYTICS
# =========================================================

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


# =========================================================
# MODEL INFORMATION
# =========================================================

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


# =========================================================
# WEEK 3
# MODEL BEHAVIOR COMPARISON
# =========================================================

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

        if float(
            item.get(
                "cnn_risk",
                0
            )
        ) >= 0.7:

            cnn_high += 1

        if float(
            item.get(
                "vit_risk",
                0
            )
        ) >= 0.7:

            vit_high += 1

        if float(
            item.get(
                "risk",
                0
            )
        ) >= 0.7:

            fused_high += 1

    if total > 0:

        cnn_percentage = (
            cnn_high /
            total *
            100
        )

        vit_percentage = (
            vit_high /
            total *
            100
        )

        fusion_percentage = (
            fused_high /
            total *
            100
        )

    else:

        cnn_percentage = 0
        vit_percentage = 0
        fusion_percentage = 0

    return {

        "status":
            "success",

        "total_zones":
            total,

        "cnn": {

            "high_risk_zones":
                cnn_high,

            "high_risk_percentage":
                round(
                    cnn_percentage,
                    2
                )
        },

        "vit": {

            "high_risk_zones":
                vit_high,

            "high_risk_percentage":
                round(
                    vit_percentage,
                    2
                )
        },

        "fusion": {

            "high_risk_zones":
                fused_high,

            "high_risk_percentage":
                round(
                    fusion_percentage,
                    2
                )
        },

        "note": (
            "This endpoint compares prediction "
            "behavior. For actual accuracy, precision, "
            "recall and F1 validation, use /validation."
        )
    }


# =========================================================
# WEEK 3
# ACTUAL MODEL VALIDATION
# =========================================================

@app.get("/validation")
def model_validation():

    try:

        results = validate_models(
            sample_limit=1000,
            batch_size=32
        )

        return results

    except Exception as error:

        return {

            "status":
                "error",

            "message":
                str(error)
        }


# =========================================================
# ANOMALY ANALYSIS
# =========================================================

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


# =========================================================
# WEEK 2
# MEMORY HANDLING AUDIT
# =========================================================

@app.get("/memory-audit")
def memory_audit():

    try:

        result = run_memory_audit()

        return result

    except Exception as error:

        return {

            "status":
                "error",

            "message":
                str(error)
        }


# =========================================================
# WEEK 4
# TILING INFORMATION
# =========================================================

@app.get("/tiling")
def tiling():

    if not os.path.exists(
        CUBE_PATH
    ):

        return {

            "status":
                "error",

            "message":
                "Hyperspectral cube not found."
        }

    cube = np.load(
        CUBE_PATH
    )

    return tile_info(
        cube,
        tile_height=16,
        tile_width=16,
        overlap=4
    )


# =========================================================
# WEEK 4
# REAL TILED MODEL INFERENCE
# =========================================================

@app.get("/tiling/inference")
def tiled_inference():

    if not os.path.exists(
        CUBE_PATH
    ):

        return {

            "status":
                "error",

            "message":
                "Hyperspectral cube not found."
        }

    cube = np.load(
        CUBE_PATH
    )

    normalized_cube = normalize_cube(
        cube
    )

    height = normalized_cube.shape[0]
    width = normalized_cube.shape[1]
    bands = normalized_cube.shape[2]

    tiles = generate_tiles(
        normalized_cube,
        tile_height=16,
        tile_width=16,
        overlap=4
    )

    stitched_predictions = []

    cnn, vit, device = load_models()

    for tile in tiles:

        y1 = tile["row_start"]
        y2 = tile["row_end"]

        x1 = tile["col_start"]
        x2 = tile["col_end"]

        patches = []

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

        if len(patches) == 0:

            continue

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

        "status":
            "success",

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


# =========================================================
# WEEK 4
# HISTORICAL TIMELINE
# =========================================================

@app.get("/history")
def history():

    try:

        records = get_progression()

        return {

            "status":
                "success",

            "count":
                len(records),

            "history":
                records
        }

    except Exception as error:

        return {

            "status":
                "error",

            "message":
                str(error)
        }


# =========================================================
# HISTORICAL PROGRESSION
# =========================================================

@app.get("/history/progression")
def history_progression():

    try:

        records = get_progression()

        return {

            "status":
                "success",

            "count":
                len(records),

            "history":
                records,

            "progression_available":
                len(records) > 1
        }

    except Exception as error:

        return {

            "status":
                "error",

            "message":
                str(error)
        }


# =========================================================
# SAVE CURRENT FUSED SNAPSHOT
# =========================================================

@app.post("/history/snapshot")
def history_snapshot():

    try:

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
            ),

            data_type="live_scan"
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

    except Exception as error:

        return {

            "status":
                "error",

            "message":
                str(error)
        }


# =========================================================
# RECONSTRUCTED TEMPORAL DEMONSTRATION
# =========================================================

@app.post("/history/demo")
def create_demo_history():

    try:

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
                current_risk /
                100.0 *
                1000.0
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
                total_zones -
                current_high -
                current_moderate
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

            total_zones -
            high_values[0] -
            moderate_values[0],

            total_zones -
            high_values[1] -
            moderate_values[1],

            total_zones -
            high_values[2] -
            moderate_values[2],

            total_zones -
            high_values[3] -
            moderate_values[3],

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

                "source":
                    "Synthetic reconstructed "
                    "demonstration data",

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

        os.makedirs(
            os.path.dirname(
                HISTORY_FILE
            ),
            exist_ok=True
        )

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                reconstructed_history,
                file,
                indent=2
            )

        progression = get_progression()

        return {

            "status":
                "demo_history_created",

            "count":
                len(progression),

            "history":
                progression,

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

    except Exception as error:

        return {

            "status":
                "error",

            "message":
                str(error)
        }
