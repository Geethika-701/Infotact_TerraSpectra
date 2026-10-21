import os
import json
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "frontend",
    "public",
    "predictions.json"
)


OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "frontend",
    "public",
    "analytics.json"
)


with open(
    PREDICTION_FILE,
    "r"
) as file:

    predictions = json.load(file)


total_pixels = len(
    predictions
)


high_risk = [
    p for p in predictions
    if p["risk"] >= 0.75
]


medium_risk = [
    p for p in predictions
    if 0.50 <= p["risk"] < 0.75
]


low_risk = [
    p for p in predictions
    if p["risk"] < 0.50
]


average_risk = np.mean([
    p["risk"]
    for p in predictions
])


# Approximate pixel area
#
# For prototype dashboard only.

pixel_area_acres = (
    1000 / (64 * 64)
)


high_risk_acres = (
    len(high_risk) *
    pixel_area_acres
)


medium_risk_acres = (
    len(medium_risk) *
    pixel_area_acres
)


analytics = {

    "total_farm_acres": 1000,

    "total_pixels": total_pixels,

    "high_risk_pixels": len(
        high_risk
    ),

    "medium_risk_pixels": len(
        medium_risk
    ),

    "low_risk_pixels": len(
        low_risk
    ),

    "high_risk_acres": round(
        high_risk_acres,
        2
    ),

    "medium_risk_acres": round(
        medium_risk_acres,
        2
    ),

    "average_risk": round(
        float(average_risk),
        4
    ),

    "model": "3D-CNN",

    "spectral_bands": 224

}


with open(
    OUTPUT_FILE,
    "w"
) as file:

    json.dump(
        analytics,
        file,
        indent=4
    )


print(
    "Analytics created:"
)

print(
    OUTPUT_FILE
)