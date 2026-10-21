
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = ROOT / "results" / "predictions" / "predictions.json"
OUTPUT_PATH = ROOT / "results" / "predictions" / "analytics.json"
FRONTEND_OUTPUT = ROOT / "frontend" / "public" / "analytics.json"

# Set this only when the actual ground area per pixel is known.
# Example: 0.0001 means each pixel represents 0.0001 acres.
# Leave as None for the current mock dataset.
PIXEL_AREA_ACRES = None


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}\n"
            "Run the backend prediction script first."
        )

    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    predictions = data.get("predictions", [])

    healthy_count = sum(
        1 for item in predictions
        if item.get("status", item.get("class")) == "healthy"
    )

    stressed_count = sum(
        1 for item in predictions
        if item.get("status", item.get("class"))
        in ("chemically_stressed", "stressed")
    )

    total = len(predictions)

    estimated_acres = (
        stressed_count * PIXEL_AREA_ACRES
        if PIXEL_AREA_ACRES is not None
        else None
    )

    analytics = {
        "total_classified_pixels": total,
        "healthy_pixels": healthy_count,
        "stressed_pixels": stressed_count,
        "stressed_percentage": (
            round(stressed_count / total * 100, 2)
            if total else 0
        ),
        "estimated_acres_at_risk": estimated_acres,
        "acreage_note": (
            "Calculated using configured pixel area."
            if PIXEL_AREA_ACRES is not None
            else "Unavailable: real ground area per pixel is not configured."
        ),
        "chemical_anomalies": [],
        "chemical_anomaly_note": (
            "Chemical-specific identities cannot be determined from "
            "the current synthetic healthy/stressed labels."
        ),
        "data_source": "Mock hyperspectral predictions"
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(analytics, file, indent=2)

    FRONTEND_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_PATH, FRONTEND_OUTPUT)

    print("Week 3 analytics generated successfully.")
    print("Total pixels:", total)
    print("Healthy pixels:", healthy_count)
    print("Stressed pixels:", stressed_count)
    print("Estimated acres at risk:", estimated_acres)
    print("Saved:", OUTPUT_PATH)
    print("Frontend file:", FRONTEND_OUTPUT)


if __name__ == "__main__":
    main()
