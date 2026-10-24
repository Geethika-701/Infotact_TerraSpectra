import json
import os
from datetime import datetime


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(BASE_DIR)

HISTORY_FILE = os.path.join(
    PROJECT_DIR,
    "dataset",
    "history.json"
)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except (
        json.JSONDecodeError,
        OSError
    ):
        return []


def save_snapshot(snapshot):
    history = load_history()

    history.append(snapshot)

    os.makedirs(
        os.path.dirname(HISTORY_FILE),
        exist_ok=True
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            history,
            file,
            indent=2
        )

    return snapshot


def create_snapshot(
    analytics,
    model="3D-CNN + Vision Transformer",
    label=None,
    data_type="live_scan"
):
    now = datetime.now()

    risk_percentage = float(
        analytics.get(
            "risk_percentage",
            0
        )
    )

    risk_acres = float(
        analytics.get(
            "risk_acres_estimated",
            0
        )
    )

    high_risk = int(
        analytics.get(
            "high_risk",
            {}
        ).get(
            "zones",
            0
        )
    )

    moderate = int(
        analytics.get(
            "moderate",
            {}
        ).get(
            "zones",
            0
        )
    )

    healthy = int(
        analytics.get(
            "healthy",
            {}
        ).get(
            "zones",
            0
        )
    )

    return {
        "timestamp": now.isoformat(),

        "label": (
            label or
            now.strftime(
                "%Y-%m-%d %H:%M"
            )
        ),

        "model": model,

        "data_type": data_type,

        "source": (
            "TerraSpectra inference pipeline"
            if data_type == "live_scan"
            else "Reconstructed demonstration data"
        ),

        "risk_percentage": round(
            risk_percentage,
            2
        ),

        "risk_acres_estimated": round(
            risk_acres,
            2
        ),

        "high_risk_zones": high_risk,

        "moderate_zones": moderate,

        "healthy_zones": healthy
    }


def add_progression_information(history):
    if not history:
        return []

    result = []

    previous_risk = None
    previous_acres = None

    for index, item in enumerate(history):

        current = dict(item)

        current["scan_number"] = index + 1

        current_risk = float(
            current.get(
                "risk_percentage",
                0
            )
        )

        current_acres = float(
            current.get(
                "risk_acres_estimated",
                0
            )
        )

        if previous_risk is None:
            current["risk_change_percentage_points"] = 0.0
            current["risk_change_direction"] = "Baseline"
        else:
            risk_change = (
                current_risk -
                previous_risk
            )

            current[
                "risk_change_percentage_points"
            ] = round(
                risk_change,
                2
            )

            if risk_change > 0:
                current[
                    "risk_change_direction"
                ] = "Increasing"

            elif risk_change < 0:
                current[
                    "risk_change_direction"
                ] = "Decreasing"

            else:
                current[
                    "risk_change_direction"
                ] = "Stable"

        if previous_acres is None:
            current["acreage_change"] = 0.0
        else:
            current["acreage_change"] = round(
                current_acres -
                previous_acres,
                2
            )

        result.append(current)

        previous_risk = current_risk
        previous_acres = current_acres

    return result


def get_progression():
    history = load_history()

    return add_progression_information(
        history
    )


def create_demo_history():
    demo_points = [
        {
            "label": "Baseline Scan",
            "risk_percentage": 8.4,
            "risk_acres_estimated": 84.0,
            "high_risk_zones": 86,
            "moderate_zones": 0,
            "healthy_zones": 914
        },
        {
            "label": "Early Stress Signal",
            "risk_percentage": 10.1,
            "risk_acres_estimated": 101.0,
            "high_risk_zones": 103,
            "moderate_zones": 0,
            "healthy_zones": 897
        },
        {
            "label": "Progressive Stress",
            "risk_percentage": 11.8,
            "risk_acres_estimated": 118.0,
            "high_risk_zones": 121,
            "moderate_zones": 0,
            "healthy_zones": 879
        },
        {
            "label": "Elevated Monitoring",
            "risk_percentage": 13.2,
            "risk_acres_estimated": 132.0,
            "high_risk_zones": 135,
            "moderate_zones": 0,
            "healthy_zones": 865
        }
    ]

    history = []

    for point in demo_points:

        record = dict(point)

        record["timestamp"] = (
            datetime.now().isoformat()
        )

        record["model"] = (
            "3D-CNN + Vision Transformer"
        )

        record["data_type"] = (
            "reconstructed_demo"
        )

        record["source"] = (
            "Synthetic reconstructed "
            "demonstration data"
        )

        history.append(record)

    return add_progression_information(
        history
    )


def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)