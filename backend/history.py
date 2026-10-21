import json
import os
from datetime import datetime


HISTORY_FILE = "dataset/history.json"


def load_history():
    """
    Load all historical crop-health snapshots.
    """

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return []


def save_snapshot(snapshot):
    """
    Save a new historical snapshot.
    """

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
    label=None
):
    """
    Create a snapshot from current analytics.
    """

    now = datetime.now()

    snapshot = {
        "timestamp": now.isoformat(),

        "label": label or now.strftime(
            "%Y-%m-%d %H:%M"
        ),

        "risk_percentage": analytics[
            "risk_percentage"
        ],

        "risk_acres_estimated": analytics[
            "risk_acres_estimated"
        ],

        "high_risk_zones": analytics[
            "high_risk"
        ]["zones"],

        "moderate_zones": analytics[
            "moderate"
        ]["zones"],

        "healthy_zones": analytics[
            "healthy"
        ]["zones"]
    }

    return snapshot


def clear_history():
    """
    Clear historical snapshots.

    Useful during development/testing.
    """

    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)