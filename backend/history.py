import json
import os
from datetime import datetime


HISTORY_FILE = "dataset/history.json"


def load_history():

    if not os.path.exists(
        HISTORY_FILE
    ):
        return []

    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_snapshot(
    snapshot
):

    history = load_history()

    history.append(
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
            history,
            file,
            indent=2
        )

    return snapshot


def create_snapshot(
    analytics,
    label=None
):

    now = datetime.now()

    snapshot = {
        "timestamp": now.isoformat(),
        "label": label or now.strftime(
            "%Y-%m-%d"
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