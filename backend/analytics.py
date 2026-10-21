FARM_ACRES = 1000.0


def calculate_analytics(predictions):

    total = len(predictions)

    high_risk = 0
    moderate_risk = 0
    healthy = 0

    for item in predictions:

        risk = float(
            item.get("risk", 0)
        )

        if risk >= 0.7:

            high_risk += 1

        elif risk >= 0.4:

            moderate_risk += 1

        else:

            healthy += 1

    risk_zones = (
        high_risk +
        moderate_risk
    )

    if total > 0:

        risk_percentage = (
            risk_zones / total
        ) * 100

        high_percentage = (
            high_risk / total
        ) * 100

        moderate_percentage = (
            moderate_risk / total
        ) * 100

        healthy_percentage = (
            healthy / total
        ) * 100

    else:

        risk_percentage = 0
        high_percentage = 0
        moderate_percentage = 0
        healthy_percentage = 0

    risk_acres = (
        risk_zones / total * FARM_ACRES
        if total > 0
        else 0
    )

    high_risk_acres = (
        high_risk / total * FARM_ACRES
        if total > 0
        else 0
    )

    moderate_acres = (
        moderate_risk / total * FARM_ACRES
        if total > 0
        else 0
    )

    healthy_acres = (
        healthy / total * FARM_ACRES
        if total > 0
        else 0
    )

    return {

        "farm_area_acres": FARM_ACRES,

        "total_zones": total,

        "risk_zones": risk_zones,

        "risk_percentage": round(
            risk_percentage,
            2
        ),

        "risk_acres_estimated": round(
            risk_acres,
            2
        ),

        "high_risk": {

            "zones": high_risk,

            "percentage": round(
                high_percentage,
                2
            ),

            "estimated_acres": round(
                high_risk_acres,
                2
            )
        },

        "moderate": {

            "zones": moderate_risk,

            "percentage": round(
                moderate_percentage,
                2
            ),

            "estimated_acres": round(
                moderate_acres,
                2
            )
        },

        "healthy": {

            "zones": healthy,

            "percentage": round(
                healthy_percentage,
                2
            ),

            "estimated_acres": round(
                healthy_acres,
                2
            )
        },

        "chemical_anomalies": {

            "chemical_stress_zones":
                risk_zones,

            "spectral_anomaly_zones":
                risk_zones,

            "stress_class": (
                "Elevated"
                if risk_percentage >= 15
                else "Normal"
            )
        }
    }