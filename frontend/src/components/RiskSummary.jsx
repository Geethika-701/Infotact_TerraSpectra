import React from "react";

export default function RiskSummary({
  highRisk = 0,
  moderateRisk = 0,
  healthy = 0,
}) {
  const total =
    highRisk +
    moderateRisk +
    healthy;

  function percentage(value) {
    if (total === 0) {
      return 0;
    }

    return Math.round(
      (value / total) * 100
    );
  }

  return (
    <div className="risk-summary">

      <div className="risk-header">

        <div>
          <span>
            WEEK 2 · AI CLASSIFICATION
          </span>

          <h3>
            Risk Summary
          </h3>
        </div>

        <strong>
          {total} ZONES
        </strong>

      </div>


      <div className="risk-list">

        {/* HIGH */}

        <div className="risk-row">

          <div className="risk-row-title">

            <div className="risk-name">
              <span className="risk-dot high"></span>
              High Risk
            </div>

            <strong>
              {highRisk}
            </strong>

          </div>

          <div className="risk-track">

            <div
              className="risk-progress high"
              style={{
                width: `${percentage(highRisk)}%`,
              }}
            />

          </div>

          <span className="risk-percent">
            {percentage(highRisk)}%
          </span>

        </div>


        {/* MODERATE */}

        <div className="risk-row">

          <div className="risk-row-title">

            <div className="risk-name">
              <span className="risk-dot moderate"></span>
              Moderate
            </div>

            <strong>
              {moderateRisk}
            </strong>

          </div>

          <div className="risk-track">

            <div
              className="risk-progress moderate"
              style={{
                width: `${percentage(moderateRisk)}%`,
              }}
            />

          </div>

          <span className="risk-percent">
            {percentage(moderateRisk)}%
          </span>

        </div>


        {/* HEALTHY */}

        <div className="risk-row">

          <div className="risk-row-title">

            <div className="risk-name">
              <span className="risk-dot healthy"></span>
              Healthy
            </div>

            <strong>
              {healthy}
            </strong>

          </div>

          <div className="risk-track">

            <div
              className="risk-progress healthy"
              style={{
                width: `${percentage(healthy)}%`,
              }}
            />

          </div>

          <span className="risk-percent">
            {percentage(healthy)}%
          </span>

        </div>

      </div>


      <div className="risk-footer">

        <span>
          Classification Model
        </span>

        <strong>
          ● 3D-CNN ACTIVE
        </strong>

      </div>

    </div>
  );
}