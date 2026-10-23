import React from "react";

export default function AnalyticsPanel({
  analytics,
  anomalies,
  modelComparison,
}) {
  if (!analytics) {
    return (
      <section className="analytics-panel">

        <div className="section-heading">

          <div>

            <span>
              AI INTELLIGENCE
            </span>

            <h2>
              Risk Analytics
            </h2>

          </div>

          <div className="prediction-count">
            LOADING ANALYTICS...
          </div>

        </div>

      </section>
    );
  }

  const riskAcres = Number(
    analytics.risk_acres_estimated || 0
  ).toFixed(2);

  const riskPercentage = Number(
    analytics.risk_percentage || 0
  ).toFixed(2);

  const riskZones =
    analytics.risk_zones || 0;

  const totalZones =
    analytics.total_zones || 0;

  const highRiskAcres = Number(
    analytics.high_risk?.estimated_acres || 0
  ).toFixed(2);

  const moderateAcres = Number(
    analytics.moderate?.estimated_acres || 0
  ).toFixed(2);

  const healthyAcres = Number(
    analytics.healthy?.estimated_acres || 0
  ).toFixed(2);

  const spectralAnomalies =
    anomalies?.spectral_anomaly_zones || 0;

  const chemicalStress =
    anomalies?.chemical_stress_zones || 0;

  const stressClass =
    anomalies?.stress_class || "Normal";

  const cnnRisk = Number(
    modelComparison?.cnn?.high_risk_percentage || 0
  ).toFixed(2);

  const vitRisk = Number(
    modelComparison?.vit?.high_risk_percentage || 0
  ).toFixed(2);

  const fusionRisk = Number(
    modelComparison?.fusion?.high_risk_percentage || 0
  ).toFixed(2);

  const maxModelRisk = Math.max(
    Number(cnnRisk),
    Number(vitRisk),
    Number(fusionRisk),
    1
  );

  return (
    <section className="analytics-panel">

      <div className="section-heading">

        <div>

          <span>
            AI INTELLIGENCE
          </span>

          <h2>
            Risk Analytics
          </h2>

        </div>

        <div className="analytics-model-badge">
          <span>●</span>
          3D-CNN + VIT FUSION
        </div>

      </div>


      <div className="analytics-cards">

        <div className="analytics-card analytics-card-primary">

          <div className="analytics-card-icon">
            ◉
          </div>

          <div className="analytics-card-content">

            <span>
              ESTIMATED AREA AT RISK
            </span>

            <div className="analytics-main-value">
              <strong>
                {riskAcres}
              </strong>

              <small>
                ACRES
              </small>
            </div>

          </div>

          <div className="analytics-card-foot">
            {riskPercentage}% of monitored area
          </div>

        </div>


        <div className="analytics-card">

          <div className="analytics-card-icon">
            ◈
          </div>

          <div className="analytics-card-content">

            <span>
              RISK ZONES
            </span>

            <div className="analytics-main-value">

              <strong>
                {riskZones}
              </strong>

              <small>
                / {totalZones}
              </small>

            </div>

          </div>

          <div className="analytics-card-foot">
            Spatial classification
          </div>

        </div>


        <div className="analytics-card">

          <div className="analytics-card-icon">
            ≋
          </div>

          <div className="analytics-card-content">

            <span>
              SPECTRAL ANOMALIES
            </span>

            <div className="analytics-main-value">

              <strong>
                {spectralAnomalies}
              </strong>

              <small>
                ZONES
              </small>

            </div>

          </div>

          <div className="analytics-card-foot">
            Stress signatures detected
          </div>

        </div>


        <div className="analytics-card">

          <div className="analytics-card-icon">
            ✦
          </div>

          <div className="analytics-card-content">

            <span>
              STRESS STATUS
            </span>

            <div className="analytics-main-value">

              <strong className="stress-status">
                {stressClass}
              </strong>

            </div>

          </div>

          <div className="analytics-card-foot">
            {chemicalStress} stress zones detected
          </div>

        </div>

      </div>


      <div className="analytics-lower-grid">


        <div className="analytics-chart-card">

          <div className="analytics-chart-header">

            <div>

              <span>
                FARM COVERAGE
              </span>

              <h3>
                Area Distribution
              </h3>

            </div>

            <strong>
              1,000 ACRES
            </strong>

          </div>


          <div className="acreage-bars">

            <div className="acreage-row">

              <div className="acreage-label">

                <span>
                  <i className="acreage-dot high"></i>
                  High Risk
                </span>

                <strong>
                  {highRiskAcres} ac
                </strong>

              </div>

              <div className="acreage-track">

                <div
                  className="acreage-progress high"
                  style={{
                    width: `${Math.min(
                      (Number(highRiskAcres) /
                        1000) *
                        100,
                      100
                    )}%`,
                  }}
                />

              </div>

            </div>


            <div className="acreage-row">

              <div className="acreage-label">

                <span>
                  <i className="acreage-dot moderate"></i>
                  Moderate
                </span>

                <strong>
                  {moderateAcres} ac
                </strong>

              </div>

              <div className="acreage-track">

                <div
                  className="acreage-progress moderate"
                  style={{
                    width: `${Math.min(
                      (Number(moderateAcres) /
                        1000) *
                        100,
                      100
                    )}%`,
                  }}
                />

              </div>

            </div>


            <div className="acreage-row">

              <div className="acreage-label">

                <span>
                  <i className="acreage-dot healthy"></i>
                  Healthy
                </span>

                <strong>
                  {healthyAcres} ac
                </strong>

              </div>

              <div className="acreage-track">

                <div
                  className="acreage-progress healthy"
                  style={{
                    width: `${Math.min(
                      (Number(healthyAcres) /
                        1000) *
                        100,
                      100
                    )}%`,
                  }}
                />

              </div>

            </div>

          </div>

        </div>


        <div className="analytics-chart-card">

          <div className="analytics-chart-header">

            <div>

              <span>
                MODEL INTELLIGENCE
              </span>

              <h3>
                Risk Detection Comparison
              </h3>

            </div>

            <strong>
              SELF-ATTENTION
            </strong>

          </div>


          <div className="model-comparison">

            <div className="model-row">

              <div className="model-row-top">

                <span>
                  3D-CNN
                </span>

                <strong>
                  {cnnRisk}%
                </strong>

              </div>

              <div className="model-track">

                <div
                  className="model-progress cnn"
                  style={{
                    width: `${Math.min(
                      (Number(cnnRisk) /
                        maxModelRisk) *
                        100,
                      100
                    )}%`,
                  }}
                />

              </div>

            </div>


            <div className="model-row">

              <div className="model-row-top">

                <span>
                  Vision Transformer
                </span>

                <strong>
                  {vitRisk}%
                </strong>

              </div>

              <div className="model-track">

                <div
                  className="model-progress vit"
                  style={{
                    width: `${Math.min(
                      (Number(vitRisk) /
                        maxModelRisk) *
                        100,
                      100
                    )}%`,
                  }}
                />

              </div>

            </div>


            <div className="model-row">

              <div className="model-row-top">

                <span>
                  CNN + ViT Fusion
                </span>

                <strong>
                  {fusionRisk}%
                </strong>

              </div>

              <div className="model-track">

                <div
                  className="model-progress fusion"
                  style={{
                    width: `${Math.min(
                      (Number(fusionRisk) /
                        maxModelRisk) *
                        100,
                      100
                    )}%`,
                  }}
                />

              </div>

            </div>

          </div>


          <div className="model-note">

            <span>
              ✦
            </span>

            <p>
              Vision Transformer self-attention
              analyzes relationships across
              spectral bands while the fusion model
              combines spatial and spectral signals.
            </p>

          </div>

        </div>

      </div>


      <div className="anomaly-summary">

        <div className="anomaly-summary-left">

          <div className="anomaly-icon">
            ≋
          </div>

          <div>

            <span>
              SPECTRAL ANOMALY ANALYSIS
            </span>

            <h3>
              Crop Stress Signatures
            </h3>

            <p>
              {spectralAnomalies} monitored zones
              show elevated spectral stress
              signatures requiring further
              agronomic inspection.
            </p>

          </div>

        </div>


        <div className="anomaly-summary-stat">

          <strong>
            {spectralAnomalies}
          </strong>

          <span>
            ANOMALOUS ZONES
          </span>

        </div>


        <div className="anomaly-summary-stat">

          <strong>
            {riskAcres}
          </strong>

          <span>
            ACRES AT RISK
          </span>

        </div>

      </div>

    </section>
  );
}