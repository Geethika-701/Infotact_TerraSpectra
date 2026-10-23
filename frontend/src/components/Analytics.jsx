import React, {
  useEffect,
  useState
} from "react";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:5000";

export default function Analytics({
  id
}) {
  const [analytics, setAnalytics] =
    useState(null);

  const [comparison, setComparison] =
    useState(null);

  const [anomalies, setAnomalies] =
    useState(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState(false);

  useEffect(() => {
    async function loadAnalytics() {
      try {
        setLoading(true);
        setError(false);

        const analyticsResponse =
          await fetch(
            `${API_URL}/analytics`
          );

        const comparisonResponse =
          await fetch(
            `${API_URL}/models/comparison`
          );

        const anomaliesResponse =
          await fetch(
            `${API_URL}/anomalies`
          );

        if (
          !analyticsResponse.ok ||
          !comparisonResponse.ok ||
          !anomaliesResponse.ok
        ) {
          throw new Error(
            "Analytics API failed"
          );
        }

        const analyticsData =
          await analyticsResponse.json();

        const comparisonData =
          await comparisonResponse.json();

        const anomaliesData =
          await anomaliesResponse.json();

        setAnalytics(
          analyticsData
        );

        setComparison(
          comparisonData
        );

        setAnomalies(
          anomaliesData
        );
      } catch (error) {
        console.error(
          "Analytics connection error:",
          error
        );

        setError(true);
      } finally {
        setLoading(false);
      }
    }

    loadAnalytics();
  }, []);

  if (loading) {
    return (
      <section
        id={id}
        className="analytics-section"
      >

        <div className="analytics-heading">

          <div>

            <span>
              AI INSIGHTS
            </span>

            <h2>
              Field Analytics
            </h2>

            <p>
              Processing crop health,
              spectral anomalies and
              model intelligence.
            </p>

          </div>

          <div className="analytics-loading">
            ANALYZING...
          </div>

        </div>

      </section>
    );
  }

  if (error || !analytics) {
    return (
      <section
        id={id}
        className="analytics-section"
      >

        <div className="analytics-heading">

          <div>

            <span>
              AI INSIGHTS
            </span>

            <h2>
              Field Analytics
            </h2>

            <p>
              Analytics data is temporarily
              unavailable.
            </p>

          </div>

          <div className="analytics-error">
            DATA UNAVAILABLE
          </div>

        </div>

      </section>
    );
  }

  const totalZones =
    Number(
      analytics.total_zones
    ) || 0;

  const highRiskZones =
    Number(
      analytics.high_risk?.zones
    ) || 0;

  const moderateZones =
    Number(
      analytics.moderate?.zones
    ) || 0;

  const healthyZones =
    Number(
      analytics.healthy?.zones
    ) || 0;

  const highPercentage =
    totalZones > 0
      ? (highRiskZones /
          totalZones) *
        100
      : 0;

  const moderatePercentage =
    totalZones > 0
      ? (moderateZones /
          totalZones) *
        100
      : 0;

  const healthyPercentage =
    totalZones > 0
      ? (healthyZones /
          totalZones) *
        100
      : 0;

  const comparisonData =
    comparison?.comparison ||
    comparison ||
    {};

  function getPercentage(
    modelData
  ) {
    if (!modelData) {
      return 0;
    }

    if (
      modelData.high_risk_percentage !==
      undefined
    ) {
      return Number(
        modelData.high_risk_percentage
      );
    }

    if (
      modelData.high_risk?.percentage !==
      undefined
    ) {
      return Number(
        modelData.high_risk.percentage
      );
    }

    if (
      modelData.high_risk?.percentage_value !==
      undefined
    ) {
      return Number(
        modelData.high_risk.percentage_value
      );
    }

    return 0;
  }

  const cnnPercentage =
    getPercentage(
      comparisonData.cnn
    );

  const vitPercentage =
    getPercentage(
      comparisonData.vit
    );

  const fusionPercentage =
    getPercentage(
      comparisonData.fusion
    );

  const anomalyZones =
    Number(
      anomalies?.chemical_stress_zones
    ) || 0;

  return (
    <section
      id={id}
      className="analytics-section"
    >

      <div className="analytics-heading">

        <div>

          <span>
            AI INSIGHTS
          </span>

          <h2>
            Field Analytics
          </h2>

          <p>
            Spatial-spectral intelligence
            for crop health assessment
            and anomaly detection.
          </p>

        </div>

        <div className="analytics-model-status">

          <span></span>

          3D-CNN + VISION TRANSFORMER

        </div>

      </div>

      <div className="analytics-kpi-grid">

        <div className="analytics-kpi">

          <span>
            AT-RISK ACREAGE
          </span>

          <strong>
            {Number(
              analytics.risk_acres_estimated
            ).toFixed(2)}
          </strong>

          <small>
            ACRES
          </small>

        </div>

        <div className="analytics-kpi">

          <span>
            RISK ZONES
          </span>

          <strong>
            {analytics.risk_zones}
          </strong>

          <small>
            OF {totalZones} ZONES
          </small>

        </div>

        <div className="analytics-kpi">

          <span>
            SPECTRAL ANOMALIES
          </span>

          <strong>
            {anomalyZones}
          </strong>

          <small>
            DETECTED
          </small>

        </div>

        <div className="analytics-kpi">

          <span>
            FIELD RISK
          </span>

          <strong>
            {Number(
              analytics.risk_percentage
            ).toFixed(2)}%
          </strong>

          <small>
            OF MONITORED AREA
          </small>

        </div>

      </div>

      <div className="analytics-content-grid">

        <div className="analytics-panel">

          <div className="analytics-panel-header">

            <div>

              <span>
                FIELD DISTRIBUTION
              </span>

              <h3>
                Crop Health Zones
              </h3>

            </div>

            <strong>
              {totalZones} ZONES
            </strong>

          </div>

          <div className="health-bars">

            <div className="health-bar-row">

              <div className="health-bar-label">

                <span>
                  <i className="health-dot high"></i>
                  High Risk
                </span>

                <strong>
                  {highRiskZones}
                </strong>

              </div>

              <div className="health-bar-track">

                <div
                  className="health-bar-fill high"
                  style={{
                    width: `${Math.min(
                      highPercentage,
                      100
                    )}%`
                  }}
                ></div>

              </div>

              <small>
                {highPercentage.toFixed(2)}%
              </small>

            </div>

            <div className="health-bar-row">

              <div className="health-bar-label">

                <span>
                  <i className="health-dot moderate"></i>
                  Moderate
                </span>

                <strong>
                  {moderateZones}
                </strong>

              </div>

              <div className="health-bar-track">

                <div
                  className="health-bar-fill moderate"
                  style={{
                    width: `${Math.min(
                      moderatePercentage,
                      100
                    )}%`
                  }}
                ></div>

              </div>

              <small>
                {moderatePercentage.toFixed(2)}%
              </small>

            </div>

            <div className="health-bar-row">

              <div className="health-bar-label">

                <span>
                  <i className="health-dot healthy"></i>
                  Healthy
                </span>

                <strong>
                  {healthyZones}
                </strong>

              </div>

              <div className="health-bar-track">

                <div
                  className="health-bar-fill healthy"
                  style={{
                    width: `${Math.min(
                      healthyPercentage,
                      100
                    )}%`
                  }}
                ></div>

              </div>

              <small>
                {healthyPercentage.toFixed(2)}%
              </small>

            </div>

          </div>

          <div className="analytics-acreage-row">

            <div>

              <span>
                HIGH-RISK AREA
              </span>

              <strong>
                {Number(
                  analytics.high_risk
                    ?.estimated_acres
                ).toFixed(2)} AC
              </strong>

            </div>

            <div>

              <span>
                MODERATE AREA
              </span>

              <strong>
                {Number(
                  analytics.moderate
                    ?.estimated_acres
                ).toFixed(2)} AC
              </strong>

            </div>

            <div>

              <span>
                HEALTHY AREA
              </span>

              <strong>
                {Number(
                  analytics.healthy
                    ?.estimated_acres
                ).toFixed(2)} AC
              </strong>

            </div>

          </div>

        </div>

        <div className="analytics-panel">

          <div className="analytics-panel-header">

            <div>

              <span>
                MODEL INTELLIGENCE
              </span>

              <h3>
                Risk Comparison
              </h3>

            </div>

            <strong>
              FUSION 60 / 40
            </strong>

          </div>

          <div className="model-bars">

            <div className="model-row">

              <div className="model-row-top">

                <span>
                  3D-CNN
                </span>

                <strong>
                  {cnnPercentage.toFixed(2)}%
                </strong>

              </div>

              <div className="model-track">

                <div
                  className="model-fill cnn"
                  style={{
                    width: `${Math.min(
                      cnnPercentage * 5,
                      100
                    )}%`
                  }}
                ></div>

              </div>

            </div>

            <div className="model-row">

              <div className="model-row-top">

                <span>
                  Vision Transformer
                </span>

                <strong>
                  {vitPercentage.toFixed(2)}%
                </strong>

              </div>

              <div className="model-track">

                <div
                  className="model-fill vit"
                  style={{
                    width: `${Math.min(
                      vitPercentage * 5,
                      100
                    )}%`
                  }}
                ></div>

              </div>

            </div>

            <div className="model-row">

              <div className="model-row-top">

                <span>
                  Fused Prediction
                </span>

                <strong>
                  {fusionPercentage.toFixed(2)}%
                </strong>

              </div>

              <div className="model-track">

                <div
                  className="model-fill fusion"
                  style={{
                    width: `${Math.min(
                      fusionPercentage * 5,
                      100
                    )}%`
                  }}
                ></div>

              </div>

            </div>

          </div>

          <div className="fusion-info">

            <div>

              <span>
                CNN WEIGHT
              </span>

              <strong>
                60%
              </strong>

            </div>

            <div>

              <span>
                ViT WEIGHT
              </span>

              <strong>
                40%
              </strong>

            </div>

            <div>

              <span>
                STATUS
              </span>

              <strong>
                ACTIVE
              </strong>

            </div>

          </div>

        </div>

      </div>

      <div className="analytics-bottom-grid">

        <div className="anomaly-card">

          <div className="anomaly-icon">
            ◈
          </div>

          <div className="anomaly-content">

            <span>
              SPECTRAL STRESS ANALYSIS
            </span>

            <h3>
              {anomalyZones} zones show
              detectable spectral anomalies
            </h3>

            <p>
              Hyperspectral signatures are
              evaluated across the monitored
              spectral bands to identify
              areas requiring attention.
            </p>

          </div>

          <div className="anomaly-value">

            <strong>
              {anomalyZones}
            </strong>

            <span>
              ZONES
            </span>

          </div>

        </div>

        <div className="stress-status-card">

          <span>
            SPECTRAL STATUS
          </span>

          <strong>
            NORMAL RANGE
          </strong>

          <p>
            Current spectral stress
            classification is within the
            monitored reference range.
          </p>

        </div>

      </div>

    </section>
  );
}