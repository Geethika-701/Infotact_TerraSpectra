import React, {
  useEffect,
  useState
} from "react";

import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import StatCard from "./components/StatCard";
import FarmMap from "./components/FarmMap";
import RiskSummary from "./components/RiskSummary";
import SpectralAnalysis from "./components/SpectralAnalysis";
import Pipeline from "./components/Pipeline";
import Analytics from "./components/Analytics";
import HistoryTimeline from "./components/HistoryTimeline";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:5000";

export default function App() {
  const [predictions, setPredictions] =
    useState([]);

  const [analyticsData, setAnalyticsData] =
    useState(null);

  const [loading, setLoading] =
    useState(true);

  const [apiOnline, setApiOnline] =
    useState(false);

  useEffect(() => {
    async function loadDashboardData() {
      try {
        setLoading(true);

        const predictionsResponse =
          await fetch(
            `${API_URL}/predictions/fused`
          );

        if (!predictionsResponse.ok) {
          throw new Error(
            "Fused prediction API failed"
          );
        }

        const predictionsData =
          await predictionsResponse.json();

        setPredictions(
          predictionsData.predictions || []
        );

        setApiOnline(true);

        try {
          const analyticsResponse =
            await fetch(
              `${API_URL}/analytics`
            );

          if (analyticsResponse.ok) {
            const data =
              await analyticsResponse.json();

            setAnalyticsData(data);
          }
        } catch (analyticsError) {
          console.error(
            "Analytics API error:",
            analyticsError
          );
        }
      } catch (error) {
        console.error(
          "Backend connection error:",
          error
        );

        setApiOnline(false);
        setPredictions([]);
      } finally {
        setLoading(false);
      }
    }

    loadDashboardData();
  }, []);

  const highRisk =
    analyticsData?.high_risk?.zones ??
    predictions.filter(
      (item) =>
        Number(item.risk) >= 0.7
    ).length;

  const moderateRisk =
    analyticsData?.moderate?.zones ??
    predictions.filter(
      (item) =>
        Number(item.risk) >= 0.4 &&
        Number(item.risk) < 0.7
    ).length;

  const healthy =
    analyticsData?.healthy?.zones ??
    predictions.filter(
      (item) =>
        Number(item.risk) < 0.4
    ).length;

  const total =
    analyticsData?.total_zones ??
    predictions.length;

  const riskPercentage =
    analyticsData?.risk_percentage ??
    (
      total > 0
        ? (
            (highRisk + moderateRisk) /
            total
          ) * 100
        : 0
    );

  return (
    <div className="app">

      <Sidebar />

      <main className="main-content">

        <Header
          apiOnline={apiOnline}
        />

        <section
          id="dashboard-section"
          className="dashboard"
        >

          <div className="page-heading">

            <div>

              <span className="eyebrow">
                PRECISION AGRICULTURE · AI MONITORING
              </span>

              <h1>
                Farm Overview
              </h1>

              <p>
                Hyperspectral crop stress
                detection using spatial-spectral
                deep learning.
              </p>

            </div>

            <div
              className={
                apiOnline
                  ? "backend-status online"
                  : "backend-status"
              }
            >

              <span></span>

              {loading
                ? "CONNECTING..."
                : apiOnline
                ? "AI ENGINE ONLINE"
                : "AI ENGINE OFFLINE"}

            </div>

          </div>

          <div className="stats-grid">

            <StatCard
              label="FARM AREA"
              value="1,000"
              unit="ACRES"
            />

            <StatCard
              label="SPECTRAL BANDS"
              value="224"
              unit="BANDS"
            />

            <StatCard
              label="MODEL"
              value="3D-CNN + ViT"
              unit="FUSION"
            />

            <StatCard
              label="RISK ZONES"
              value={`${Number(
                riskPercentage
              ).toFixed(2)}%`}
              unit="DETECTED"
            />

          </div>

          <section
            id="risk-map-section"
            className="map-section"
          >

            <div className="section-heading">

              <div>

                <span>
                  GEOSPATIAL AI
                </span>

                <h2>
                  Satellite Risk Map
                </h2>

              </div>

              <div className="prediction-count">

                {loading
                  ? "ANALYZING HYPERSPECTRAL DATA..."
                  : `${predictions.length} ZONES ANALYZED`}

              </div>

            </div>

            <FarmMap
              predictions={predictions}
            />

          </section>

          <div className="dashboard-grid">

            <RiskSummary
              highRisk={highRisk}
              moderateRisk={moderateRisk}
              healthy={healthy}
            />

            <SpectralAnalysis
              id="spectral-section"
            />

          </div>

          <Pipeline />

          <Analytics
            id="analytics-section"
          />

          <HistoryTimeline
            id="history-section"
          />

        </section>

      </main>

    </div>
  );
}