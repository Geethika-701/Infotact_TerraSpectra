
import React, { useEffect, useState } from "react";

import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import StatCard from "./components/StatCard";
import FarmMap from "./components/FarmMap";
import RiskSummary from "./components/RiskSummary";
import SpectralAnalysis from "./components/SpectralAnalysis";
import Pipeline from "./components/Pipeline";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

export default function App() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);

  useEffect(() => {
    async function loadPredictions() {
      try {
        const response = await fetch(`${API_URL}/predictions`);

        if (!response.ok) {
          throw new Error("Prediction API failed");
        }

        const data = await response.json();

        setPredictions(data.predictions || []);
        setApiOnline(true);
      } catch (error) {
        console.error("Backend connection error:", error);
        setApiOnline(false);
      } finally {
        setLoading(false);
      }
    }

    loadPredictions();
  }, []);

  // -----------------------------
  // RISK CALCULATIONS
  // -----------------------------

  const highRisk = predictions.filter(
    (item) => Number(item.risk) >= 0.7
  ).length;

  const moderateRisk = predictions.filter(
    (item) =>
      Number(item.risk) >= 0.4 &&
      Number(item.risk) < 0.7
  ).length;

  const healthy = predictions.filter(
    (item) => Number(item.risk) < 0.4
  ).length;

  const total = predictions.length;

  const riskPercentage =
    total > 0
      ? (((highRisk + moderateRisk) / total) * 100).toFixed(1)
      : "0.0";

  return (
    <div className="app">

      {/* SIDEBAR */}
      <Sidebar />

      <main className="main-content">

        {/* HEADER */}
        <Header />

        <section className="dashboard">

          {/* PAGE TITLE */}
          <div className="page-heading">

            <div>
              <span className="eyebrow">
                PRECISION AGRICULTURE · AI MONITORING
              </span>

              <h1>Farm Overview</h1>

              <p>
                Hyperspectral crop stress detection using
                spatial-spectral deep learning.
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

          {/* KPI CARDS */}
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
              value="3D-CNN"
              unit="ACTIVE"
            />

            <StatCard
              label="RISK ZONES"
              value={`${riskPercentage}%`}
              unit="DETECTED"
            />

          </div>

          {/* MAP */}
          <section className="map-section">

            <div className="section-heading">

              <div>
                <span>WEEK 2 · GEOSPATIAL AI</span>
                <h2>Satellite Risk Map</h2>
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

          {/* ANALYTICS */}
          <div className="dashboard-grid">

            <RiskSummary
              highRisk={highRisk}
              moderateRisk={moderateRisk}
              healthy={healthy}
            />

            <SpectralAnalysis />

          </div>

          {/* PIPELINE */}
          <Pipeline />

        </section>

      </main>

    </div>
  );
}
