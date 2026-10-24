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

/*
  Synthetic TerraSpectra farm reference.

  These coordinates match the coordinate system
  already used by the dashboard.

  IMPORTANT:
  longitude = X
  latitude  = Y
*/

const FARM_CENTER_LNG = 83.2185;
const FARM_CENTER_LAT = 17.6868;

/*
  841 prediction zones = 29 x 29 grid.

  Each prediction cell is displayed as a small
  geographic polygon around the farm center.
*/

const GRID_SIZE = 29;

/*
  Geographic size of one displayed prediction cell.

  This is intentionally small so the prediction
  grid stays around the farm area.
*/

const CELL_SIZE = 0.00022;

/*
  Convert backend predictions into Deck.gl polygons.
*/
function prepareMapPredictions(rawPredictions) {

  if (
    !Array.isArray(rawPredictions) ||
    rawPredictions.length === 0
  ) {
    return [];
  }

  const prepared = [];

  for (
    let index = 0;
    index < rawPredictions.length;
    index++
  ) {

    const item =
      rawPredictions[index] || {};

    /*
      Try to use row/column information if the
      backend already provides it.
    */

    let row;

    let col;

    if (
      item.row !== undefined &&
      item.row !== null
    ) {
      row = Number(item.row);
    } else if (
      item.y !== undefined &&
      item.y !== null
    ) {
      row = Number(item.y);
    } else {
      row =
        Math.floor(
          index / GRID_SIZE
        );
    }

    if (
      item.col !== undefined &&
      item.col !== null
    ) {
      col = Number(item.col);
    } else if (
      item.x !== undefined &&
      item.x !== null
    ) {
      col = Number(item.x);
    } else {
      col =
        index % GRID_SIZE;
    }

    /*
      Keep row/column inside the expected
      29 x 29 grid.
    */

    if (
      !Number.isFinite(row)
    ) {
      row =
        Math.floor(
          index / GRID_SIZE
        );
    }

    if (
      !Number.isFinite(col)
    ) {
      col =
        index % GRID_SIZE;
    }

    /*
      Risk can come from several backend field
      names depending on the prediction response.
    */

    let risk = Number(
      item.risk ??
      item.probability ??
      item.fused_risk ??
      item.fused_probability ??
      item.stressed_probability ??
      0
    );

    if (!Number.isFinite(risk)) {
      risk = 0;
    }

    /*
      If the backend ever returns a percentage
      such as 75 instead of 0.75, normalize it.
    */

    if (risk > 1) {
      risk = risk / 100;
    }

    /*
      Calculate the geographic position.

      Center the 29 x 29 grid around the farm center.
    */

    const gridOffset =
      (GRID_SIZE - 1) / 2;

    const centerLng =
      FARM_CENTER_LNG +
      (col - gridOffset) *
      CELL_SIZE;

    const centerLat =
      FARM_CENTER_LAT +
      (gridOffset - row) *
      CELL_SIZE;

    /*
      Half cell size.
    */

    const half =
      CELL_SIZE / 2;

    /*
      Deck.gl polygon coordinate order:

      [longitude, latitude]
    */

    const polygon = [
      [
        centerLng - half,
        centerLat - half
      ],
      [
        centerLng + half,
        centerLat - half
      ],
      [
        centerLng + half,
        centerLat + half
      ],
      [
        centerLng - half,
        centerLat + half
      ]
    ];

    prepared.push({
      ...item,

      row,
      col,

      risk,

      probability: risk,

      polygon
    });
  }

  return prepared;
}

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

        /*
          Load fused CNN + ViT predictions.
        */

        const predictionsResponse =
          await fetch(
            `${API_URL}/predictions/fused`
          );

        if (
          !predictionsResponse.ok
        ) {
          throw new Error(
            "Fused prediction API failed"
          );
        }

        const predictionsData =
          await predictionsResponse.json();

        /*
          Get raw backend predictions.
        */

        const rawPredictions =
          predictionsData.predictions || [];

        /*
          IMPORTANT:

          Convert backend prediction records
          into map-ready records containing
          polygon coordinates.
        */

        const mapPredictions =
          prepareMapPredictions(
            rawPredictions
          );

        console.log(
          "TerraSpectra raw predictions:",
          rawPredictions
        );

        console.log(
          "TerraSpectra map predictions:",
          mapPredictions
        );

        if (
          mapPredictions.length > 0
        ) {

          console.log(
            "First map polygon:",
            mapPredictions[0].polygon
          );

          console.log(
            "First map risk:",
            mapPredictions[0].risk
          );
        }

        setPredictions(
          mapPredictions
        );

        setApiOnline(true);

        /*
          Load analytics separately.
        */

        try {

          const analyticsResponse =
            await fetch(
              `${API_URL}/analytics`
            );

          if (
            analyticsResponse.ok
          ) {

            const data =
              await analyticsResponse.json();

            setAnalyticsData(
              data
            );
          }

        } catch (
          analyticsError
        ) {

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