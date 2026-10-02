import React, { useEffect, useState } from "react";

import FarmMap from "./components/FarmMap";

function App() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/predictions.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load predictions.json");
        }

        return response.json();
      })
      .then((data) => {
        const rawPredictions = data.predictions || [];

        // Normalize prediction field names
        // Supports either:
        // status + probability
        // OR
        // class + confidence
        const normalizedPredictions = rawPredictions.map((item) => ({
          ...item,

          status:
            item.status ||
            item.class ||
            "healthy",

          probability:
            item.probability ??
            item.confidence ??
            0.8,
        }));

        setPredictions(normalizedPredictions);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Prediction data error:", err);

        setError(err.message);
        setLoading(false);
      });
  }, []);

  const healthyCount = predictions.filter(
    (item) => item.status === "healthy"
  ).length;

  const stressedCount = predictions.filter(
    (item) => item.status === "chemically_stressed"
  ).length;

  return (
    <div className="app">

      {/* HEADER */}

      <header className="header">

        <div>
          <h1>TerraSpectra</h1>

          <p>
            Hyperspectral Crop Disease Forecasting
          </p>
        </div>

        <div className="week-badge">
          Week 2 — 3D-CNN Classification
        </div>

      </header>


      {/* MAIN CONTENT */}

      <main className="main-content">

        {/* PROJECT INFORMATION */}

        <section className="project-info">

          <div>
            <h2>
              Healthy vs Chemically Stressed
              Pixel Classification
            </h2>

            <p>
              Hybrid 3D-CNN prediction results
              visualized on the farm map
            </p>
          </div>

          <div className="status">

            <span className="status-dot"></span>

            {loading
              ? "Loading predictions..."
              : error
              ? "Prediction data error"
              : "Prediction data loaded"}

          </div>

        </section>


        {/* ERROR */}

        {error && (
          <div
            style={{
              padding: "15px",
              marginBottom: "20px",
              background: "#fee2e2",
              color: "#991b1b",
              borderRadius: "8px",
              border: "1px solid #fecaca",
            }}
          >
            Error: {error}
          </div>
        )}


        {/* STATISTICS */}

        {!loading && !error && (
          <section className="stats">

            <div className="stat-card">

              <span className="stat-label">
                Total Classified Pixels
              </span>

              <strong>
                {predictions.length}
              </strong>

            </div>


            <div className="stat-card healthy">

              <span className="stat-label">
                Healthy Pixels
              </span>

              <strong>
                {healthyCount}
              </strong>

            </div>


            <div className="stat-card stressed">

              <span className="stat-label">
                Chemically Stressed Pixels
              </span>

              <strong>
                {stressedCount}
              </strong>

            </div>

          </section>
        )}


        {/* MAP */}

        <section className="map-section">

          <div className="map-header">

            <div>

              <h2>
                Hyperspectral Classification Map
              </h2>

              <p>
                Hybrid 3D-CNN pixel classification
                overlay
              </p>

            </div>


            <div className="legend">

              <div className="legend-item">

                <span className="legend-color healthy-color"></span>

                Healthy

              </div>


              <div className="legend-item">

                <span className="legend-color stressed-color"></span>

                Chemically Stressed

              </div>

            </div>

          </div>


          <div className="map-wrapper">

            {loading ? (
              <div className="loading">

                <div className="spinner"></div>

                <p>
                  Loading classification results...
                </p>

              </div>
            ) : error ? (
              <div className="loading">

                <p>
                  Unable to load prediction data.
                </p>

              </div>
            ) : (
              <FarmMap
                predictions={predictions}
              />
            )}

          </div>

        </section>


        {/* MODEL INFORMATION */}

        <section className="model-info">

          <div className="info-card">

            <h3>
              Week 2 Model
            </h3>

            <p>
              Hybrid 3D-CNN implemented using
              PyTorch for hyperspectral pixel
              classification into healthy and
              chemically stressed categories.
            </p>

          </div>


          <div className="info-card">

            <h3>
              Processing Pipeline
            </h3>

            <div className="pipeline">

              <span>
                Hyperspectral Data
              </span>

              <span className="arrow">
                →
              </span>

              <span>
                PCA
              </span>

              <span className="arrow">
                →
              </span>

              <span>
                Hybrid 3D-CNN
              </span>

              <span className="arrow">
                →
              </span>

              <span>
                Predictions
              </span>

              <span className="arrow">
                →
              </span>

              <span>
                Map
              </span>

            </div>

          </div>

        </section>

      </main>


      {/* FOOTER */}

      <footer>
        TerraSpectra — Hyperspectral Crop Disease
        Forecasting
      </footer>

    </div>
  );
}

export default App;