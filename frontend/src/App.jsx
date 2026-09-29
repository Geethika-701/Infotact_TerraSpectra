
import React, { useEffect, useState } from "react";
import Map from "react-map-gl/mapbox";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";

import "mapbox-gl/dist/mapbox-gl.css";

// Map starting location
const INITIAL_VIEW_STATE = {
  longitude: 83.2185,
  latitude: 17.6868,
  zoom: 14,
  pitch: 0,
  bearing: 0,
};

function App() {
  const [pcaData, setPcaData] = useState(null);
  const [error, setError] = useState(null);

  // Load PCA data
  useEffect(() => {
    fetch("/pca_map.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load PCA map data");
        }

        return response.json();
      })
      .then((data) => {
        setPcaData(data);
      })
      .catch((err) => {
        console.error("PCA data error:", err);
        setError(err.message);
      });
  }, []);

  // PCA points
  const points = pcaData ? pcaData.points : [];

  // Deck.gl layer
  const layers = [
    new ScatterplotLayer({
      id: "pca-points",

      data: points,

      getPosition: (point) => [
        point.longitude,
        point.latitude,
      ],

      getRadius: 4,

      radiusMinPixels: 2,
      radiusMaxPixels: 8,

      // PCA value converted to color
      getFillColor: (point) => {
        const value = point.value;

        return [
          Math.round(255 * value),
          Math.round(255 * (1 - value)),
          0,
        ];
      },

      pickable: true,

      getTooltip: (info) => {
        if (!info.object) {
          return null;
        }

        return {
          text: `PCA Value: ${info.object.value.toFixed(4)}`,
        };
      },
    }),
  ];

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        position: "relative",
      }}
    >
      {/* Mapbox + Deck.gl */}
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
      >
        <Map
          mapboxAccessToken={import.meta.env.VITE_MAPBOX_TOKEN}
          mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
        />
      </DeckGL>

      {/* Information Panel */}
      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          background: "white",
          padding: "15px 20px",
          borderRadius: "8px",
          boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
          minWidth: "240px",
          zIndex: 10,
        }}
      >
        <h2
          style={{
            margin: "0 0 5px 0",
          }}
        >
          TerraSpectra
        </h2>

        <p
          style={{
            margin: "0 0 10px 0",
          }}
        >
          Hyperspectral Crop Disease Forecasting
        </p>

        <p
          style={{
            margin: "0 0 5px 0",
            fontWeight: "bold",
          }}
        >
          PCA Hyperspectral Visualization
        </p>

        {pcaData && (
          <>
            <p style={{ margin: "5px 0" }}>
              Pixels: {pcaData.points.length}
            </p>

            <p style={{ margin: "5px 0" }}>
              Components: {pcaData.components}
            </p>

            <p style={{ margin: "5px 0" }}>
              Component Used: PCA {pcaData.component_used}
            </p>
          </>
        )}

        {!pcaData && !error && (
          <p style={{ margin: "8px 0 0 0" }}>
            Loading PCA data...
          </p>
        )}

        {error && (
          <p
            style={{
              margin: "8px 0 0 0",
              color: "red",
            }}
          >
            Error: {error}
          </p>
        )}
      </div>
    </div>
  );
}

export default App;

