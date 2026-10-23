import React, {
  useMemo
} from "react";

import Map from "react-map-gl/mapbox";
import { DeckGL } from "@deck.gl/react";
import Heatmap from "./Heatmap";

import "mapbox-gl/dist/mapbox-gl.css";

const MAPBOX_TOKEN =
  import.meta.env.VITE_MAPBOX_TOKEN;

const INITIAL_VIEW_STATE = {
  longitude: 83.2185,
  latitude: 17.6868,
  zoom: 14,
  pitch: 35,
  bearing: 0
};

export default function FarmMap({
  predictions = []
}) {
  const layers = useMemo(() => {

    if (
      !predictions ||
      predictions.length === 0
    ) {
      return [];
    }

    return [
      Heatmap({
        predictions
      })
    ];

  }, [predictions]);

  return (
    <div className="farm-map-container">

      <DeckGL
        initialViewState={
          INITIAL_VIEW_STATE
        }
        controller={true}
        layers={layers}
        style={{
          position: "absolute",
          inset: 0
        }}
      >

        <Map
          mapboxAccessToken={
            MAPBOX_TOKEN
          }
          mapStyle={
            "mapbox://styles/mapbox/satellite-streets-v12"
          }
          style={{
            width: "100%",
            height: "100%"
          }}
        />

      </DeckGL>

      <div className="map-overlay">

        <div className="map-live">
          <span></span>
          LIVE RISK MONITORING
        </div>

        <div className="map-bands">
          224 BANDS
        </div>

      </div>

      <div className="map-legend">

        <strong>
          RISK LEVEL
        </strong>

        <div>
          <span className="legend-dot healthy"></span>
          Healthy
        </div>

        <div>
          <span className="legend-dot moderate"></span>
          Moderate
        </div>

        <div>
          <span className="legend-dot high"></span>
          High Risk
        </div>

      </div>

      <div className="map-coordinates">
        17.6868° N · 83.2185° E
      </div>

    </div>
  );
}