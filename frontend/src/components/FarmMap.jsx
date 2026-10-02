import React, { useMemo } from "react";
import Map from "react-map-gl/mapbox";
import { DeckGL } from "@deck.gl/react";

import Heatmap from "./Heatmap";

import "mapbox-gl/dist/mapbox-gl.css";

const MAPBOX_TOKEN =
  import.meta.env.VITE_MAPBOX_TOKEN;

const INITIAL_VIEW_STATE = {
  longitude: 83.2185,
  latitude: 17.6868,
  zoom: 15,
  pitch: 45,
  bearing: 0,
};

export default function FarmMap({
  predictions,
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
        predictions,
      }),
    ];
  }, [predictions]);

  return (
    <div className="farm-map-container">

      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
      >

        <Map
          mapboxAccessToken={MAPBOX_TOKEN}
          mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
        />

      </DeckGL>

    </div>
  );
}