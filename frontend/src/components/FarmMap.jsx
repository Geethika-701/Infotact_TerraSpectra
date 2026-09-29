import { useState } from "react";
import Map from "react-map-gl/mapbox";
import { DeckGL } from "@deck.gl/react";
import { GeoJsonLayer } from "@deck.gl/layers";

import "mapbox-gl/dist/mapbox-gl.css";

const INITIAL_VIEW_STATE = {
  longitude: 83.30,
  latitude: 17.72,
  zoom: 14,
  pitch: 45,
  bearing: 0
};

// Mock farm boundary
const FARM_BOUNDARY = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: {
        name: "TerraSpectra Farm"
      },
      geometry: {
        type: "Polygon",
        coordinates: [[
          [83.2980, 17.7180],
          [83.3020, 17.7180],
          [83.3020, 17.7220],
          [83.2980, 17.7220],
          [83.2980, 17.7180]
        ]]
      }
    }
  ]
};

function FarmMap() {

  const [viewState, setViewState] = useState(
    INITIAL_VIEW_STATE
  );

  const layers = [
    new GeoJsonLayer({
      id: "farm-boundary",
      data: FARM_BOUNDARY,

      filled: true,

      getFillColor: [0, 128, 0, 80],

      getLineColor: [0, 80, 0, 255],

      getLineWidth: 4,

      lineWidthMinPixels: 2,

      pickable: true,

      onClick: ({ object }) => {
        if (object) {
          alert(
            object.properties.name
          );
        }
      }
    })
  ];

  return (
    <DeckGL
      viewState={viewState}
      onViewStateChange={({ viewState }) =>
        setViewState(viewState)
      }
      controller={true}
      layers={layers}
    >
      <Map
        mapboxAccessToken={
          import.meta.env.VITE_MAPBOX_TOKEN
        }

        mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
      />
    </DeckGL>
  );
}

export default FarmMap;