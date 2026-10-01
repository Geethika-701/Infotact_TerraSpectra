import React from "react";
import { ScatterplotLayer } from "@deck.gl/layers";

const HEALTHY_COLOR = [34, 197, 94, 190];
const STRESSED_COLOR = [239, 68, 68, 210];

function getColor(status, probability) {
  if (status === "chemically_stressed") {
    return [
      239,
      68,
      68,
      Math.max(120, Math.min(230, probability * 255)),
    ];
  }

  return [
    34,
    197,
    94,
    Math.max(120, Math.min(230, probability * 255)),
  ];
}

export default function Heatmap({ predictions = [] }) {
  const layer = new ScatterplotLayer({
    id: "terraspectra-week2-heatmap",

    data: predictions,

    getPosition: (d) => [d.longitude, d.latitude],

    getFillColor: (d) =>
      getColor(
        d.status,
        d.probability ?? d.confidence ?? 0.8
      ),

    getRadius: 35,

    radiusMinPixels: 5,
    radiusMaxPixels: 25,

    pickable: true,

    stroked: true,

    getLineColor: [255, 255, 255, 180],

    lineWidthMinPixels: 1,

    onHover: ({ object, x, y }) => {
      if (object) {
        console.log("Prediction:", object);
      }
    },
  });

  return layer;
}