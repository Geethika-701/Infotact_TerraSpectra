import React from "react";
import { ScatterplotLayer } from "@deck.gl/layers";

const HEALTHY_COLOR = [34, 197, 94, 190];
const STRESSED_COLOR = [239, 68, 68, 210];

function getColor(status, probability) {
  const alpha = Math.max(
    120,
    Math.min(230, probability * 255)
  );

  if (status === "chemically_stressed") {
    return [
      STRESSED_COLOR[0],
      STRESSED_COLOR[1],
      STRESSED_COLOR[2],
      alpha,
    ];
  }

  return [
    HEALTHY_COLOR[0],
    HEALTHY_COLOR[1],
    HEALTHY_COLOR[2],
    alpha,
  ];
}

export default function Heatmap({
  predictions = [],
}) {
  return new ScatterplotLayer({
    id: "terraspectra-week2-heatmap",

    data: predictions,

    getPosition: (d) => [
      d.longitude,
      d.latitude,
    ],

    getFillColor: (d) =>
      getColor(
        d.status,
        d.probability ?? 0.8
      ),

    getRadius: 18,

    radiusMinPixels: 4,
    radiusMaxPixels: 20,

    pickable: true,

    stroked: true,

    getLineColor: [
      255,
      255,
      255,
      180,
    ],

    lineWidthMinPixels: 1,

    getTooltip: (info) => {
      if (!info.object) {
        return null;
      }

      return {
        text:
          `Classification: ${info.object.status}\n` +
          `Probability: ${(info.object.probability * 100).toFixed(2)}%`,
      };
    },
  });
}