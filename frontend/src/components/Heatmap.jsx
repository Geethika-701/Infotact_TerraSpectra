
import { PolygonLayer } from "@deck.gl/layers";

const MAP_CENTER_LONGITUDE = 83.2185;
const MAP_CENTER_LATITUDE = 17.6868;

const CELL_SIZE = 0.00025;

function getRiskValue(prediction) {
  const possibleValues = [
    prediction?.risk,
    prediction?.probability,
    prediction?.fused_risk,
    prediction?.fused_probability,
    prediction?.score,
  ];

  for (const value of possibleValues) {
    if (
      value !== undefined &&
      value !== null &&
      value !== ""
    ) {
      const number = Number(value);

      if (Number.isFinite(number)) {
        return number;
      }
    }
  }

  return 0;
}

function getPercentile(values, percentile) {
  const sorted = [...values].sort((a, b) => a - b);

  if (sorted.length === 0) return 0;

  const index = (percentile / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (lower === upper) {
    return sorted[lower];
  }

  return (
    sorted[lower] +
    (sorted[upper] - sorted[lower]) * (index - lower)
  );
}

function getRow(prediction, index) {
  const value = Number(
    prediction?.row ?? prediction?.r ?? prediction?.y
  );

  return Number.isFinite(value)
    ? value
    : Math.floor(index / 29);
}

function getColumn(prediction, index) {
  const value = Number(
    prediction?.col ??
    prediction?.column ??
    prediction?.c ??
    prediction?.x
  );

  return Number.isFinite(value)
    ? value
    : index % 29;
}

function createPolygon(prediction, index) {
  if (
    prediction &&
    Array.isArray(prediction.polygon) &&
    prediction.polygon.length >= 3
  ) {
    return prediction.polygon;
  }

  const row = getRow(prediction, index);
  const column = getColumn(prediction, index);

  const longitude =
    MAP_CENTER_LONGITUDE + (column - 14) * CELL_SIZE;

  const latitude =
    MAP_CENTER_LATITUDE - (row - 14) * CELL_SIZE;

  const half = CELL_SIZE / 2;

  return [
    [longitude - half, latitude - half],
    [longitude + half, latitude - half],
    [longitude + half, latitude + half],
    [longitude - half, latitude + half],
  ];
}

export default function Heatmap({
  predictions = [],
}) {
  if (!predictions || predictions.length === 0) {
    return null;
  }

  const risks = predictions.map(getRiskValue);

  // Original model thresholds
  const hasModerate = risks.some(
    (risk) => risk >= 0.40 && risk < 0.70
  );

  // Match the analytics fallback if no moderate predictions exist
  const moderateThreshold = hasModerate
    ? 0.40
    : getPercentile(risks, 60);

  const mapData = predictions.map((prediction, index) => ({
    ...prediction,
    risk: risks[index],
    polygon: createPolygon(prediction, index),
  }));

  return new PolygonLayer({
    id: "terra-spectra-risk-grid",

    data: mapData,

    pickable: true,
    filled: true,
    stroked: true,

    getPolygon: (d) => d.polygon,

    getFillColor: (d) => {
      const risk = d.risk;

      // High Risk - Red
      if (risk >= 0.70) {
        return [239, 68, 68, 210];
      }

      // Moderate Risk - Orange
      if (risk >= moderateThreshold) {
        return [249, 115, 22, 210];
      }

      // Healthy - Green
      return [34, 197, 94, 180];
    },

    getLineColor: [255, 255, 255, 120],

    getLineWidth: 1,
    lineWidthMinPixels: 1,

    opacity: 0.9,

    autoHighlight: true,

    highlightColor: [255, 255, 255, 100],

    parameters: {
      depthTest: false,
    },

    updateTriggers: {
      getFillColor: [predictions, moderateThreshold],
      getPolygon: [predictions],
    },
  });
}
