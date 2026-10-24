import { PolygonLayer } from "@deck.gl/layers";

const MAP_CENTER_LONGITUDE = 83.2185;
const MAP_CENTER_LATITUDE = 17.6868;

// Small geographic cell size for the 29 x 29 prediction grid.
// 841 predictions = 29 x 29 when stride = 2.
const CELL_SIZE = 0.00025;

function getRiskValue(prediction) {
  const possibleValues = [
    prediction?.risk,
    prediction?.probability,
    prediction?.fused_risk,
    prediction?.fused_probability,
    prediction?.score,
  ];

  for (let i = 0; i < possibleValues.length; i++) {
    const value = Number(possibleValues[i]);

    if (Number.isFinite(value)) {
      return value;
    }
  }

  return 0;
}

function getRow(prediction, index) {
  const possibleRows = [
    prediction?.row,
    prediction?.r,
    prediction?.y,
  ];

  for (let i = 0; i < possibleRows.length; i++) {
    const value = Number(possibleRows[i]);

    if (Number.isFinite(value)) {
      return value;
    }
  }

  return Math.floor(index / 29);
}

function getColumn(prediction, index) {
  const possibleColumns = [
    prediction?.col,
    prediction?.column,
    prediction?.c,
    prediction?.x,
  ];

  for (let i = 0; i < possibleColumns.length; i++) {
    const value = Number(possibleColumns[i]);

    if (Number.isFinite(value)) {
      return value;
    }
  }

  return index % 29;
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
    MAP_CENTER_LONGITUDE +
    (column - 14) * CELL_SIZE;

  const latitude =
    MAP_CENTER_LATITUDE -
    (row - 14) * CELL_SIZE;

  const half = CELL_SIZE / 2;

  return [
    [
      longitude - half,
      latitude - half,
    ],
    [
      longitude + half,
      latitude - half,
    ],
    [
      longitude + half,
      latitude + half,
    ],
    [
      longitude - half,
      latitude + half,
    ],
  ];
}

export default function Heatmap({
  predictions = [],
}) {
  if (
    !predictions ||
    predictions.length === 0
  ) {
    return null;
  }

  const mapData = [];

  for (let i = 0; i < predictions.length; i++) {
    const prediction = predictions[i];

    const risk = getRiskValue(prediction);

    const polygon = createPolygon(
      prediction,
      i
    );

    mapData.push({
      ...prediction,
      risk,
      polygon,
    });
  }

  return new PolygonLayer({
    id: "terra-spectra-risk-grid",

    data: mapData,

    pickable: true,

    filled: true,

    stroked: true,

    getPolygon: (d) => d.polygon,

    getFillColor: (d) => {
      const risk = Number(d.risk);

      if (risk >= 0.7) {
        return [
          239,
          68,
          68,
          210,
        ];
      }

      if (risk >= 0.4) {
        return [
          234,
          179,
          8,
          200,
        ];
      }

      return [
        34,
        197,
        94,
        180,
      ];
    },

    getLineColor: [
      255,
      255,
      255,
      120,
    ],

    getLineWidth: 1,

    lineWidthMinPixels: 1,

    opacity: 0.9,

    autoHighlight: true,

    highlightColor: [
      255,
      255,
      255,
      100,
    ],

    parameters: {
      depthTest: false,
    },

    updateTriggers: {
      getFillColor: [
        predictions,
      ],
      getPolygon: [
        predictions,
      ],
    },
  });
}