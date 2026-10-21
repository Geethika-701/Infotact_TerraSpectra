import { PolygonLayer } from "@deck.gl/layers";

export default function Heatmap({
  predictions = [],
}) {

  if (
    !predictions ||
    predictions.length === 0
  ) {
    return null;
  }

  return new PolygonLayer({

    id: "terra-spectra-risk-grid",

    data: predictions,

    pickable: true,

    filled: true,

    stroked: true,

    getPolygon: (d) => d.polygon,

    getFillColor: (d) => {

      const risk =
        Number(
          d.risk ??
          d.probability ??
          0
        );

      if (risk >= 0.7) {
        return [
          239,
          68,
          68,
          190,
        ];
      }

      if (risk >= 0.4) {
        return [
          234,
          179,
          8,
          175,
        ];
      }

      return [
        34,
        197,
        94,
        150,
      ];
    },

    getLineColor: [
      255,
      255,
      255,
      100,
    ],

    getLineWidth: 1,

    lineWidthMinPixels: 1,

    opacity: 0.85,

    autoHighlight: true,

    highlightColor: [
      255,
      255,
      255,
      80,
    ],

    parameters: {
      depthTest: false,
    },

  });
}