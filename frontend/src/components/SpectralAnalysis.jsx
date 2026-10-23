import React from "react";

export default function SpectralAnalysis({
  id
}) {
  const components = [
    {
      name: "PC1",
      value: 41.8
    },
    {
      name: "PC2",
      value: 27.4
    },
    {
      name: "PC3",
      value: 18.2
    }
  ];

  return (
    <div
      id={id}
      className="spectral-card"
    >

      <div className="spectral-top">

        <div className="spectral-description">

          <strong>
            Hyperspectral Dimensionality Reduction
          </strong>

          <p>
            224 spectral bands reduced to 3
            principal components for efficient
            spatial-spectral analysis.
          </p>

        </div>

        <div className="spectral-reduction">

          <strong>
            224
          </strong>

          <span>
            →
          </span>

          <strong>
            3
          </strong>

        </div>

      </div>

      <div className="pca-bars">

        {components.map(
          (component) => (

            <div
              className="pca-row"
              key={component.name}
            >

              <div className="pca-label">

                <strong>
                  {component.name}
                </strong>

                <span>
                  {component.value}%
                </span>

              </div>

              <div className="pca-track">

                <div
                  className={`pca-progress ${component.name.toLowerCase()}`}
                  style={{
                    width: `${component.value}%`
                  }}
                />

              </div>

            </div>

          )
        )}

      </div>

      <div className="spectral-bottom">

        <div>

          <span>
            INPUT
          </span>

          <strong>
            224 BANDS
          </strong>

        </div>

        <div>

          <span>
            OUTPUT
          </span>

          <strong>
            3 COMPONENTS
          </strong>

        </div>

        <div>

          <span>
            METHOD
          </span>

          <strong>
            PCA
          </strong>

        </div>

      </div>

    </div>
  );
}