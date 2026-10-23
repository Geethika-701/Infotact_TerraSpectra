import React from "react";

export default function Pipeline() {

  const steps = [
    {
      number: "01",
      title: "HYPERSPECTRAL INGESTION",
      description:
        "224-band satellite data cube",
      status: "COMPLETE"
    },
    {
      number: "02",
      title: "PCA REDUCTION",
      description:
        "224 bands → 3 components",
      status: "COMPLETE"
    },
    {
      number: "03",
      title: "AI CLASSIFICATION",
      description:
        "3D-CNN + Vision Transformer",
      status: "ACTIVE"
    },
    {
      number: "04",
      title: "RISK & ANOMALY MAPPING",
      description:
        "Spatial-spectral crop stress analysis",
      status: "ACTIVE"
    }
  ];

  return (
    <section className="pipeline-section">

      <div className="section-heading">

        <div>

          <span>
            AI PROCESSING
          </span>

          <h2>
            Analysis Pipeline
          </h2>

        </div>

        <div className="pipeline-week">
          AI FUSION ACTIVE
        </div>

      </div>

      <div className="pipeline">

        {steps.map(
          (step, index) => (

            <React.Fragment
              key={step.number}
            >

              <div className="pipeline-step">

                <div className="pipeline-number">
                  {step.number}
                </div>

                <div>

                  <strong>
                    {step.title}
                  </strong>

                  <p>
                    {step.description}
                  </p>

                  <span
                    className={
                      step.status ===
                      "ACTIVE"
                        ? "pipeline-status active"
                        : "pipeline-status complete"
                    }
                  >
                    ● {step.status}
                  </span>

                </div>

              </div>

              {index <
                steps.length - 1 && (
                <div className="pipeline-arrow">
                  →
                </div>
              )}

            </React.Fragment>

          )
        )}

      </div>

    </section>
  );
}