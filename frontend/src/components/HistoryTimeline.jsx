import React, {
  useEffect,
  useState
} from "react";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:5000";

export default function HistoryTimeline({
  id
}) {
  const [history, setHistory] =
    useState([]);

  const [selectedIndex, setSelectedIndex] =
    useState(0);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState(false);

  useEffect(() => {
    async function loadHistory() {
      try {
        setLoading(true);
        setError(false);

        const response =
          await fetch(
            `${API_URL}/history`
          );

        if (!response.ok) {
          throw new Error(
            "History API failed"
          );
        }

        const data =
          await response.json();

        const snapshots =
          data.history || [];

        setHistory(snapshots);

        if (snapshots.length > 0) {
          setSelectedIndex(
            snapshots.length - 1
          );
        }
      } catch (err) {
        console.error(
          "History API error:",
          err
        );

        setError(true);
      } finally {
        setLoading(false);
      }
    }

    loadHistory();
  }, []);

  const selectedScan =
    history.length > 0
      ? history[selectedIndex]
      : null;

  function formatDate(value) {
    if (!value) {
      return "No timestamp";
    }

    const date =
      new Date(value);

    if (Number.isNaN(date.getTime())) {
      return value;
    }

    return date.toLocaleString(
      "en-IN",
      {
        day: "2-digit",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit"
      }
    );
  }

  function getHealthStatus(risk) {
    const value =
      Number(risk || 0);

    if (value >= 25) {
      return {
        label: "ELEVATED STRESS",
        className: "elevated"
      };
    }

    if (value >= 15) {
      return {
        label: "WATCH RANGE",
        className: "watch"
      };
    }

    return {
      label: "NORMAL RANGE",
      className: "normal"
    };
  }

  if (loading) {
    return (
      <section
        id={id}
        className="history-section"
      >
        <div className="history-card">

          <div className="history-heading">

            <div>
              <span>
                TEMPORAL MONITORING
              </span>

              <h2>
                Historical Spectral Health
              </h2>

              <p>
                Loading historical crop
                health observations...
              </p>
            </div>

          </div>

          <div className="history-loading">
            ANALYZING HISTORICAL SCANS...
          </div>

        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section
        id={id}
        className="history-section"
      >
        <div className="history-card">

          <div className="history-heading">

            <div>
              <span>
                TEMPORAL MONITORING
              </span>

              <h2>
                Historical Spectral Health
              </h2>

              <p>
                Historical scan information
                is currently unavailable.
              </p>
            </div>

          </div>

          <div className="history-empty">
            <strong>
              HISTORY SERVICE UNAVAILABLE
            </strong>

            <span>
              Connect the FastAPI backend to
              load previous spectral scans.
            </span>
          </div>

        </div>
      </section>
    );
  }

  if (history.length === 0) {
    return (
      <section
        id={id}
        className="history-section"
      >
        <div className="history-card">

          <div className="history-heading">

            <div>
              <span>
                TEMPORAL MONITORING
              </span>

              <h2>
                Historical Spectral Health
              </h2>

              <p>
                Track crop spectral health
                across successive AI scans.
              </p>
            </div>

            <div className="history-live-badge">
              ● LIVE MONITORING
            </div>

          </div>

          <div className="history-empty">

            <strong>
              NO HISTORICAL SCANS YET
            </strong>

            <span>
              Historical snapshots will appear
              here after the AI analysis pipeline
              records scan results.
            </span>

          </div>

        </div>
      </section>
    );
  }

  const health =
    getHealthStatus(
      selectedScan?.risk_percentage
    );

  return (
    <section
      id={id}
      className="history-section"
    >

      <div className="history-card">

        <div className="history-heading">

          <div>

            <span>
              TEMPORAL MONITORING
            </span>

            <h2>
              Historical Spectral Health
            </h2>

            <p>
              Explore the progression of
              crop spectral health across
              previous AI scans.
            </p>

          </div>

          <div className="history-live-badge">
            ● HISTORICAL ANALYSIS
          </div>

        </div>

        <div className="history-selected">

          <div>

            <span>
              SELECTED SCAN
            </span>

            <strong>
              {selectedScan?.label ||
                formatDate(
                  selectedScan?.timestamp
                )}
            </strong>

          </div>

          <div
            className={`history-status ${health.className}`}
          >
            ● {health.label}
          </div>

        </div>

        <div className="history-slider-area">

          <div className="history-slider-labels">

            <span>
              {formatDate(
                history[0]?.timestamp
              )}
            </span>

            <strong>
              {selectedIndex + 1}
              {" / "}
              {history.length}
            </strong>

            <span>
              {formatDate(
                history[
                  history.length - 1
                ]?.timestamp
              )}
            </span>

          </div>

          <input
            className="history-slider"
            type="range"
            min="0"
            max={history.length - 1}
            step="1"
            value={selectedIndex}
            onChange={(event) =>
              setSelectedIndex(
                Number(
                  event.target.value
                )
              )
            }
          />

          <div className="history-timeline-dots">

            {history.map(
              (item, index) => (
                <button
                  key={
                    item.timestamp ||
                    index
                  }
                  className={
                    index === selectedIndex
                      ? "timeline-dot active"
                      : "timeline-dot"
                  }
                  onClick={() =>
                    setSelectedIndex(
                      index
                    )
                  }
                  aria-label={
                    `Select scan ${index + 1}`
                  }
                />
              )
            )}

          </div>

        </div>

        <div className="history-metrics">

          <div className="history-metric">

            <span>
              FIELD RISK
            </span>

            <strong>
              {Number(
                selectedScan?.risk_percentage ||
                  0
              ).toFixed(2)}
              %
            </strong>

            <small>
              spectral stress coverage
            </small>

          </div>

          <div className="history-metric">

            <span>
              AFFECTED AREA
            </span>

            <strong>
              {Number(
                selectedScan
                  ?.risk_acres_estimated ||
                  0
              ).toFixed(2)}
              <small className="metric-unit">
                {" "}
                ACRES
              </small>
            </strong>

            <small>
              estimated risk area
            </small>

          </div>

          <div className="history-metric">

            <span>
              HIGH RISK
            </span>

            <strong>
              {selectedScan
                ?.high_risk_zones ??
                0}
            </strong>

            <small>
              detected zones
            </small>

          </div>

          <div className="history-metric">

            <span>
              HEALTHY
            </span>

            <strong>
              {selectedScan
                ?.healthy_zones ??
                0}
            </strong>

            <small>
              monitored zones
            </small>

          </div>

        </div>

        <div className="history-footer">

          <div>

            <span>
              AI MODEL
            </span>

            <strong>
              {selectedScan?.model ||
                "3D-CNN + Vision Transformer"}
            </strong>

          </div>

          <div>

            <span>
              TEMPORAL POSITION
            </span>

            <strong>
              SCAN {selectedIndex + 1}
              {" "}
              OF
              {" "}
              {history.length}
            </strong>

          </div>

          <div>

            <span>
              SPECTRAL STATUS
            </span>

            <strong
              className={
                health.className
              }
            >
              {health.label}
            </strong>

          </div>

        </div>

      </div>

    </section>
  );
}