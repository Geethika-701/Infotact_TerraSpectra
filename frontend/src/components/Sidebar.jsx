import React, {
  useState
} from "react";

export default function Sidebar() {
  const [activeSection, setActiveSection] =
    useState("dashboard");

  function navigateTo(
    section,
    elementId
  ) {
    setActiveSection(section);

    const element =
      document.getElementById(
        elementId
      );

    if (element) {
      element.scrollIntoView({
        behavior: "smooth",
        block: "start"
      });
    }
  }

  return (
    <aside className="sidebar">

      <div className="sidebar-brand">

        <div className="sidebar-logo">
          🌿
        </div>

        <div>

          <strong>
            TERRASPECTRA
          </strong>

          <span>
            Precision Agriculture
          </span>

        </div>

      </div>

      <div className="sidebar-title">
        OVERVIEW
      </div>

      <nav className="sidebar-menu">

        <button
          className={
            activeSection === "dashboard"
              ? "sidebar-link active"
              : "sidebar-link"
          }
          onClick={() =>
            navigateTo(
              "dashboard",
              "dashboard-section"
            )
          }
        >
          <span>▣</span>
          Dashboard
        </button>

        <button
          className={
            activeSection === "risk-map"
              ? "sidebar-link active"
              : "sidebar-link"
          }
          onClick={() =>
            navigateTo(
              "risk-map",
              "risk-map-section"
            )
          }
        >
          <span>◉</span>
          Risk Map
        </button>

        <button
          className={
            activeSection === "spectral"
              ? "sidebar-link active"
              : "sidebar-link"
          }
          onClick={() =>
            navigateTo(
              "spectral",
              "spectral-section"
            )
          }
        >
          <span>◈</span>
          Spectral
        </button>

        <button
          className={
            activeSection === "analytics"
              ? "sidebar-link active"
              : "sidebar-link"
          }
          onClick={() =>
            navigateTo(
              "analytics",
              "analytics-section"
            )
          }
        >
          <span>◫</span>
          Analytics
        </button>

      </nav>

      <div className="sidebar-title monitoring-title">
        MONITORING
      </div>

      <div className="monitoring-list">

        <div>
          <span className="monitor-dot healthy"></span>
          Healthy
        </div>

        <div>
          <span className="monitor-dot moderate"></span>
          Moderate
        </div>

        <div>
          <span className="monitor-dot high"></span>
          High Risk
        </div>

      </div>

      <div className="sidebar-bottom">

        <div className="sidebar-ai">

          <span className="ai-dot"></span>

          <div>

            <strong>
              AI ENGINE
            </strong>

            <small>
              3D-CNN + ViT ACTIVE
            </small>

          </div>

        </div>

        <small className="version">
          TERRASPECTRA v1.0
        </small>

      </div>

    </aside>
  );
}