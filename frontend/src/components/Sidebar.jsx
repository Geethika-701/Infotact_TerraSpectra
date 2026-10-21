import React from "react";

export default function Sidebar() {
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

        <button className="sidebar-link active">
          <span>▣</span>
          Dashboard
        </button>

        <button className="sidebar-link">
          <span>◉</span>
          Risk Map
        </button>

        <button className="sidebar-link">
          <span>◈</span>
          Spectral
        </button>

        <button className="sidebar-link">
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
              3D-CNN ACTIVE
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