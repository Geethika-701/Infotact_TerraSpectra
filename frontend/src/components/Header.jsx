import React from "react";

export default function Header({
  apiOnline = false
}) {
  return (
    <header className="top-header">

      <div className="header-brand">

        <div className="header-logo">
          🌿
        </div>

        <div>

          <h2>
            TerraSpectra
          </h2>

          <span>
            Precision Crop Intelligence
          </span>

        </div>

      </div>

      <div className="header-right">

        <div
          className={
            apiOnline
              ? "system-status online"
              : "system-status"
          }
        >

          <span></span>

          {apiOnline
            ? "SYSTEM ONLINE"
            : "SYSTEM OFFLINE"}

        </div>

        <div className="last-scan">
          Last scan: 2 min ago
        </div>

      </div>

    </header>
  );
}