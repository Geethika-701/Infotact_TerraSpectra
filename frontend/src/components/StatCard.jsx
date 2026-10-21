import React from "react";

export default function StatCard({
  label,
  value,
  unit,
}) {
  return (
    <div className="stat-card">

      <span className="stat-label">
        {label}
      </span>

      <div className="stat-value">
        {value}
      </div>

      <span className="stat-unit">
        {unit}
      </span>

    </div>
  );
}