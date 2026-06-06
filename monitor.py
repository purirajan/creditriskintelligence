"""
routes/monitor.py
=================
Model monitoring endpoints: health status, PSI/KS/Gini metrics, alert history.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status

from ..schemas import ModelHealthResponse

router = APIRouter()

# ---------------------------------------------------------------------------
# Mock model health store (replace with DB / monitoring service in production)
# ---------------------------------------------------------------------------

_MODEL_HEALTH: dict[str, dict[str, Any]] = {
    "PD-CONSUMER-001": {
        "status": "healthy",
        "last_run": "2026-06-05T06:00:00Z",
        "metrics": {
            "psi":              0.041,
            "ks_statistic":     0.412,
            "gini_coefficient": 0.631,
            "auc_roc":          0.816,
            "default_rate":     0.042,
            "n_observations":   12450,
        },
        "alerts": [],
        "next_scheduled_run": "2026-06-06T06:00:00Z",
    },
    "LGD-CONSUMER-001": {
        "status": "healthy",
        "last_run": "2026-06-05T06:05:00Z",
        "metrics": {
            "psi":     0.028,
            "mae":     0.062,
            "mean_lgd": 0.431,
            "n_observations": 8320,
        },
        "alerts": [],
        "next_scheduled_run": "2026-06-06T06:05:00Z",
    },
    "EAD-CONSUMER-001": {
        "status": "warning",
        "last_run": "2026-06-05T06:10:00Z",
        "metrics": {
            "psi":      0.138,
            "mae_ccf":  0.071,
            "mean_ccf": 0.612,
            "n_observations": 5890,
        },
        "alerts": [
            {
                "metric": "PSI",
                "current_value": 0.138,
                "threshold": 0.10,
                "severity": "warning",
                "timestamp": "2026-06-05T06:10:00Z",
                "message": "Moderate population shift detected (PSI=0.1380).",
                "recommended_action": "Investigate BNPL origination channel changes.",
            }
        ],
        "next_scheduled_run": "2026-06-06T06:10:00Z",
    },
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/monitor/{model_id}",
    response_model=ModelHealthResponse,
    summary="Get model health metrics",
    description=(
        "Returns current PSI, KS, Gini, AUC, and any active alerts "
        "for a given model. Data refreshed on each scheduled monitoring run."
    ),
)
def get_model_health(model_id: str) -> ModelHealthResponse:
    health = _MODEL_HEALTH.get(model_id)
    if health is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in monitoring registry.",
        )
    return ModelHealthResponse(
        model_id=model_id,
        status=health["status"],
        last_run=health["last_run"],
        metrics=health["metrics"],
        alerts=health.get("alerts", []),
        next_scheduled_run=health.get("next_scheduled_run"),
    )


@router.get(
    "/monitor",
    summary="Get health summary for all registered models",
    description="Returns a summary view of all models in the monitoring registry.",
)
def list_model_health() -> dict[str, Any]:
    summary = []
    for model_id, health in _MODEL_HEALTH.items():
        summary.append({
            "model_id": model_id,
            "status": health["status"],
            "last_run": health["last_run"],
            "alert_count": len(health.get("alerts", [])),
            "psi": health["metrics"].get("psi"),
        })

    critical = sum(1 for s in summary if s["status"] == "critical")
    warning  = sum(1 for s in summary if s["status"] == "warning")
    healthy  = sum(1 for s in summary if s["status"] == "healthy")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": "critical" if critical else ("warning" if warning else "healthy"),
        "summary": {
            "total_models": len(summary),
            "healthy": healthy,
            "warning": warning,
            "critical": critical,
        },
        "models": summary,
    }
