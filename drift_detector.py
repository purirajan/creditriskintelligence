"""
drift_detector.py
=================
Production model drift detection using industry-standard credit risk metrics.

Implements:
  - Population Stability Index (PSI) — score distribution shift
  - KS Statistic — discriminatory power
  - Gini Coefficient / AUC-ROC — overall model performance
  - Characteristic Stability Index (CSI) — individual feature drift

PSI thresholds (industry standard):
  PSI < 0.10  → Stable (no action)
  PSI 0.10–0.25 → Moderate shift (investigate, increase monitoring frequency)
  PSI > 0.25  → Significant shift (model review / redevelopment required)

Regulatory alignment: SR 11-7 §IV (ongoing monitoring),
                      OCC 2011-12 (model validation),
                      Basel II IRB ongoing monitoring requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

PSI_STABLE   = 0.10
PSI_MODERATE = 0.25   # above = critical

KS_MIN_ACCEPTABLE = 0.20    # KS below this → poor discrimination
GINI_DEGRADATION_WARN = 0.05   # absolute Gini drop from baseline
AUC_DEGRADATION_WARN  = 0.025  # absolute AUC drop from baseline


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DriftAlert:
    metric: str
    current_value: float
    baseline_value: float | None
    threshold: float
    severity: str           # "info" | "warning" | "critical"
    timestamp: str
    message: str
    recommended_action: str


@dataclass
class MonitoringRun:
    model_id: str
    run_id: str
    run_timestamp: str
    status: str             # "healthy" | "warning" | "critical"
    metrics: dict[str, float]
    alerts: list[DriftAlert]
    alert_count: int
    feature_stability: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "status": self.status,
            "metrics": self.metrics,
            "alerts": [a.__dict__ for a in self.alerts],
            "alert_count": self.alert_count,
            "feature_stability": self.feature_stability,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index.

    Uses equal-width bins on [0, 1] (score range).
    Returns PSI ≥ 0; higher = more drift.
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)

    breakpoints = np.linspace(0.0, 1.0, bins + 1)
    e_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    a_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)

    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return round(psi, 6)


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic.

    Measures separation between default and non-default score distributions.
    Returns KS ∈ [0, 1]; higher = better separation.
    """
    from scipy import stats as scipy_stats

    defaults     = y_score[y_true == 1]
    non_defaults = y_score[y_true == 0]

    if len(defaults) == 0 or len(non_defaults) == 0:
        logger.warning("compute_ks: no defaults or no non-defaults in sample.")
        return 0.0

    ks_stat, _ = scipy_stats.ks_2samp(defaults, non_defaults)
    return round(float(ks_stat), 6)


def compute_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient = 2 × AUC − 1."""
    try:
        auc = roc_auc_score(y_true, y_score)
        return round(float(2 * auc - 1), 6)
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_gini error: %s", exc)
        return 0.0


def compute_csi(
    expected_feature: np.ndarray,
    actual_feature: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Characteristic Stability Index for a single input feature.

    Same formula as PSI but applied to an input feature distribution.
    Used to identify which features are drifting and may be causing score drift.
    """
    return compute_psi(expected_feature, actual_feature, bins=bins)


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Scheduled model health monitor.

    Instantiate once per model with baseline statistics, then call
    .run() each monitoring period (daily, weekly, monthly) with
    current production scores and outcomes.

    Usage
    -----
    >>> monitor = DriftMonitor(
    ...     model_id="PD-CONSUMER-001",
    ...     baseline_scores=dev_scores,
    ...     baseline_ks=0.42,
    ...     baseline_gini=0.63,
    ... )
    >>> result = monitor.run(current_scores, y_true_current)
    >>> result.status
    'healthy'
    """

    def __init__(
        self,
        model_id: str,
        baseline_scores: np.ndarray,
        baseline_ks: float,
        baseline_gini: float,
        baseline_features: pd.DataFrame | None = None,
        run_id_prefix: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.baseline_scores = np.asarray(baseline_scores)
        self.baseline_ks = baseline_ks
        self.baseline_gini = baseline_gini
        self.baseline_features = baseline_features
        self._run_counter = 0
        self._run_id_prefix = run_id_prefix or model_id

    # ------------------------------------------------------------------
    # Main monitoring run
    # ------------------------------------------------------------------

    def run(
        self,
        current_scores: np.ndarray,
        y_true: np.ndarray,
        current_features: pd.DataFrame | None = None,
        notes: str = "",
    ) -> MonitoringRun:
        """
        Execute a full monitoring run.

        Parameters
        ----------
        current_scores   : model scores from production population
        y_true           : realised default labels (1/0)
        current_features : optional DataFrame for CSI analysis
        notes            : free-text annotation for the audit log

        Returns
        -------
        MonitoringRun with status, metrics, and any alerts
        """
        import uuid

        self._run_counter += 1
        now = datetime.now(timezone.utc).isoformat()
        run_id = f"{self._run_id_prefix}-RUN-{self._run_counter:04d}-{uuid.uuid4().hex[:6]}"

        current_scores = np.asarray(current_scores)
        y_true = np.asarray(y_true)

        alerts: list[DriftAlert] = []

        # --- PSI ---
        psi = compute_psi(self.baseline_scores, current_scores)
        alerts += self._check_psi(psi, now)

        # --- KS ---
        ks = compute_ks(y_true, current_scores)
        alerts += self._check_ks(ks, now)

        # --- Gini ---
        gini = compute_gini(y_true, current_scores)
        alerts += self._check_gini(gini, now)

        # --- AUC ---
        try:
            auc = float(roc_auc_score(y_true, current_scores))
        except Exception:  # noqa: BLE001
            auc = 0.0
        alerts += self._check_auc(auc, now)

        # --- CSI (per-feature) ---
        feature_stability: dict[str, float] = {}
        if (
            current_features is not None
            and self.baseline_features is not None
        ):
            shared_cols = [
                c for c in current_features.columns
                if c in self.baseline_features.columns
            ]
            for col in shared_cols:
                try:
                    csi = compute_csi(
                        self.baseline_features[col].values,
                        current_features[col].values,
                    )
                    feature_stability[col] = csi
                    if csi > PSI_MODERATE:
                        alerts.append(DriftAlert(
                            metric=f"CSI_{col}",
                            current_value=csi,
                            baseline_value=None,
                            threshold=PSI_MODERATE,
                            severity="critical",
                            timestamp=now,
                            message=f"Feature '{col}' has critical distribution shift (CSI={csi:.4f}).",
                            recommended_action=f"Investigate data pipeline for {col}. Consider feature recalibration.",
                        ))
                    elif csi > PSI_STABLE:
                        alerts.append(DriftAlert(
                            metric=f"CSI_{col}",
                            current_value=csi,
                            baseline_value=None,
                            threshold=PSI_STABLE,
                            severity="warning",
                            timestamp=now,
                            message=f"Feature '{col}' shows moderate drift (CSI={csi:.4f}).",
                            recommended_action=f"Monitor {col} distribution weekly.",
                        ))

        # --- Overall status ---
        if any(a.severity == "critical" for a in alerts):
            status = "critical"
        elif any(a.severity == "warning" for a in alerts):
            status = "warning"
        else:
            status = "healthy"

        metrics = {
            "psi":             round(psi,  6),
            "ks_statistic":    round(ks,   6),
            "gini_coefficient":round(gini, 6),
            "auc_roc":         round(auc,  6),
            "n_observations":  len(current_scores),
            "default_rate":    round(float(y_true.mean()), 6),
        }

        if status != "healthy":
            logger.warning(
                "[%s] run=%s status=%s  alerts=%d  PSI=%.4f  KS=%.4f  Gini=%.4f",
                self.model_id, run_id, status, len(alerts), psi, ks, gini,
            )
        else:
            logger.info(
                "[%s] run=%s status=healthy  PSI=%.4f  KS=%.4f  Gini=%.4f",
                self.model_id, run_id, psi, ks, gini,
            )

        return MonitoringRun(
            model_id=self.model_id,
            run_id=run_id,
            run_timestamp=now,
            status=status,
            metrics=metrics,
            alerts=alerts,
            alert_count=len(alerts),
            feature_stability=feature_stability,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Alert helpers
    # ------------------------------------------------------------------

    def _check_psi(self, psi: float, ts: str) -> list[DriftAlert]:
        if psi > PSI_MODERATE:
            return [DriftAlert(
                metric="PSI", current_value=psi, baseline_value=None,
                threshold=PSI_MODERATE, severity="critical", timestamp=ts,
                message=f"Critical population shift detected (PSI={psi:.4f} > {PSI_MODERATE}).",
                recommended_action=(
                    "Immediate model review required. "
                    "Notify model owner and risk management. "
                    "Investigate population composition changes. "
                    "Consider model redevelopment or recalibration."
                ),
            )]
        if psi > PSI_STABLE:
            return [DriftAlert(
                metric="PSI", current_value=psi, baseline_value=None,
                threshold=PSI_STABLE, severity="warning", timestamp=ts,
                message=f"Moderate population shift detected (PSI={psi:.4f}).",
                recommended_action=(
                    "Investigate population and origination channel changes. "
                    "Increase monitoring frequency to weekly."
                ),
            )]
        return []

    def _check_ks(self, ks: float, ts: str) -> list[DriftAlert]:
        if ks < KS_MIN_ACCEPTABLE:
            return [DriftAlert(
                metric="KS_statistic", current_value=ks, baseline_value=self.baseline_ks,
                threshold=KS_MIN_ACCEPTABLE, severity="warning", timestamp=ts,
                message=f"KS statistic ({ks:.4f}) below minimum acceptable threshold ({KS_MIN_ACCEPTABLE}).",
                recommended_action="Review model discriminatory power. Schedule validation review.",
            )]
        ks_drop = self.baseline_ks - ks
        if ks_drop > 0.05:
            return [DriftAlert(
                metric="KS_degradation", current_value=ks, baseline_value=self.baseline_ks,
                threshold=0.05, severity="warning", timestamp=ts,
                message=f"KS degraded by {ks_drop:.4f} from baseline ({self.baseline_ks:.4f} → {ks:.4f}).",
                recommended_action="Evaluate need for model recalibration.",
            )]
        return []

    def _check_gini(self, gini: float, ts: str) -> list[DriftAlert]:
        drop = self.baseline_gini - gini
        if drop > GINI_DEGRADATION_WARN:
            severity = "critical" if drop > 0.10 else "warning"
            return [DriftAlert(
                metric="Gini_degradation", current_value=gini, baseline_value=self.baseline_gini,
                threshold=GINI_DEGRADATION_WARN, severity=severity, timestamp=ts,
                message=f"Gini dropped {drop:.4f} from baseline ({self.baseline_gini:.4f} → {gini:.4f}).",
                recommended_action=(
                    "Schedule immediate model validation review."
                    if severity == "critical"
                    else "Monitor weekly; consider recalibration."
                ),
            )]
        return []

    def _check_auc(self, auc: float, ts: str) -> list[DriftAlert]:
        baseline_auc = (self.baseline_gini + 1) / 2
        drop = baseline_auc - auc
        if drop > AUC_DEGRADATION_WARN:
            return [DriftAlert(
                metric="AUC_degradation", current_value=auc, baseline_value=baseline_auc,
                threshold=AUC_DEGRADATION_WARN, severity="warning", timestamp=ts,
                message=f"AUC-ROC dropped {drop:.4f} from baseline ({baseline_auc:.4f} → {auc:.4f}).",
                recommended_action="Evaluate model performance; schedule validation.",
            )]
        return []
