"""
drift_detector.py — Population Stability Index (PSI) and KS-based model drift detection.

PSI is the industry standard for credit model monitoring (Basel, SR 11-7).
Alerts when score distribution shifts between development and production populations.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DriftAlert:
    metric: str
    value: float
    threshold: float
    severity: str         # "warning" | "critical"
    timestamp: str
    recommendation: str


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    
    PSI < 0.10  → No significant shift (stable)
    PSI 0.10–0.25 → Moderate shift (investigate)
    PSI > 0.25  → Significant shift (model review required)
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.linspace(0, 1, bins + 1)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Clip to avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct   = np.clip(actual_pct,   1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 5)


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS statistic — separation between default and non-default score distributions."""
    from scipy import stats
    defaults     = y_score[y_true == 1]
    non_defaults = y_score[y_true == 0]
    ks_stat, _   = stats.ks_2samp(defaults, non_defaults)
    return round(float(ks_stat), 5)


class DriftMonitor:
    """
    Monitors credit model score drift in production.
    Runs on a scheduled basis (daily/weekly) per SR 11-7 requirements.
    """

    PSI_WARNING  = 0.10
    PSI_CRITICAL = 0.25
    KS_WARNING   = 0.05   # >5% degradation from dev KS
    GINI_WARNING = 0.05

    def __init__(self, model_id: str, baseline_scores: np.ndarray,
                 baseline_ks: float, baseline_gini: float):
        self.model_id = model_id
        self.baseline_scores = baseline_scores
        self.baseline_ks = baseline_ks
        self.baseline_gini = baseline_gini

    def run_monitoring(
        self,
        current_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict:
        """
        Full monitoring run. Returns health status + any alerts.
        """
        alerts: List[DriftAlert] = []
        now = datetime.utcnow().isoformat()

        # PSI
        psi = compute_psi(self.baseline_scores, current_scores)
        if psi > self.PSI_CRITICAL:
            alerts.append(DriftAlert(
                metric="PSI", value=psi, threshold=self.PSI_CRITICAL,
                severity="critical", timestamp=now,
                recommendation="Immediate model review required. Consider redevelopment."
            ))
        elif psi > self.PSI_WARNING:
            alerts.append(DriftAlert(
                metric="PSI", value=psi, threshold=self.PSI_WARNING,
                severity="warning", timestamp=now,
                recommendation="Investigate population shift. Monitor closely."
            ))

        # KS degradation
        current_ks = compute_ks(y_true, current_scores)
        ks_delta = self.baseline_ks - current_ks
        if ks_delta > self.KS_WARNING:
            alerts.append(DriftAlert(
                metric="KS_degradation", value=ks_delta, threshold=self.KS_WARNING,
                severity="warning", timestamp=now,
                recommendation="Discriminatory power declining. Schedule validation."
            ))

        # Gini
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, current_scores)
        gini = 2 * auc - 1
        gini_delta = self.baseline_gini - gini
        if gini_delta > self.GINI_WARNING:
            alerts.append(DriftAlert(
                metric="Gini_degradation", value=gini_delta, threshold=self.GINI_WARNING,
                severity="warning", timestamp=now,
                recommendation="Gini coefficient dropped. Evaluate model recalibration."
            ))

        overall_status = (
            "critical" if any(a.severity == "critical" for a in alerts)
            else "warning" if alerts
            else "healthy"
        )

        return {
            "model_id": self.model_id,
            "run_timestamp": now,
            "status": overall_status,
            "metrics": {
                "psi": psi,
                "ks_statistic": current_ks,
                "gini_coefficient": round(gini, 5),
                "auc_roc": round(float(auc), 5),
            },
            "alerts": [vars(a) for a in alerts],
            "alert_count": len(alerts),
        }
