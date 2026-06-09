"""
governance/thresholds.py
========================
Centralised model performance and monitoring threshold definitions.

All threshold values are sourced from:
  - SR 11-7 (Federal Reserve model risk guidance)
  - Basel II/III IRB minimum requirements
  - OCC 2011-12 (model validation)
  - Internal CreditRisk Intelligence standards

Modifying thresholds requires a governance change ticket and
approval from the model owner and risk team.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Threshold:
    name: str
    warning: float
    critical: float
    direction: str      # "above" (flag if > threshold) | "below" (flag if < threshold)
    unit: str
    regulatory_basis: str
    description: str


# ---------------------------------------------------------------------------
# PSI / Drift thresholds
# ---------------------------------------------------------------------------

PSI = Threshold(
    name="Population Stability Index",
    warning=0.10,
    critical=0.25,
    direction="above",
    unit="dimensionless",
    regulatory_basis="SR 11-7 §IV; OCC 2011-12",
    description=(
        "Measures score distribution shift between development and production. "
        "PSI < 0.10: stable; 0.10–0.25: investigate; > 0.25: redevelopment required."
    ),
)

CSI = Threshold(
    name="Characteristic Stability Index",
    warning=0.10,
    critical=0.25,
    direction="above",
    unit="dimensionless",
    regulatory_basis="SR 11-7 §IV",
    description=(
        "Per-feature distribution drift. Same interpretation as PSI. "
        "Critical CSI on key features often precedes score PSI degradation."
    ),
)

# ---------------------------------------------------------------------------
# Discriminatory power thresholds
# ---------------------------------------------------------------------------

GINI = Threshold(
    name="Gini Coefficient",
    warning=0.30,
    critical=0.20,
    direction="below",
    unit="fraction",
    regulatory_basis="Basel II §285; SR 11-7",
    description=(
        "Minimum acceptable Gini. "
        "Regulatory minimum is typically 0.20–0.25 for IRB models. "
        "Internal warning threshold set at 0.30."
    ),
)

KS = Threshold(
    name="Kolmogorov-Smirnov Statistic",
    warning=0.25,
    critical=0.20,
    direction="below",
    unit="fraction",
    regulatory_basis="SR 11-7; industry practice",
    description=(
        "KS measures separation of default and non-default distributions. "
        "KS < 0.20 indicates poor discriminatory power."
    ),
)

AUC_ROC = Threshold(
    name="AUC-ROC",
    warning=0.65,
    critical=0.60,
    direction="below",
    unit="fraction",
    regulatory_basis="Basel II §285",
    description=(
        "Area Under the ROC Curve. AUC = 0.5 is no better than random. "
        "Minimum acceptable for regulatory use is typically 0.65."
    ),
)

# ---------------------------------------------------------------------------
# Calibration thresholds
# ---------------------------------------------------------------------------

HOSMER_LEMESHOW_P = Threshold(
    name="Hosmer-Lemeshow p-value",
    warning=0.05,
    critical=0.01,
    direction="below",
    unit="p-value",
    regulatory_basis="Basel II §482; OCC 2011-12",
    description=(
        "Tests PD calibration (observed vs predicted default rates by score decile). "
        "p < 0.05 indicates significant calibration failure."
    ),
)

MEAN_PD_DEVIATION = Threshold(
    name="Mean PD Deviation",
    warning=0.005,
    critical=0.015,
    direction="above",
    unit="fraction",
    regulatory_basis="Basel II §501",
    description=(
        "Absolute difference between mean predicted PD and observed default rate. "
        "Persistent positive deviation (over-prediction) or negative deviation "
        "(under-prediction) triggers recalibration."
    ),
)

# ---------------------------------------------------------------------------
# LGD-specific
# ---------------------------------------------------------------------------

LGD_MEAN_DEVIATION = Threshold(
    name="Mean LGD Deviation",
    warning=0.03,
    critical=0.07,
    direction="above",
    unit="fraction",
    regulatory_basis="Basel II §468; EBA GL/2017/16",
    description=(
        "Absolute deviation of mean predicted LGD from realised LGD. "
        "Basel IRB requires LGD estimates to include a downturn adjustment."
    ),
)

# ---------------------------------------------------------------------------
# Threshold registry (for programmatic access)
# ---------------------------------------------------------------------------

ALL_THRESHOLDS: dict[str, Threshold] = {
    "psi":                PSI,
    "csi":                CSI,
    "gini":               GINI,
    "ks":                 KS,
    "auc_roc":            AUC_ROC,
    "hosmer_lemeshow_p":  HOSMER_LEMESHOW_P,
    "mean_pd_deviation":  MEAN_PD_DEVIATION,
    "lgd_mean_deviation": LGD_MEAN_DEVIATION,
}


def check_threshold(metric_name: str, value: float) -> tuple[bool, Severity]:
    """
    Check a metric value against registered thresholds.

    Returns (is_flagged, severity).
    """
    t = ALL_THRESHOLDS.get(metric_name.lower())
    if t is None:
        return False, Severity.INFO

    if t.direction == "above":
        if value > t.critical:
            return True, Severity.CRITICAL
        if value > t.warning:
            return True, Severity.WARNING
    else:  # below
        if value < t.critical:
            return True, Severity.CRITICAL
        if value < t.warning:
            return True, Severity.WARNING

    return False, Severity.INFO
