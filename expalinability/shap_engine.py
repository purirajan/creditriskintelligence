"""
shap_engine.py
==============
SHAP-based model explainability for fair lending compliance.

Generates:
  1. Individual prediction explanations with signed SHAP contributions
  2. ECOA / Regulation B adverse action reason codes (top 4)
  3. Portfolio-level global feature importance via mean |SHAP|
  4. Force-plot data for dashboard visualisation

Regulatory alignment:
  - ECOA (Equal Credit Opportunity Act) — adverse action reason codes
  - Regulation B — requires up to 4 specific reasons for adverse action
  - Fair Housing Act / HMDA — disparate impact inputs
  - SR 11-7 — model explainability documentation

Dependencies: shap  (pip install shap)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regulation B adverse action code catalogue
# ---------------------------------------------------------------------------

# Maps internal feature names → (Reg-B code, plain-language description)
# Descriptions are written to be consumer-facing (ECOA §202.9(b)(2))
ADVERSE_ACTION_CODES: dict[str, tuple[str, str]] = {
    "fico_score":              ("AA-01", "Insufficient credit history or low credit score"),
    "dti_ratio":               ("AA-02", "Debt-to-income ratio too high"),
    "utilization_rate":        ("AA-03", "Credit utilisation rate too high"),
    "months_on_book":          ("AA-04", "Insufficient length of credit history"),
    "delinquency_count":       ("AA-05", "Delinquency on existing account(s)"),
    "income_verified":         ("AA-06", "Unable to verify income"),
    "collateral_value":        ("AA-07", "Insufficient collateral value"),
    "loan_amount":             ("AA-08", "Requested loan amount exceeds guidelines"),
    "months_since_last_draw":  ("AA-09", "Recent account inactivity"),
    "payment_behaviour_score": ("AA-10", "Insufficient payment history"),
    "origination_ltv":         ("AA-11", "Loan-to-value ratio too high"),
    "secured_flag":            ("AA-12", "Collateral requirements not met"),
}

MAX_ADVERSE_REASONS = 4   # Regulation B maximum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AdverseActionReason:
    code: str
    description: str
    shap_contribution: float     # positive = increases PD / risk


@dataclass
class ExplanationResult:
    application_id: str | None
    model_id: str
    base_value: float
    prediction: float
    feature_contributions: dict[str, float]   # feature → SHAP value
    top_risk_drivers: list[str]               # sorted by |SHAP|, descending
    adverse_action_reasons: list[AdverseActionReason]
    is_adverse_action: bool
    regulatory_note: str = (
        "Adverse action reasons comply with ECOA/Regulation B requirements. "
        "Maximum of 4 specific reasons are provided per 12 C.F.R. §202.9."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "model_id": self.model_id,
            "base_value": self.base_value,
            "prediction": self.prediction,
            "feature_contributions": self.feature_contributions,
            "top_risk_drivers": self.top_risk_drivers,
            "adverse_action_reasons": [
                {
                    "code": r.code,
                    "description": r.description,
                    "shap_contribution": r.shap_contribution,
                }
                for r in self.adverse_action_reasons
            ],
            "is_adverse_action": self.is_adverse_action,
            "regulatory_note": self.regulatory_note,
        }


# ---------------------------------------------------------------------------
# SHAP explainer wrapper
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """
    Wraps a fitted credit risk model with SHAP-based explanations.

    Supports:
      - Linear models  → shap.LinearExplainer
      - Tree models    → shap.TreeExplainer  (exact, fast)
      - Any model      → shap.KernelExplainer (slower, model-agnostic fallback)

    Usage
    -----
    >>> explainer = SHAPExplainer(fitted_pd_model, feature_names=FEATURES)
    >>> explainer.fit(X_background)
    >>> result = explainer.explain(X_single, application_id="APP-001")
    >>> result.adverse_action_reasons
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        model_family: str = "auto",       # "linear" | "tree" | "kernel" | "auto"
        adverse_action_threshold: float = 0.05,  # PD above which adverse action applies
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.model_family = model_family
        self.adverse_action_threshold = adverse_action_threshold
        self._explainer: Any = None
        self._background_data: np.ndarray | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_background: pd.DataFrame | np.ndarray,
        background_sample_size: int = 200,
    ) -> "SHAPExplainer":
        """
        Initialise the SHAP explainer on background data.

        Parameters
        ----------
        X_background    : representative sample of training data (for baseline)
        background_sample_size : number of rows to subsample for SHAP background
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required. pip install shap") from exc

        if isinstance(X_background, pd.DataFrame):
            X_bg = X_background[self.feature_names].values
        else:
            X_bg = np.array(X_background)

        # Subsample background for speed
        if len(X_bg) > background_sample_size:
            idx = np.random.choice(len(X_bg), background_sample_size, replace=False)
            X_bg = X_bg[idx]
        self._background_data = X_bg

        family = self.model_family
        if family == "auto":
            family = self._detect_family()

        raw_model = self._unwrap_model()

        if family == "linear":
            self._explainer = shap.LinearExplainer(raw_model, shap.maskers.Independent(X_bg))
        elif family == "tree":
            self._explainer = shap.TreeExplainer(raw_model, data=X_bg)
        else:
            # Kernel explainer: model-agnostic but slower
            def predict_fn(X: np.ndarray) -> np.ndarray:
                return raw_model.predict_proba(
                    pd.DataFrame(X, columns=self.feature_names)
                )[:, 1]
            self._explainer = shap.KernelExplainer(predict_fn, X_bg)

        logger.info("SHAPExplainer fitted (family=%s, background_n=%d)", family, len(X_bg))
        return self

    # ------------------------------------------------------------------
    # explain — individual prediction
    # ------------------------------------------------------------------

    def explain(
        self,
        X: pd.DataFrame | dict,
        application_id: str | None = None,
        prediction: float | None = None,
    ) -> ExplanationResult:
        """
        Explain a single prediction.

        Parameters
        ----------
        X              : single-row input (DataFrame or dict)
        application_id : optional reference for audit trail
        prediction     : pre-computed model output (avoids re-scoring);
                         if None, base_value is used as proxy

        Returns
        -------
        ExplanationResult with SHAP values and Reg B adverse action reasons
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit() before .explain()")

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame([X])
        X_feat = X_df[self.feature_names].values

        shap_values = self._explainer.shap_values(X_feat)

        # Normalise to 1D array (class-1 slice for classifiers)
        if isinstance(shap_values, list):
            shap_array = np.array(shap_values[1]).flatten()
        else:
            shap_array = np.array(shap_values).flatten()

        base_value = self._get_base_value()

        # Build signed contribution dict
        contributions: dict[str, float] = {
            f: round(float(v), 6)
            for f, v in zip(self.feature_names, shap_array)
        }

        # Sort by absolute contribution
        sorted_feats = sorted(
            contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        top_risk_drivers = [f for f, _ in sorted_feats[:6]]

        # Prediction
        pred = prediction if prediction is not None else float(
            base_value + sum(contributions.values())
        )
        pred = float(np.clip(pred, 0.0, 1.0))

        # Adverse action (triggered if prediction > threshold)
        is_adverse = pred >= self.adverse_action_threshold
        adverse_reasons = self._build_adverse_reasons(sorted_feats) if is_adverse else []

        model_id = getattr(self.model, "metadata", None)
        model_id = model_id.model_id if model_id else "unknown"

        return ExplanationResult(
            application_id=application_id,
            model_id=model_id,
            base_value=round(float(base_value), 6),
            prediction=round(pred, 6),
            feature_contributions=contributions,
            top_risk_drivers=top_risk_drivers,
            adverse_action_reasons=adverse_reasons,
            is_adverse_action=is_adverse,
        )

    # ------------------------------------------------------------------
    # Portfolio-level importance
    # ------------------------------------------------------------------

    def portfolio_importance(
        self, X: pd.DataFrame, sample_size: int = 500
    ) -> dict[str, float]:
        """
        Compute global feature importance as mean |SHAP| across a portfolio.

        Returns {feature: mean_abs_shap} sorted descending.
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit() before portfolio_importance()")

        X_feat = X[self.feature_names]
        if len(X_feat) > sample_size:
            X_feat = X_feat.sample(sample_size, random_state=42)

        shap_values = self._explainer.shap_values(X_feat.values)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        total = mean_abs.sum() or 1.0

        return dict(
            sorted(
                {
                    f: round(float(v / total), 5)
                    for f, v in zip(self.feature_names, mean_abs)
                }.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_family(self) -> str:
        """Heuristic model family detection."""
        model = self._unwrap_model()
        cls = type(model).__name__.lower()
        if any(k in cls for k in ("logistic", "linear", "ridge", "lasso")):
            return "linear"
        if any(k in cls for k in ("xgb", "lgbm", "catboost", "forest", "tree", "gradient")):
            return "tree"
        return "kernel"

    def _unwrap_model(self) -> Any:
        """Unwrap CalibratedClassifierCV or Pipeline to the core estimator."""
        model = self.model
        # Unwrap governance wrapper
        if hasattr(model, "_pipeline"):
            model = model._pipeline
        # Unwrap calibration
        if hasattr(model, "estimator"):
            model = model.estimator
        # Unwrap Pipeline to final step
        if hasattr(model, "named_steps"):
            model = list(model.named_steps.values())[-1]
        return model

    def _get_base_value(self) -> float:
        ev = getattr(self._explainer, "expected_value", 0.0)
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[1] if len(ev) > 1 else ev[0]
        return float(ev)

    @staticmethod
    def _build_adverse_reasons(
        sorted_contribs: list[tuple[str, float]],
    ) -> list[AdverseActionReason]:
        """
        Build Regulation B adverse action reasons from SHAP values.

        Only features with positive SHAP (i.e., increasing PD / risk) qualify.
        Returns at most MAX_ADVERSE_REASONS reasons.
        """
        reasons: list[AdverseActionReason] = []
        for feature, shap_val in sorted_contribs:
            if shap_val <= 0:
                continue   # risk-reducing feature — not an adverse reason
            if feature not in ADVERSE_ACTION_CODES:
                continue   # no Reg B code mapped
            code, description = ADVERSE_ACTION_CODES[feature]
            reasons.append(
                AdverseActionReason(
                    code=code,
                    description=description,
                    shap_contribution=round(shap_val, 6),
                )
            )
            if len(reasons) == MAX_ADVERSE_REASONS:
                break
        return reasons
