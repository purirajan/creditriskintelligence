"""
shap_engine.py — SHAP-based model explainability for fair lending compliance.

Generates:
  - Individual prediction explanations (ECOA/Reg B adverse action reasons)
  - Portfolio-level feature importance
  - Disparate impact analysis inputs
"""

import numpy as np
import shap
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ECOA Reg B — standard adverse action reason codes
ADVERSE_ACTION_CODES = {
    "fico_score":         ("AA-01", "Insufficient credit history / low credit score"),
    "dti_ratio":          ("AA-02", "Debt-to-income ratio too high"),
    "utilization_rate":   ("AA-03", "Credit utilization too high"),
    "months_on_book":     ("AA-04", "Insufficient length of credit history"),
    "delinquency_count":  ("AA-05", "Delinquency on account(s)"),
    "income_verified":    ("AA-06", "Unable to verify income"),
}


class SHAPExplainer:
    """
    Wraps a trained credit risk model with SHAP explanations.
    
    Produces:
    - Top adverse action reasons (ECOA Reg B compliant)
    - SHAP force plot data
    - Feature contribution breakdown
    """

    def __init__(self, model, feature_names: List[str], model_type: str = "logistic"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self._explainer = None

    def fit_explainer(self, X_background):
        """Fit the SHAP explainer on background data."""
        if self.model_type == "logistic":
            self._explainer = shap.LinearExplainer(
                self.model.named_steps["clf"],
                shap.sample(X_background, 100)
            )
        else:
            self._explainer = shap.TreeExplainer(self.model)
        logger.info("SHAP explainer fitted.")

    def explain_prediction(
        self, X_instance, application_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Returns SHAP values + top 4 adverse action reasons (Reg B).
        """
        if self._explainer is None:
            raise RuntimeError("Call fit_explainer() before explain_prediction()")

        shap_values = self._explainer.shap_values(X_instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        shap_array = np.array(shap_values).flatten()

        # Build contribution dict
        contributions = {
            f: round(float(v), 5)
            for f, v in zip(self.feature_names, shap_array)
        }

        # Sort by absolute contribution
        sorted_contribs = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Map top negative contributors to adverse action codes
        adverse_reasons = []
        for feature, shap_val in sorted_contribs:
            if shap_val > 0 and feature in ADVERSE_ACTION_CODES:  # increases PD
                code, description = ADVERSE_ACTION_CODES[feature]
                adverse_reasons.append({
                    "code": code,
                    "description": description,
                    "shap_contribution": round(shap_val, 5),
                })
            if len(adverse_reasons) == 4:  # Reg B requires top 4
                break

        return {
            "application_id": application_id,
            "feature_contributions": contributions,
            "top_risk_drivers": [f for f, _ in sorted_contribs[:5]],
            "adverse_action_reasons": adverse_reasons,   # ECOA Reg B output
            "base_value": float(self._explainer.expected_value
                                if not isinstance(self._explainer.expected_value, list)
                                else self._explainer.expected_value[1]),
        }

    def portfolio_feature_importance(self, X) -> Dict[str, float]:
        """Aggregate SHAP values across portfolio for global importance."""
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        return {
            f: round(float(v), 5)
            for f, v in sorted(
                zip(self.feature_names, mean_abs),
                key=lambda x: x[1], reverse=True
            )
        }
