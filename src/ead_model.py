"""
ead_model.py
============
Exposure at Default (EAD) model for revolving consumer credit facilities.

EAD = Drawn Balance + CCF × Undrawn Commitment

Credit Conversion Factor (CCF) estimates the fraction of currently undrawn
credit that will be drawn by the time of default.  For term loans the CCF
is irrelevant (EAD = outstanding balance); for revolving facilities (credit
cards, BNPL credit limits, HELOC) the CCF drives significant EAD uncertainty.

Regulatory alignment: Basel II/III IRB (EAD pillar), CECL,
                      IFRS 9 (Stage 2/3 ECL calculation).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseCreditRiskModel, ModelMetadata, ValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

REQUIRED_FEATURES = [
    "current_balance",          # USD  (drawn amount)
    "credit_limit",             # USD  (total facility limit)
    "utilization_rate",         # current_balance / credit_limit
    "months_to_maturity",       # 0 for revolving (no fixed maturity)
    "product_type_code",        # 0=personal_revolving, 1=credit_card, 2=BNPL, 3=term_loan
    "months_since_last_draw",   # behavioural draw frequency indicator
    "payment_behaviour_score",  # internal 0-100 score (100 = always pays on time)
]

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "current_balance":        (0, 500_000),
    "credit_limit":           (100, 500_000),
    "utilization_rate":       (0, 1),
    "months_to_maturity":     (0, 360),
    "product_type_code":      (0, 10),
    "months_since_last_draw": (0, 120),
    "payment_behaviour_score":(0, 100),
}

# Basel regulatory CCF floors by product (Basel III CRR Art. 111)
_REGULATORY_CCF_FLOORS: dict[int, float] = {
    0: 0.75,   # personal revolving
    1: 0.75,   # credit card
    2: 0.50,   # BNPL (shorter tenors, lower draw uncertainty)
    3: 0.00,   # term loan (CCF irrelevant)
}


class EADModel(BaseCreditRiskModel):
    """
    Credit Conversion Factor (CCF) model → Exposure at Default.

    Workflow
    --------
    1. Predict CCF for undrawn commitment (model output ∈ [0,1])
    2. EAD = drawn + CCF × (limit − drawn)
    3. Apply regulatory CCF floor where model < floor

    Outputs
    -------
    ccf_estimate        : model CCF ∈ [0, 1]
    ead_estimate        : drawn + ccf × undrawn   (USD)
    ead_regulatory      : ead using Basel floor CCF (USD)
    undrawn_commitment  : limit − drawn            (USD)
    """

    def __init__(
        self,
        apply_regulatory_floors: bool = True,
        random_state: int = 42,
    ) -> None:
        metadata = ModelMetadata(
            model_id="EAD-CONSUMER-001",
            model_name="Consumer EAD / CCF Model",
            version="1.0.0",
            purpose=(
                "Estimate Credit Conversion Factor and Exposure at Default "
                "for revolving consumer credit facilities (credit cards, BNPL lines)."
            ),
            regulatory_use=["Basel II IRB", "Basel III IRB", "CECL", "IFRS 9"],
            developer="CreditRisk Intelligence",
            development_date="2024-04-01",
            asset_class="Consumer",
            risk_tier="High",
            status="Production",
            assumptions=[
                "CCF is estimated at a one-year horizon.",
                "Regulatory CCF floors are applied per Basel III CRR Art. 111.",
                "Term loans (product_type_code=3) have CCF=0; EAD = current_balance.",
            ],
            known_limitations=[
                "CCF model does not capture intra-day drawdown behaviour.",
                "Seasonal drawdown patterns (e.g. holiday spending) not explicitly modelled.",
                "Model trained on US market data; non-US revolvers may require recalibration.",
            ],
        )
        super().__init__(metadata)
        self.apply_regulatory_floors = apply_regulatory_floors
        self.random_state = random_state
        self._pipeline: Pipeline | None = None
        self._train_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "EADModel":
        """
        Train the CCF model.

        Parameters
        ----------
        X : DataFrame with REQUIRED_FEATURES
        y : Realised CCF ∈ [0, 1]  (drawn_at_default − drawn_at_obs) / undrawn_at_obs
        """
        X_feat = X[REQUIRED_FEATURES].copy()
        y_clipped = np.clip(y, 0.0, 1.0)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=self.random_state)),
        ])
        self._pipeline.fit(X_feat, y_clipped)

        preds = np.clip(self._pipeline.predict(X_feat), 0, 1)
        mae = mean_absolute_error(y_clipped, preds)
        self._train_metrics = {
            "train_mae": round(float(mae), 5),
            "mean_ccf":  round(float(y_clipped.mean()), 5),
            "train_size": len(y),
        }
        logger.info(
            "[%s] fit complete — MAE=%.4f  mean_CCF=%.4f  n=%d",
            self.metadata.model_id, mae, y_clipped.mean(), len(y),
        )
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: Any) -> dict[str, Any]:
        X_df = self._coerce_to_dataframe(X)
        X_feat = X_df[REQUIRED_FEATURES]

        # Model CCF
        ccf_model = float(np.clip(self._pipeline.predict(X_feat)[0], 0.0, 1.0))

        row = X_feat.iloc[0]
        product_code = int(row["product_type_code"])
        current_balance = float(row["current_balance"])
        credit_limit = float(row["credit_limit"])
        undrawn = max(credit_limit - current_balance, 0.0)

        # Apply regulatory floor
        reg_floor = _REGULATORY_CCF_FLOORS.get(product_code, 0.75)
        ccf_final = max(ccf_model, reg_floor) if self.apply_regulatory_floors else ccf_model

        ead_model = current_balance + ccf_model * undrawn
        ead_regulatory = current_balance + ccf_final * undrawn

        return {
            "ccf_estimate":       round(ccf_model, 5),
            "ccf_with_floor":     round(ccf_final, 5),
            "ead_estimate":       round(ead_model, 2),
            "ead_regulatory":     round(ead_regulatory, 2),
            "current_balance":    round(current_balance, 2),
            "undrawn_commitment": round(undrawn, 2),
            "regulatory_ccf_floor": reg_floor,
            "model_type":         "ridge_regression",
            "train_metrics":      self._train_metrics,
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        self._check_fitted()
        try:
            coefs = np.abs(self._pipeline.named_steps["reg"].coef_)
            total = coefs.sum() or 1.0
            return dict(
                sorted(
                    zip(REQUIRED_FEATURES, [round(v / total, 5) for v in coefs]),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feature importance unavailable: %s", exc)
            return {f: 0.0 for f in REQUIRED_FEATURES}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_inputs(self, X: Any) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            df = self._coerce_to_dataframe(X)
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(passed=False, errors=[str(exc)])

        missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
        if missing:
            errors.append(f"Missing features: {missing}")
            return ValidationResult(passed=False, errors=errors)

        if (df["credit_limit"] <= 0).any():
            errors.append("credit_limit must be > 0.")
        if (df["current_balance"] < 0).any():
            errors.append("current_balance must be ≥ 0.")
        if (df["current_balance"] > df["credit_limit"]).any():
            warnings.append("current_balance > credit_limit for some rows (over-limit accounts).")

        return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_to_dataframe(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, dict):
            return pd.DataFrame([X])
        raise TypeError(f"Unsupported input type: {type(X)}")
