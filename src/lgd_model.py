"""
lgd_model.py
============
Loss Given Default (LGD) model for consumer credit portfolios.

LGD = 1 − Recovery Rate.  Outputs are bounded [0, 1] representing the
fraction of Exposure at Default expected to be permanently lost.

Two backends:
  - "beta_regression" : Beta-distributed outcome (natural [0,1] support,
                         interpretable, preferred for regulatory submissions)
  - "xgboost"         : Gradient-boosted trees (higher accuracy, useful
                         where collateral data is rich)

Regulatory alignment: Basel II/III IRB (LGD pillar, downturn LGD),
                      CECL (loss rate component).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseCreditRiskModel, ModelMetadata, ValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

REQUIRED_FEATURES = [
    "collateral_value",     # USD; 0 for unsecured
    "loan_amount",          # USD
    "months_past_due",      # at time of default
    "secured_flag",         # 1 = secured (auto, mortgage), 0 = unsecured
    "product_type_code",    # encoded: 0=personal, 1=auto, 2=BNPL, 3=credit_card
    "origination_ltv",      # loan-to-value at origination; 0 for unsecured
    "time_in_default_months",  # months since default event
]

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "collateral_value":       (0, 10_000_000),
    "loan_amount":            (100, 10_000_000),
    "months_past_due":        (0, 120),
    "secured_flag":           (0, 1),
    "product_type_code":      (0, 10),
    "origination_ltv":        (0, 3.0),
    "time_in_default_months": (0, 120),
]

# Basel downturn LGD floor by product type
_DOWNTURN_FLOORS: dict[int, float] = {
    0: 0.45,   # personal loan (unsecured)
    1: 0.25,   # auto loan    (secured)
    2: 0.60,   # BNPL         (unsecured, high uncertainty)
    3: 0.50,   # credit card  (unsecured)
}


class LGDModel(BaseCreditRiskModel):
    """
    Consumer Loss Given Default model.

    Outputs
    -------
    lgd_estimate        : point estimate ∈ [0, 1]
    lgd_downturn        : Basel downturn LGD (stressed estimate)
    recovery_rate       : 1 − lgd_estimate
    confidence_interval : (lower_95, upper_95)
    """

    def __init__(
        self,
        model_type: str = "beta_regression",
        apply_downturn_floor: bool = True,
        random_state: int = 42,
    ) -> None:
        if model_type not in ("beta_regression", "xgboost"):
            raise ValueError("model_type must be 'beta_regression' or 'xgboost'")

        metadata = ModelMetadata(
            model_id="LGD-CONSUMER-001",
            model_name="Consumer LGD Model",
            version="1.1.0",
            purpose=(
                "Estimate Loss Given Default for consumer credit facilities. "
                "Outputs used in Basel IRB RWA calculation and CECL allowance estimation."
            ),
            regulatory_use=["Basel II IRB", "Basel III IRB", "CECL"],
            developer="CreditRisk Intelligence",
            development_date="2024-03-01",
            asset_class="Consumer",
            risk_tier="High",
            status="Production",
            assumptions=[
                "LGD estimates are through-the-cycle (long-run average).",
                "Downturn LGD is estimated as TTC LGD + 15% stress add-on (floored at product-level minimums).",
                "Collateral values are current market values, not origination values.",
            ],
            known_limitations=[
                "Thin post-default data for BNPL products; estimates have wider CIs.",
                "Does not explicitly model cure rates (re-performing loans).",
                "Collateral haircuts not modelled — external haircut table should be applied post-prediction.",
            ],
        )
        super().__init__(metadata)

        self.model_type = model_type
        self.apply_downturn_floor = apply_downturn_floor
        self.random_state = random_state

        self._pipeline: Pipeline | None = None
        self._train_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "LGDModel":
        """
        Train the LGD model.

        Parameters
        ----------
        X : DataFrame with REQUIRED_FEATURES columns
        y : Realised LGD ∈ [0, 1]  (1 = total loss, 0 = full recovery)
        """
        X_feat = X[REQUIRED_FEATURES].copy()

        # Clip y to avoid Beta distribution boundary issues
        y_clipped = np.clip(y, 0.001, 0.999)

        if self.model_type == "beta_regression":
            # TweedieRegressor with power=0 approximates Beta regression
            # For production, swap in a true Beta GLM (statsmodels or custom)
            base = TweedieRegressor(power=0, alpha=0.1, link="log", max_iter=500)
            self._pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", base),
            ])
        else:
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError(
                    "xgboost required for model_type='xgboost'. pip install xgboost"
                ) from exc
            base = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=self.random_state,
                verbosity=0,
            )
            self._pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", base),
            ])

        self._pipeline.fit(X_feat, y_clipped)

        preds = np.clip(self._pipeline.predict(X_feat), 0, 1)
        mae = mean_absolute_error(y_clipped, preds)
        rmse = np.sqrt(mean_squared_error(y_clipped, preds))
        self._train_metrics = {
            "train_mae":  round(float(mae), 5),
            "train_rmse": round(float(rmse), 5),
            "mean_lgd":   round(float(y_clipped.mean()), 5),
            "train_size": len(y),
        }
        logger.info(
            "[%s] fit complete — MAE=%.4f  RMSE=%.4f  mean_LGD=%.4f  n=%d",
            self.metadata.model_id,
            mae, rmse, y_clipped.mean(), len(y),
        )
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: Any) -> dict[str, Any]:
        X_df = self._coerce_to_dataframe(X)
        X_feat = X_df[REQUIRED_FEATURES]

        lgd_raw = float(np.clip(self._pipeline.predict(X_feat)[0], 0.0, 1.0))

        # Downturn LGD: 15% stress add-on, floored by product type
        product_code = int(X_feat["product_type_code"].iloc[0])
        dt_floor = _DOWNTURN_FLOORS.get(product_code, 0.45) if self.apply_downturn_floor else 0.0
        lgd_downturn = float(np.clip(max(lgd_raw * 1.15, dt_floor), 0.0, 1.0))

        # CI via residual standard deviation proxy
        sigma = max(0.03, lgd_raw * (1 - lgd_raw) * 0.4)
        lgd_lo = float(np.clip(lgd_raw - 1.96 * sigma, 0.0, 1.0))
        lgd_hi = float(np.clip(lgd_raw + 1.96 * sigma, 0.0, 1.0))

        return {
            "lgd_estimate":    round(lgd_raw,      5),
            "lgd_downturn":    round(lgd_downturn, 5),
            "recovery_rate":   round(1 - lgd_raw,  5),
            "lgd_lower_95":    round(lgd_lo,        5),
            "lgd_upper_95":    round(lgd_hi,        5),
            "model_type":      self.model_type,
            "train_metrics":   self._train_metrics,
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        self._check_fitted()
        try:
            reg = self._pipeline.named_steps["reg"]
            if self.model_type == "xgboost":
                imp = reg.feature_importances_.tolist()
            else:
                imp = np.abs(reg.coef_).tolist()
            total = sum(imp) or 1.0
            return dict(
                sorted(
                    zip(REQUIRED_FEATURES, [round(v / total, 5) for v in imp]),
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

        if (df["loan_amount"] <= 0).any():
            errors.append("loan_amount must be > 0.")
        if df["secured_flag"].isin([0, 1]).sum() != len(df):
            warnings.append("secured_flag contains values other than 0/1.")

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
        if isinstance(X, (list, np.ndarray)):
            return pd.DataFrame(X, columns=REQUIRED_FEATURES)
        raise TypeError(f"Unsupported input type: {type(X)}")
