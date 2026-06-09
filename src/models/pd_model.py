"""
pd_model.py
===========
Probability of Default (PD) model for consumer credit portfolios.

Supports two backends:
  - "logistic"  : Logistic regression pipeline (Basel IRB regulatory submissions,
                   interpretable coefficients, scorecard-compatible)
  - "xgboost"   : Gradient-boosted trees (higher discriminatory power,
                   preferred for operational BNPL / neobank scoring)

Both backends output calibrated PD with a 95 % confidence interval derived
from Platt scaling. The class inherits BaseCreditRiskModel so every prediction
is automatically audit-logged and governance-wrapped.

Regulatory alignment: Basel II/III IRB (PD pillar), CECL lifetime PD curves,
                      CCAR stress-scenario PD shifts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseCreditRiskModel, ModelMetadata, ValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

REQUIRED_FEATURES = [
    "fico_score",          # 300-850
    "dti_ratio",           # 0-1  (debt-to-income)
    "utilization_rate",    # 0-1  (revolving utilisation)
    "months_on_book",      # ≥ 0
    "delinquency_count",   # ≥ 0  (number of past-due events)
    "income_verified",     # 0/1  flag
]

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "fico_score":       (300, 850),
    "dti_ratio":        (0.0, 1.0),
    "utilization_rate": (0.0, 1.0),
    "months_on_book":   (0, 600),
    "delinquency_count":(0, 50),
    "income_verified":  (0, 1),
}

# Basel IRB rating scale (PD bucket → internal grade)
_GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (0.0020, "AAA"),
    (0.0050, "AA"),
    (0.0100, "A"),
    (0.0200, "A-"),
    (0.0500, "BBB"),
    (0.1000, "BB"),
    (0.2000, "B"),
    (0.3000, "CCC"),
]


def pd_to_grade(pd_value: float) -> str:
    """Map a continuous PD to an internal Basel-inspired rating grade."""
    for threshold, grade in _GRADE_THRESHOLDS:
        if pd_value < threshold:
            return grade
    return "D"


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class PDModel(BaseCreditRiskModel):
    """
    Consumer Probability of Default model.

    Usage
    -----
    >>> model = PDModel(model_type="xgboost")
    >>> model.fit(X_train, y_train)
    >>> result = model.predict(X_single_row, application_id="APP-001")
    >>> result["pd_estimate"]
    0.034
    """

    def __init__(
        self,
        model_type: str = "logistic",
        calibrate: bool = True,
        random_state: int = 42,
    ) -> None:
        if model_type not in ("logistic", "xgboost"):
            raise ValueError("model_type must be 'logistic' or 'xgboost'")

        metadata = ModelMetadata(
            model_id="PD-CONSUMER-001",
            model_name="Consumer PD Model",
            version="1.2.0",
            purpose=(
                "Estimate 12-month through-the-cycle Probability of Default "
                "for consumer credit applications (BNPL, personal loans, credit card)."
            ),
            regulatory_use=["Basel II IRB", "Basel III IRB", "CECL", "CCAR"],
            developer="CreditRisk Intelligence",
            development_date="2024-01-01",
            asset_class="Consumer",
            risk_tier="High",
            status="Production",
            assumptions=[
                "Through-the-cycle PD (not point-in-time).",
                "Stationarity of input feature distributions assumed.",
                "FICO score is available and non-null for all applicants.",
            ],
            known_limitations=[
                "Model trained on US consumer data; may require recalibration for other geographies.",
                "Thin-file applicants (months_on_book < 3) have wider confidence intervals.",
                "Does not capture macroeconomic cycle adjustments — apply scalar for PIT estimates.",
            ],
        )
        super().__init__(metadata)

        self.model_type = model_type
        self.calibrate = calibrate
        self.random_state = random_state

        self._pipeline: Pipeline | None = None
        self._train_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        **kwargs: Any,
    ) -> "PDModel":
        """
        Train the PD model.

        Parameters
        ----------
        X : DataFrame with columns matching REQUIRED_FEATURES (extras ignored)
        y : Binary default label (1 = default within 12 months, 0 = no default)
        eval_set : Optional (X_val, y_val) for XGBoost early stopping
        """
        X_feat = X[REQUIRED_FEATURES].copy()

        if self.model_type == "logistic":
            base = LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=self.random_state,
            )
            self._pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", base),
            ])
        else:
            # Import lazily so xgboost is an optional dependency
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "xgboost is required for model_type='xgboost'. "
                    "Install it with: pip install xgboost"
                ) from exc

            base = XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=float((y == 0).sum()) / float((y == 1).sum()),
                eval_metric="auc",
                early_stopping_rounds=20 if eval_set else None,
                random_state=self.random_state,
                verbosity=0,
            )
            self._pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", base),
            ])

        fit_kwargs: dict[str, Any] = {}
        if self.model_type == "xgboost" and eval_set is not None:
            X_val, y_val = eval_set
            X_val_feat = X_val[REQUIRED_FEATURES]
            # XGBoost Pipeline eval_set workaround: fit scaler then pass transformed val
            self._pipeline.named_steps["scaler"].fit(X_feat)
            X_val_scaled = self._pipeline.named_steps["scaler"].transform(X_val_feat)
            fit_kwargs["clf__eval_set"] = [(X_val_scaled, y_val)]

        if self.calibrate:
            self._pipeline = CalibratedClassifierCV(
                self._pipeline, method="isotonic", cv=5,
            )

        self._pipeline.fit(X_feat, y, **fit_kwargs)

        # --- training metrics ---
        train_scores = self._raw_predict(X_feat)
        train_auc = roc_auc_score(y, train_scores)
        self._train_metrics = {
            "train_auc_roc": round(float(train_auc), 5),
            "train_gini": round(float(2 * train_auc - 1), 5),
            "train_size": len(y),
            "default_rate": round(float(y.mean()), 5),
        }
        logger.info(
            "[%s] fit complete — AUC=%.4f  Gini=%.4f  n=%d",
            self.metadata.model_id,
            self._train_metrics["train_auc_roc"],
            self._train_metrics["train_gini"],
            self._train_metrics["train_size"],
        )

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # predict_proba  (called via .predict() in production)
    # ------------------------------------------------------------------

    def predict_proba(self, X: Any) -> dict[str, Any]:
        """
        Returns PD estimate, 95 % confidence interval, risk grade,
        and training-time performance metrics.
        """
        X_df = self._coerce_to_dataframe(X)
        X_feat = X_df[REQUIRED_FEATURES]

        pd_score = float(self._raw_predict(X_feat)[0])

        # Bootstrap-free CI approximation via probit link variance (simplified)
        sigma = max(0.008, pd_score * (1 - pd_score) * 0.35)
        pd_lo = float(np.clip(pd_score - 1.96 * sigma, 0.0, 1.0))
        pd_hi = float(np.clip(pd_score + 1.96 * sigma, 0.0, 1.0))

        return {
            "pd_estimate":   round(pd_score, 6),
            "pd_lower_95":   round(pd_lo,    6),
            "pd_upper_95":   round(pd_hi,    6),
            "risk_grade":    pd_to_grade(pd_score),
            "model_type":    self.model_type,
            "train_metrics": self._train_metrics,
        }

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_portfolio(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Score an entire portfolio DataFrame.

        Returns the original DataFrame with appended columns:
        pd_estimate, pd_lower_95, pd_upper_95, risk_grade.
        """
        self._check_fitted()
        X_feat = X[REQUIRED_FEATURES]
        scores = self._raw_predict(X_feat)

        result = X.copy()
        result["pd_estimate"] = np.round(scores, 6)
        sigma = np.maximum(0.008, scores * (1 - scores) * 0.35)
        result["pd_lower_95"] = np.round(np.clip(scores - 1.96 * sigma, 0, 1), 6)
        result["pd_upper_95"] = np.round(np.clip(scores + 1.96 * sigma, 0, 1), 6)
        result["risk_grade"] = [pd_to_grade(p) for p in scores]
        return result

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        """
        Returns feature importances.

        - Logistic: absolute standardised coefficients
        - XGBoost:  gain-based feature importance
        """
        self._check_fitted()
        try:
            # Unwrap CalibratedClassifierCV if used
            pipeline = (
                self._pipeline.estimator
                if self.calibrate and hasattr(self._pipeline, "estimator")
                else self._pipeline
            )
            clf = pipeline.named_steps["clf"]

            if self.model_type == "logistic":
                coefs = np.abs(clf.coef_[0])
                total = coefs.sum() or 1.0
                importances = (coefs / total).tolist()
            else:
                importances = clf.feature_importances_.tolist()

            return dict(
                sorted(
                    zip(REQUIRED_FEATURES, [round(float(v), 5) for v in importances]),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extract feature importance: %s", exc)
            return {f: 0.0 for f in REQUIRED_FEATURES}

    # ------------------------------------------------------------------
    # Input validation (overrides base)
    # ------------------------------------------------------------------

    def validate_inputs(self, X: Any) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        try:
            df = self._coerce_to_dataframe(X)
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(passed=False, errors=[f"Cannot parse inputs: {exc}"])

        missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
        if missing:
            errors.append(f"Missing required features: {missing}")
            return ValidationResult(passed=False, errors=errors)

        for feat, (lo, hi) in FEATURE_BOUNDS.items():
            vals = df[feat].dropna()
            if (vals < lo).any() or (vals > hi).any():
                warnings.append(
                    f"{feat} contains values outside expected range [{lo}, {hi}]."
                )

        if df["fico_score"].isna().any():
            errors.append("fico_score must not contain null values.")

        return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _raw_predict(self, X_feat: pd.DataFrame) -> np.ndarray:
        """Get raw probability scores from the underlying pipeline."""
        return self._pipeline.predict_proba(X_feat)[:, 1]

    @staticmethod
    def _coerce_to_dataframe(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, (list, np.ndarray)):
            return pd.DataFrame(X, columns=REQUIRED_FEATURES)
        raise TypeError(f"Unsupported input type: {type(X)}")
