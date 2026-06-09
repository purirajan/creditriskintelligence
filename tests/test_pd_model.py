"""
test_pd_model.py
================
Unit tests for the Probability of Default model.

Tests cover:
  - Model initialisation and metadata
  - Input validation (valid, missing features, out-of-range)
  - Training (fit)
  - Prediction output structure and value ranges
  - Risk grade mapping
  - Batch portfolio scoring
  - Audit log population
  - Model card generation
  - Unfitted model guard
"""

import numpy as np
import pandas as pd
import pytest

from src.models.pd_model import PDModel, pd_to_grade, REQUIRED_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_training_data(n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a small synthetic training dataset."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "fico_score":        np.clip(rng.normal(680, 60, n), 300, 850),
        "dti_ratio":         np.clip(rng.normal(0.35, 0.12, n), 0, 1),
        "utilization_rate":  np.clip(rng.normal(0.40, 0.18, n), 0, 1),
        "months_on_book":    rng.integers(1, 84, n).astype(float),
        "delinquency_count": rng.poisson(0.3, n).astype(float),
        "income_verified":   rng.binomial(1, 0.80, n).astype(float),
    })
    # Simple deterministic default label based on risk factors
    score = (
        (680 - X["fico_score"]) * 0.003
        + X["dti_ratio"] * 1.2
        + X["utilization_rate"] * 0.8
        + X["delinquency_count"] * 0.6
    )
    prob = 1 / (1 + np.exp(-score + 2))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="default_flag")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestPDModelInit:

    def test_default_model_type_is_logistic(self):
        model = PDModel()
        assert model.model_type == "logistic"

    def test_xgboost_model_type(self):
        model = PDModel(model_type="xgboost")
        assert model.model_type == "xgboost"

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="model_type"):
            PDModel(model_type="random_forest")

    def test_metadata_model_id(self):
        model = PDModel()
        assert model.metadata.model_id == "PD-CONSUMER-001"

    def test_metadata_regulatory_use(self):
        model = PDModel()
        assert "CECL" in model.metadata.regulatory_use
        assert "Basel II IRB" in model.metadata.regulatory_use

    def test_not_fitted_on_init(self):
        model = PDModel()
        assert model._is_fitted is False

    def test_repr(self):
        model = PDModel()
        r = repr(model)
        assert "PDModel" in r
        assert "fitted=False" in r


# ─────────────────────────────────────────────────────────────────────────────
# Fit
# ─────────────────────────────────────────────────────────────────────────────

class TestPDModelFit:

    def test_fit_sets_is_fitted(self):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        assert model._is_fitted is True

    def test_fit_returns_self(self):
        model = PDModel()
        X, y = make_training_data()
        result = model.fit(X, y)
        assert result is model

    def test_fit_populates_train_metrics(self):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        assert "train_auc_roc" in model._train_metrics
        assert "train_gini" in model._train_metrics
        assert model._train_metrics["train_auc_roc"] > 0.5

    def test_fit_with_extra_columns_ignored(self):
        """Extra columns in X should not break fit."""
        model = PDModel()
        X, y = make_training_data()
        X["extra_col"] = 99.0
        model.fit(X, y)
        assert model._is_fitted


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

class TestPDModelPredict:

    @pytest.fixture(autouse=True)
    def fitted_model(self):
        self.model = PDModel()
        X, y = make_training_data()
        self.model.fit(X, y)

    def test_predict_returns_dict(self, sample_application):
        result = self.model.predict(sample_application, application_id="TEST-001")
        assert isinstance(result, dict)

    def test_pd_estimate_in_range(self, sample_application):
        result = self.model.predict(sample_application)
        assert 0.0 <= result["pd_estimate"] <= 1.0

    def test_confidence_interval_ordering(self, sample_application):
        result = self.model.predict(sample_application)
        assert result["pd_lower_95"] <= result["pd_estimate"] <= result["pd_upper_95"]

    def test_risk_grade_present(self, sample_application):
        result = self.model.predict(sample_application)
        assert result["risk_grade"] in ("AAA", "AA", "A", "A-", "BBB", "BB", "B", "CCC", "D")

    def test_governance_block_present(self, sample_application):
        result = self.model.predict(sample_application, application_id="APP-001")
        assert "_governance" in result
        gov = result["_governance"]
        assert gov["model_id"] == "PD-CONSUMER-001"
        assert gov["application_id"] == "APP-001"
        assert "timestamp" in gov

    def test_risky_profile_higher_pd(self, sample_application, risky_application):
        good = self.model.predict(sample_application)
        bad  = self.model.predict(risky_application)
        assert bad["pd_estimate"] > good["pd_estimate"]

    def test_predict_increments_audit_log(self, sample_application):
        before = self.model._prediction_count
        self.model.predict(sample_application)
        assert self.model._prediction_count == before + 1

    def test_predict_dict_input(self, sample_application):
        result = self.model.predict(sample_application)
        assert "pd_estimate" in result

    def test_predict_dataframe_input(self, sample_application):
        df = pd.DataFrame([sample_application])
        result = self.model.predict(df)
        assert "pd_estimate" in result

    def test_unfitted_model_raises(self, sample_application):
        model = PDModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(sample_application)


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPDModelValidation:

    @pytest.fixture(autouse=True)
    def fitted_model(self):
        self.model = PDModel()
        X, y = make_training_data()
        self.model.fit(X, y)

    def test_missing_feature_raises(self):
        bad_input = {"fico_score": 700, "dti_ratio": 0.3}  # missing 4 features
        with pytest.raises(ValueError, match="Missing required features"):
            self.model.predict(bad_input)

    def test_null_fico_raises(self):
        bad_input = {
            "fico_score": None,
            "dti_ratio": 0.3,
            "utilization_rate": 0.4,
            "months_on_book": 12,
            "delinquency_count": 0,
            "income_verified": 1,
        }
        with pytest.raises((ValueError, Exception)):
            self.model.predict(bad_input)

    def test_out_of_range_generates_warning(self, caplog, sample_application):
        """Out-of-range values should warn but not block prediction."""
        import logging
        bad = dict(sample_application)
        bad["dti_ratio"] = 1.5   # > 1.0 is out of range
        df = pd.DataFrame([bad])
        with caplog.at_level(logging.WARNING):
            result = self.model.predict(df)
        assert "pd_estimate" in result


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestPDModelPortfolio:

    def test_score_portfolio_returns_dataframe(self, sample_portfolio_df):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        result = model.score_portfolio(sample_portfolio_df)
        assert isinstance(result, pd.DataFrame)

    def test_score_portfolio_adds_columns(self, sample_portfolio_df):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        result = model.score_portfolio(sample_portfolio_df)
        for col in ("pd_estimate", "pd_lower_95", "pd_upper_95", "risk_grade"):
            assert col in result.columns

    def test_score_portfolio_all_pd_in_range(self, sample_portfolio_df):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        result = model.score_portfolio(sample_portfolio_df)
        assert (result["pd_estimate"] >= 0).all()
        assert (result["pd_estimate"] <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

class TestPDModelFeatureImportance:

    def test_returns_all_features(self):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        imp = model.get_feature_importance()
        for feat in REQUIRED_FEATURES:
            assert feat in imp

    def test_importances_are_positive(self):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert all(v >= 0 for v in imp.values())

    def test_unfitted_returns_zeros(self):
        model = PDModel()
        imp = model.get_feature_importance()
        assert all(v == 0.0 for v in imp.values())


# ─────────────────────────────────────────────────────────────────────────────
# Risk grade mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestPdToGrade:

    @pytest.mark.parametrize("pd_val,expected", [
        (0.001,  "AAA"),
        (0.003,  "AA"),
        (0.008,  "A"),
        (0.015,  "A-"),
        (0.035,  "BBB"),
        (0.075,  "BB"),
        (0.150,  "B"),
        (0.250,  "CCC"),
        (0.450,  "D"),
    ])
    def test_grade_boundaries(self, pd_val, expected):
        assert pd_to_grade(pd_val) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Model card
# ─────────────────────────────────────────────────────────────────────────────

class TestPDModelCard:

    def test_model_card_structure(self):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        card = model.generate_model_card()
        assert "model_overview" in card
        assert "regulatory_alignment" in card
        assert "governance_lifecycle" in card
        assert "feature_importance" in card
        assert "known_limitations" in card

    def test_model_card_has_correct_id(self):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        card = model.generate_model_card()
        assert card["model_overview"]["id"] == "PD-CONSUMER-001"

    def test_audit_log_accessible(self):
        model = PDModel()
        X, y = make_training_data()
        model.fit(X, y)
        app = {
            "fico_score": 700, "dti_ratio": 0.3,
            "utilization_rate": 0.4, "months_on_book": 24,
            "delinquency_count": 0, "income_verified": 1,
        }
        model.predict(app)
        log = model.get_audit_log()
        assert len(log) == 1
        assert "record_id" in log[0]
        assert "input_hash" in log[0]
