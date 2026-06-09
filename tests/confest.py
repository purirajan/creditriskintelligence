"""
conftest.py
===========
Shared pytest fixtures for all test modules.

Fixtures defined here are automatically available to every test file
without needing to import them.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# Sample data fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_application():
    """Single borrower application — good credit profile (should APPROVE)."""
    return {
        "fico_score":        720,
        "dti_ratio":         0.28,
        "utilization_rate":  0.30,
        "months_on_book":    36,
        "delinquency_count": 0,
        "income_verified":   1,
    }


@pytest.fixture
def risky_application():
    """Single borrower application — poor credit profile (should DECLINE)."""
    return {
        "fico_score":        580,
        "dti_ratio":         0.55,
        "utilization_rate":  0.85,
        "months_on_book":    3,
        "delinquency_count": 2,
        "income_verified":   0,
    }


@pytest.fixture
def borderline_application():
    """Borderline application — should trigger REVIEW."""
    return {
        "fico_score":        640,
        "dti_ratio":         0.42,
        "utilization_rate":  0.55,
        "months_on_book":    12,
        "delinquency_count": 1,
        "income_verified":   1,
    }


@pytest.fixture
def sample_portfolio_df():
    """Small synthetic portfolio DataFrame for batch and monitoring tests."""
    np.random.seed(42)
    n = 200

    default_flags = np.random.binomial(1, 0.08, n)

    return pd.DataFrame({
        "fico_score":        np.clip(np.random.normal(680, 60, n), 300, 850),
        "dti_ratio":         np.clip(np.random.normal(0.35, 0.12, n), 0, 1),
        "utilization_rate":  np.clip(np.random.normal(0.40, 0.18, n), 0, 1),
        "months_on_book":    np.random.randint(1, 84, n),
        "delinquency_count": np.random.poisson(0.3, n),
        "income_verified":   np.random.binomial(1, 0.80, n),
        "default_flag":      default_flags,
    })


@pytest.fixture
def lgd_application():
    """Single LGD model input."""
    return {
        "collateral_value":       0.0,
        "loan_amount":            5000.0,
        "months_past_due":        3,
        "secured_flag":           0,
        "product_type_code":      2,      # BNPL
        "origination_ltv":        0.0,
        "time_in_default_months": 2,
    }


@pytest.fixture
def ead_application():
    """Single EAD model input."""
    return {
        "current_balance":          1200.0,
        "credit_limit":             5000.0,
        "utilization_rate":         0.24,
        "months_to_maturity":       0,
        "product_type_code":        1,    # credit card
        "months_since_last_draw":   2,
        "payment_behaviour_score":  72.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# API client fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def api_client():
    """FastAPI test client — no real server needed."""
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """API key header for authenticated endpoints."""
    return {"X-API-Key": "dev-key-replace-in-production"}
