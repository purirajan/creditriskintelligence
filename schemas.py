"""
schemas.py
==========
Pydantic v2 request and response schemas for the CreditRisk Intelligence API.

Design principles:
  - Every request schema validates ranges at the Pydantic layer (before model inference).
  - Every response schema includes a _governance block for audit traceability.
  - All monetary amounts are in USD (float, rounded to cents).
  - All rates/fractions are ∈ [0, 1].
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class GovernanceBlock(BaseModel):
    model_id: str
    model_version: str
    record_id: str
    timestamp: str
    application_id: str | None = None
    regulatory_use: list[str] = []
    risk_tier: str = "High"
    environment: str = "production"


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class ScoringRequest(BaseModel):
    """
    Input schema for POST /v1/score.
    All six features are required for the PD model.
    """
    application_id: str | None = Field(
        None, description="Client-supplied application reference", example="APP-20240101-001"
    )
    fico_score: float = Field(
        ..., ge=300, le=850,
        description="FICO credit score (300–850)",
        example=680,
    )
    dti_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description="Debt-to-income ratio (0–1)",
        example=0.35,
    )
    utilization_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Revolving credit utilisation rate (0–1)",
        example=0.42,
    )
    months_on_book: int = Field(
        ..., ge=0, le=600,
        description="Months since account origination",
        example=18,
    )
    delinquency_count: int = Field(
        0, ge=0, le=50,
        description="Number of past-due events in credit history",
        example=0,
    )
    income_verified: int = Field(
        1,
        description="Income verification flag: 1 = verified, 0 = stated",
        example=1,
    )

    @field_validator("income_verified")
    @classmethod
    def validate_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("income_verified must be 0 or 1")
        return v


class ScoringResponse(BaseModel):
    application_id: str | None
    pd_estimate: float = Field(..., description="Probability of Default ∈ [0, 1]")
    pd_lower_95: float = Field(..., description="95% CI lower bound")
    pd_upper_95: float = Field(..., description="95% CI upper bound")
    risk_grade: str = Field(..., description="Internal Basel-inspired rating (AAA–D)")
    decision_recommendation: str = Field(..., description="APPROVE | REVIEW | DECLINE")
    governance: GovernanceBlock


# ---------------------------------------------------------------------------
# LGD scoring
# ---------------------------------------------------------------------------

class LGDRequest(BaseModel):
    application_id: str | None = None
    collateral_value: float = Field(..., ge=0, description="Current collateral value (USD)")
    loan_amount: float = Field(..., gt=0, description="Outstanding loan balance (USD)")
    months_past_due: int = Field(..., ge=0, le=120)
    secured_flag: int = Field(..., description="1 = secured, 0 = unsecured")
    product_type_code: int = Field(..., ge=0, le=10, description="0=personal, 1=auto, 2=BNPL, 3=card")
    origination_ltv: float = Field(..., ge=0, le=3.0, description="LTV at origination (0 for unsecured)")
    time_in_default_months: int = Field(..., ge=0, le=120)


class LGDResponse(BaseModel):
    application_id: str | None
    lgd_estimate: float
    lgd_downturn: float
    recovery_rate: float
    lgd_lower_95: float
    lgd_upper_95: float
    governance: GovernanceBlock


# ---------------------------------------------------------------------------
# EAD scoring
# ---------------------------------------------------------------------------

class EADRequest(BaseModel):
    application_id: str | None = None
    current_balance: float = Field(..., ge=0, description="Current drawn balance (USD)")
    credit_limit: float = Field(..., gt=0, description="Total facility limit (USD)")
    utilization_rate: float = Field(..., ge=0, le=1)
    months_to_maturity: int = Field(..., ge=0, le=360)
    product_type_code: int = Field(..., ge=0, le=10)
    months_since_last_draw: int = Field(..., ge=0, le=120)
    payment_behaviour_score: float = Field(..., ge=0, le=100)


class EADResponse(BaseModel):
    application_id: str | None
    ccf_estimate: float
    ccf_with_floor: float
    ead_estimate: float
    ead_regulatory: float
    current_balance: float
    undrawn_commitment: float
    regulatory_ccf_floor: float
    governance: GovernanceBlock


# ---------------------------------------------------------------------------
# Full ECL (PD × LGD × EAD)
# ---------------------------------------------------------------------------

class ECLRequest(BaseModel):
    """Combined ECL request — all three component models scored together."""
    application_id: str | None = None
    pd_inputs: ScoringRequest
    lgd_inputs: LGDRequest
    ead_inputs: EADRequest


class ECLResponse(BaseModel):
    application_id: str | None
    pd_estimate: float
    lgd_estimate: float
    ead_estimate: float
    ecl_12_month: float = Field(..., description="ECL = PD × LGD × EAD (12-month)")
    ecl_lifetime: float = Field(..., description="Lifetime ECL (requires maturity schedule)")
    risk_grade: str
    governance: GovernanceBlock


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    """Input schema for POST /v1/explain."""
    application_id: str | None = None
    fico_score: float = Field(..., ge=300, le=850)
    dti_ratio: float = Field(..., ge=0, le=1)
    utilization_rate: float = Field(..., ge=0, le=1)
    months_on_book: int = Field(..., ge=0)
    delinquency_count: int = Field(0, ge=0)
    income_verified: int = Field(1)
    pd_estimate: float | None = Field(
        None,
        description="Pre-computed PD (optional; avoids re-scoring)",
    )


class AdverseActionReason(BaseModel):
    code: str = Field(..., example="AA-02")
    description: str = Field(..., example="Debt-to-income ratio too high")
    shap_contribution: float


class ExplainResponse(BaseModel):
    application_id: str | None
    pd_estimate: float
    risk_grade: str
    is_adverse_action: bool
    feature_contributions: dict[str, float]
    top_risk_drivers: list[str]
    adverse_action_reasons: list[AdverseActionReason]
    regulatory_note: str
    governance: GovernanceBlock


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

class ModelHealthResponse(BaseModel):
    model_id: str
    status: str
    last_run: str
    metrics: dict[str, float]
    alerts: list[dict[str, Any]] = []
    next_scheduled_run: str | None = None


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

class BatchScoringRequest(BaseModel):
    """Batch portfolio scoring — list of individual scoring requests."""
    portfolio_id: str | None = None
    applications: list[ScoringRequest] = Field(
        ..., min_length=1, max_length=10_000
    )


class BatchScoringResponse(BaseModel):
    portfolio_id: str | None
    total: int
    results: list[dict[str, Any]]
    summary: dict[str, float]  # e.g. mean_pd, grade_distribution
