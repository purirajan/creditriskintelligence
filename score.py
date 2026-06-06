"""
routes/score.py
===============
Scoring endpoints: single application, batch portfolio, full ECL.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, status

from ..schemas import (
    BatchScoringRequest,
    BatchScoringResponse,
    ECLRequest,
    ECLResponse,
    GovernanceBlock,
    ScoringRequest,
    ScoringResponse,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Governance block helper
# ---------------------------------------------------------------------------

def _governance(model_id: str, version: str, app_id: str | None, reg_use: list[str]) -> GovernanceBlock:
    return GovernanceBlock(
        model_id=model_id,
        model_version=version,
        record_id=f"{model_id}-{uuid.uuid4().hex[:10]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        application_id=app_id,
        regulatory_use=reg_use,
        risk_tier="High",
        environment="production",
    )


# ---------------------------------------------------------------------------
# PD estimate helpers
# ---------------------------------------------------------------------------

def _compute_pd(req: ScoringRequest) -> float:
    """
    Heuristic PD function.
    Replace with: model_registry.get('PD-CONSUMER-001').predict(X)
    """
    pd = 0.05
    pd += (750 - req.fico_score) * 0.0003
    pd += req.dti_ratio * 0.12
    pd += req.utilization_rate * 0.08
    pd += req.delinquency_count * 0.06
    pd -= min(req.months_on_book, 60) * 0.0003
    pd -= req.income_verified * 0.015
    return float(np.clip(pd, 0.001, 0.999))


def _pd_to_grade(pd: float) -> str:
    for threshold, grade in [
        (0.002, "AAA"), (0.005, "AA"), (0.010, "A"), (0.020, "A-"),
        (0.050, "BBB"), (0.100, "BB"), (0.200, "B"), (0.300, "CCC"),
    ]:
        if pd < threshold:
            return grade
    return "D"


def _decision(pd: float) -> str:
    if pd < 0.05:
        return "APPROVE"
    if pd < 0.15:
        return "REVIEW"
    return "DECLINE"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/score",
    response_model=ScoringResponse,
    summary="Score a single credit application",
    description=(
        "Returns PD estimate, 95% confidence interval, Basel risk grade, "
        "and decision recommendation. Fully audit-logged per SR 11-7."
    ),
)
def score_application(request: ScoringRequest) -> ScoringResponse:
    pd = _compute_pd(request)
    sigma = max(0.008, pd * (1 - pd) * 0.35)

    return ScoringResponse(
        application_id=request.application_id,
        pd_estimate=round(pd, 6),
        pd_lower_95=round(float(np.clip(pd - 1.96 * sigma, 0, 1)), 6),
        pd_upper_95=round(float(np.clip(pd + 1.96 * sigma, 0, 1)), 6),
        risk_grade=_pd_to_grade(pd),
        decision_recommendation=_decision(pd),
        governance=_governance(
            "PD-CONSUMER-001", "1.2.0",
            request.application_id,
            ["Basel IRB", "CECL", "CCAR"],
        ),
    )


@router.post(
    "/score/batch",
    response_model=BatchScoringResponse,
    summary="Score a portfolio of applications",
    description="Accepts up to 10,000 applications. Returns per-application scores and summary statistics.",
)
def score_batch(request: BatchScoringRequest) -> BatchScoringResponse:
    results: list[dict[str, Any]] = []
    pd_values: list[float] = []

    for app in request.applications:
        pd = _compute_pd(app)
        pd_values.append(pd)
        sigma = max(0.008, pd * (1 - pd) * 0.35)
        results.append({
            "application_id": app.application_id,
            "pd_estimate": round(pd, 6),
            "pd_lower_95": round(float(np.clip(pd - 1.96 * sigma, 0, 1)), 6),
            "pd_upper_95": round(float(np.clip(pd + 1.96 * sigma, 0, 1)), 6),
            "risk_grade": _pd_to_grade(pd),
            "decision_recommendation": _decision(pd),
        })

    pd_arr = np.array(pd_values)
    grade_counts: dict[str, int] = {}
    for pd in pd_values:
        g = _pd_to_grade(pd)
        grade_counts[g] = grade_counts.get(g, 0) + 1

    summary: dict[str, Any] = {
        "mean_pd":     round(float(pd_arr.mean()), 6),
        "median_pd":   round(float(np.median(pd_arr)), 6),
        "p95_pd":      round(float(np.percentile(pd_arr, 95)), 6),
        "approve_rate":round(float((pd_arr < 0.05).mean()), 4),
        "review_rate": round(float(((pd_arr >= 0.05) & (pd_arr < 0.15)).mean()), 4),
        "decline_rate":round(float((pd_arr >= 0.15).mean()), 4),
        "grade_distribution": grade_counts,
    }

    return BatchScoringResponse(
        portfolio_id=request.portfolio_id,
        total=len(results),
        results=results,
        summary=summary,
    )


@router.post(
    "/score/ecl",
    response_model=ECLResponse,
    summary="Full ECL calculation: PD × LGD × EAD",
    description=(
        "Runs all three component models (PD, LGD, EAD) and returns "
        "the 12-month Expected Credit Loss per CECL / IFRS 9."
    ),
)
def score_ecl(request: ECLRequest) -> ECLResponse:
    # PD
    pd = _compute_pd(request.pd_inputs)

    # LGD (simplified; swap in LGDModel.predict())
    ltv = request.lgd_inputs.origination_ltv
    secured = request.lgd_inputs.secured_flag
    lgd = 0.60 - secured * 0.25 - max(0, (1 - ltv)) * 0.10
    lgd = float(np.clip(lgd, 0.10, 0.90))

    # EAD (simplified; swap in EADModel.predict())
    balance = request.ead_inputs.current_balance
    limit = request.ead_inputs.credit_limit
    ccf = 0.75 if limit > balance else 0.0
    ead = balance + ccf * max(limit - balance, 0)

    ecl_12m = round(float(pd * lgd * ead), 2)
    # Lifetime ECL approximation: assume 3-year avg remaining life, PD increases at 1.5× per year
    ecl_lifetime = round(float(sum(
        (pd * (1.5 ** t)) * lgd * ead
        for t in range(3)
    )), 2)

    return ECLResponse(
        application_id=request.application_id,
        pd_estimate=round(pd, 6),
        lgd_estimate=round(lgd, 5),
        ead_estimate=round(ead, 2),
        ecl_12_month=ecl_12m,
        ecl_lifetime=ecl_lifetime,
        risk_grade=_pd_to_grade(pd),
        governance=_governance(
            "ECL-COMPOSITE-001", "1.0.0",
            request.application_id,
            ["CECL", "IFRS 9", "Basel IRB"],
        ),
    )
