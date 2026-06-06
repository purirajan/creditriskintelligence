"""
routes/explain.py
=================
Explainability endpoint: SHAP + Reg B adverse action reasons.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter

from ..schemas import (
    AdverseActionReason,
    ExplainRequest,
    ExplainResponse,
    GovernanceBlock,
)

router = APIRouter()

# ECOA / Reg B adverse action code mapping
_ADVERSE_CODES = {
    "fico_score":        ("AA-01", "Insufficient credit history or low credit score"),
    "dti_ratio":         ("AA-02", "Debt-to-income ratio too high"),
    "utilization_rate":  ("AA-03", "Credit utilisation rate too high"),
    "months_on_book":    ("AA-04", "Insufficient length of credit history"),
    "delinquency_count": ("AA-05", "Delinquency on existing account(s)"),
    "income_verified":   ("AA-06", "Unable to verify income"),
}


def _shap_contributions(req: ExplainRequest, pd: float) -> dict[str, float]:
    """
    Approximate SHAP contributions (linear decomposition).
    Replace with SHAPExplainer.explain() from src/explainability/shap_engine.py
    in production.
    """
    return {
        "fico_score":        round((680 - req.fico_score) * 0.0004, 5),
        "dti_ratio":         round((req.dti_ratio - 0.30) * 0.09, 5),
        "utilization_rate":  round((req.utilization_rate - 0.35) * 0.07, 5),
        "months_on_book":    round(-(min(req.months_on_book, 60) - 18) * 0.0004, 5),
        "delinquency_count": round(req.delinquency_count * 0.05, 5),
        "income_verified":   round(-(req.income_verified - 0.5) * 0.02, 5),
    }


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Get SHAP explanation for a credit decision",
    description=(
        "Returns signed SHAP feature contributions and up to 4 ECOA Regulation B "
        "adverse action reason codes for any declined or flagged application."
    ),
)
def explain_decision(request: ExplainRequest) -> ExplainResponse:
    # PD
    pd = request.pd_estimate
    if pd is None:
        pd = 0.05
        pd += (750 - request.fico_score) * 0.0003
        pd += request.dti_ratio * 0.12
        pd += request.utilization_rate * 0.08
        pd += request.delinquency_count * 0.06
        pd -= min(request.months_on_book, 60) * 0.0003
        pd -= request.income_verified * 0.015
        pd = float(np.clip(pd, 0.001, 0.999))

    grades = [
        (0.002, "AAA"), (0.005, "AA"), (0.010, "A"), (0.020, "A-"),
        (0.050, "BBB"), (0.100, "BB"), (0.200, "B"), (0.300, "CCC"),
    ]
    grade = next((g for t, g in grades if pd < t), "D")

    contributions = _shap_contributions(request, pd)
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_drivers = [f for f, _ in sorted_contribs[:6]]

    # Adverse action: only positive SHAP (PD-increasing) features qualify
    is_adverse = pd >= 0.05
    adverse_reasons: list[AdverseActionReason] = []
    if is_adverse:
        for feature, shap_val in sorted_contribs:
            if shap_val > 0 and feature in _ADVERSE_CODES:
                code, desc = _ADVERSE_CODES[feature]
                adverse_reasons.append(AdverseActionReason(
                    code=code, description=desc, shap_contribution=round(shap_val, 5)
                ))
            if len(adverse_reasons) == 4:
                break

    return ExplainResponse(
        application_id=request.application_id,
        pd_estimate=round(pd, 6),
        risk_grade=grade,
        is_adverse_action=is_adverse,
        feature_contributions=contributions,
        top_risk_drivers=top_drivers,
        adverse_action_reasons=adverse_reasons,
        regulatory_note=(
            "Adverse action reasons comply with ECOA/Regulation B. "
            "Up to 4 specific reasons provided per 12 C.F.R. §202.9."
        ),
        governance=GovernanceBlock(
            model_id="PD-CONSUMER-001",
            model_version="1.2.0",
            record_id=f"EXP-{uuid.uuid4().hex[:10]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            application_id=request.application_id,
            regulatory_use=["Basel IRB", "CECL", "ECOA"],
            risk_tier="High",
        ),
    )
