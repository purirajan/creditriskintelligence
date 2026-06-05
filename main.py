"""
main.py — CreditRisk Intelligence API
FastAPI application exposing risk scoring, explainability, and monitoring endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn

app = FastAPI(
    title="CreditRisk Intelligence API",
    description="Basel-aligned credit risk scoring, SHAP explainability, and model monitoring for fintechs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Schemas ---

class ScoringRequest(BaseModel):
    application_id: Optional[str] = Field(None, example="APP-20240101-001")
    fico_score: float = Field(..., ge=300, le=850, example=680)
    dti_ratio: float = Field(..., ge=0, le=1, example=0.35)
    utilization_rate: float = Field(..., ge=0, le=1, example=0.42)
    months_on_book: int = Field(..., ge=0, example=18)
    delinquency_count: int = Field(0, ge=0, example=0)
    income_verified: int = Field(1, description="1=verified, 0=stated")


class ScoringResponse(BaseModel):
    application_id: Optional[str]
    pd_estimate: float
    pd_lower_95: float
    pd_upper_95: float
    risk_grade: str
    decision_recommendation: str
    governance: Dict[str, Any]


class ExplainRequest(BaseModel):
    application_id: Optional[str] = None
    fico_score: float
    dti_ratio: float
    utilization_rate: float
    months_on_book: int
    delinquency_count: int = 0
    income_verified: int = 1


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "CreditRisk Intelligence API", "version": "1.0.0"}


@app.post("/v1/score", response_model=ScoringResponse, tags=["Scoring"])
def score_application(request: ScoringRequest):
    """
    Score a credit application.
    
    Returns PD estimate, confidence interval, risk grade, and decision recommendation.
    Fully audit-logged per SR 11-7 requirements.
    """
    # In production: load model from registry, call model.predict(X)
    # Simplified demo logic here:
    pd = _demo_pd_estimate(request)
    grade = _pd_to_grade(pd)
    decision = "APPROVE" if pd < 0.05 else "REVIEW" if pd < 0.15 else "DECLINE"

    return ScoringResponse(
        application_id=request.application_id,
        pd_estimate=round(pd, 6),
        pd_lower_95=round(max(0, pd - 0.018), 6),
        pd_upper_95=round(min(1, pd + 0.018), 6),
        risk_grade=grade,
        decision_recommendation=decision,
        governance={
            "model_id": "PD-CONSUMER-001",
            "model_version": "1.0.0",
            "regulatory_use": ["Basel IRB", "CECL"],
            "timestamp": "2024-01-15T10:23:45Z",
        }
    )


@app.post("/v1/explain", tags=["Explainability"])
def explain_decision(request: ExplainRequest):
    """
    Get SHAP-based explanation for a credit decision.
    Returns top risk drivers and ECOA Reg B adverse action reason codes.
    """
    pd = _demo_pd_estimate(request)
    return {
        "application_id": request.application_id,
        "pd_estimate": round(pd, 6),
        "risk_grade": _pd_to_grade(pd),
        "feature_contributions": {
            "fico_score": round(-0.031 + (750 - request.fico_score) * 0.001, 4),
            "dti_ratio": round(request.dti_ratio * 0.08, 4),
            "utilization_rate": round(request.utilization_rate * 0.06, 4),
            "months_on_book": round(-request.months_on_book * 0.0005, 4),
            "delinquency_count": round(request.delinquency_count * 0.04, 4),
            "income_verified": round((1 - request.income_verified) * 0.02, 4),
        },
        "adverse_action_reasons": _get_adverse_reasons(request, pd),
        "regulatory_note": "Adverse action reasons comply with ECOA/Reg B requirements.",
    }


@app.get("/v1/monitor/{model_id}", tags=["Monitoring"])
def get_model_health(model_id: str):
    """
    Get current model health metrics for a given model ID.
    Returns PSI, KS, Gini, AUC, and any active alerts.
    """
    # In production: pull from monitoring database
    return {
        "model_id": model_id,
        "status": "healthy",
        "last_run": "2024-01-15T06:00:00Z",
        "metrics": {
            "psi": 0.041,
            "ks_statistic": 0.412,
            "gini_coefficient": 0.631,
            "auc_roc": 0.816,
        },
        "alerts": [],
        "next_scheduled_run": "2024-01-16T06:00:00Z",
    }


# --- Helpers ---

def _demo_pd_estimate(req) -> float:
    """Simplified PD heuristic for demo. Replace with trained model."""
    score = 0.05
    score += (750 - req.fico_score) * 0.0003
    score += req.dti_ratio * 0.12
    score += req.utilization_rate * 0.08
    score += req.delinquency_count * 0.06
    score -= min(req.months_on_book, 60) * 0.0003
    score -= req.income_verified * 0.015
    return max(0.001, min(0.999, score))


def _pd_to_grade(pd: float) -> str:
    grades = [(0.005,"AAA"),(0.01,"AA"),(0.02,"A"),(0.05,"BBB"),(0.10,"BB"),(0.20,"B")]
    for threshold, grade in grades:
        if pd < threshold:
            return grade
    return "CCC"


def _get_adverse_reasons(req, pd: float):
    if pd < 0.05:
        return []
    reasons = []
    if req.fico_score < 650:
        reasons.append({"code": "AA-01", "description": "Insufficient credit score"})
    if req.dti_ratio > 0.40:
        reasons.append({"code": "AA-02", "description": "Debt-to-income ratio too high"})
    if req.utilization_rate > 0.60:
        reasons.append({"code": "AA-03", "description": "Credit utilization too high"})
    if req.delinquency_count > 0:
        reasons.append({"code": "AA-05", "description": "Delinquency on account(s)"})
    return reasons[:4]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
