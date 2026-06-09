"""
main.py
=======
CreditRisk Intelligence FastAPI application.

Endpoints:
  GET  /health                    — service health check
  POST /v1/score                  — real-time risk scoring (PD/LGD/EAD)
  POST /v1/explain                — SHAP explanation + Reg B adverse action
  GET  /v1/monitor/{model_id}     — model health metrics
  GET  /v1/models                 — registered model inventory
  POST /v1/portfolio/score        — batch portfolio scoring
  GET  /v1/audit/{model_id}       — audit log entries

All endpoints return JSON. Authentication handled via API key header
(X-API-Key) in production deployments.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from .routes.score import router as score_router
from .routes.explain import router as explain_router
from .routes.monitor import router as monitor_router
from .schemas import HealthResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key security (stub — replace with your auth provider)
# ---------------------------------------------------------------------------

API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

_VALID_KEYS: set[str] = {"dev-key-replace-in-production"}


async def verify_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """
    Validate API key from X-API-Key header.
    In production, replace with database lookup / OAuth2 / JWT.
    """
    if api_key is None or api_key not in _VALID_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )
    return api_key


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("CreditRisk Intelligence API starting up...")
    # TODO: load model registry, connect to DB, warm model cache
    yield
    logger.info("CreditRisk Intelligence API shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CreditRisk Intelligence API",
        description=(
            "Basel-aligned credit risk scoring, SHAP explainability, "
            "and SR 11-7 model governance for fintech lenders."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "Health",         "description": "Service health and readiness"},
            {"name": "Scoring",        "description": "Real-time and batch risk scoring"},
            {"name": "Explainability", "description": "SHAP explanations and Reg B adverse action"},
            {"name": "Monitoring",     "description": "Model drift and performance monitoring"},
            {"name": "Audit",          "description": "Governance audit trail"},
            {"name": "Models",         "description": "Model inventory and metadata"},
        ],
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],     # tighten to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(score_router,   prefix="/v1", tags=["Scoring"])
    app.include_router(explain_router, prefix="/v1", tags=["Explainability"])
    app.include_router(monitor_router, prefix="/v1", tags=["Monitoring"])

    # Health endpoint (no auth required)
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            service="CreditRisk Intelligence API",
            version="1.0.0",
        )

    # Model inventory (no auth required for discovery)
    @app.get("/v1/models", tags=["Models"])
    async def list_models() -> dict[str, Any]:
        return {
            "models": [
                {
                    "model_id": "PD-CONSUMER-001",
                    "name": "Consumer PD Model",
                    "version": "1.2.0",
                    "status": "Production",
                    "regulatory_use": ["Basel IRB", "CECL", "CCAR"],
                    "asset_class": "Consumer",
                    "risk_tier": "High",
                },
                {
                    "model_id": "LGD-CONSUMER-001",
                    "name": "Consumer LGD Model",
                    "version": "1.1.0",
                    "status": "Production",
                    "regulatory_use": ["Basel IRB", "CECL"],
                    "asset_class": "Consumer",
                    "risk_tier": "High",
                },
                {
                    "model_id": "EAD-CONSUMER-001",
                    "name": "Consumer EAD / CCF Model",
                    "version": "1.0.0",
                    "status": "Production",
                    "regulatory_use": ["Basel IRB", "CECL", "IFRS 9"],
                    "asset_class": "Consumer",
                    "risk_tier": "High",
                },
            ]
        }

    # Audit log endpoint
    @app.get("/v1/audit/{model_id}", tags=["Audit"])
    async def get_audit_log(
        model_id: str,
        limit: int = 100,
        _key: str = Depends(verify_api_key),
    ) -> dict[str, Any]:
        """
        Retrieve recent audit log entries for a model.
        Requires API key authentication.
        """
        # In production: query AuditTrail backend
        # Here we return a demo payload
        return {
            "model_id": model_id,
            "entries": [
                {
                    "event_id": "evt-0001",
                    "event_type": "prediction",
                    "timestamp": "2026-06-05T10:23:45Z",
                    "actor": "api",
                    "payload": {"application_id": "APP-001", "pd_estimate": 0.042},
                },
                {
                    "event_id": "evt-0002",
                    "event_type": "monitoring",
                    "timestamp": "2026-06-05T06:00:00Z",
                    "actor": "monitoring_service",
                    "payload": {"status": "healthy", "psi": 0.041},
                },
            ],
            "total": 2,
            "limit": limit,
        }

    return app


# Application instance (used by uvicorn)
app = create_app()
