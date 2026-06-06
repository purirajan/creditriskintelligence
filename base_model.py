"""
base_model.py
=============
Abstract base class for all CreditRisk Intelligence models.

Enforces SR 11-7 / Model Risk Management governance on every subclass:
  - Immutable audit logging on every prediction
  - Privacy-preserving input hashing
  - Auto-generated model cards
  - Pre/post-prediction hooks for monitoring integration
  - Standardised ModelMetadata contract

All production models inherit from BaseCreditRiskModel and call
super().__init__(metadata) before doing anything else.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    """
    Governance metadata required for every model.

    Mirrors the minimum fields expected by SR 11-7 model inventory systems
    and Basel model documentation standards.
    """
    model_id: str
    model_name: str
    version: str
    purpose: str                        # plain-language description of use case
    regulatory_use: list[str]           # e.g. ["CECL", "Basel IRB", "CCAR"]
    developer: str
    development_date: str               # ISO-8601 date string
    asset_class: str = "Consumer"
    risk_tier: str = "High"             # High / Medium / Low  (SR 11-7 §IV)
    status: str = "Production"          # Development|Validation|Production|Retired
    last_validation_date: str | None = None
    next_review_date: str | None = None
    model_owner: str | None = None
    approved_by: str | None = None
    approval_date: str | None = None
    assumptions: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PredictionRecord:
    """
    Immutable audit record written for every call to .predict().

    input_hash is a SHA-256 hex digest of the serialised inputs, enabling
    forensic reproducibility without storing raw PII.
    """
    record_id: str
    model_id: str
    model_version: str
    timestamp: str                      # UTC ISO-8601
    input_hash: str                     # SHA-256 of inputs (privacy-preserving)
    output: dict[str, Any]
    environment: str = "production"
    application_id: str | None = None


@dataclass
class ValidationResult:
    """Returned by validate_inputs(); failures block prediction."""
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseCreditRiskModel(ABC):
    """
    Abstract base for all CreditRisk Intelligence models.

    Subclasses must implement:
      - fit(X, y, **kwargs)
      - predict_proba(X) -> dict
      - get_feature_importance() -> dict[str, float]
      - validate_inputs(X) -> ValidationResult          (optional override)

    Call .predict() in production — never .predict_proba() directly —
    so that every output is audit-logged.
    """

    def __init__(self, metadata: ModelMetadata) -> None:
        self.metadata = metadata
        self._audit_log: list[PredictionRecord] = []
        self._prediction_count: int = 0
        self._is_fitted: bool = False
        logger.info(
            "Model initialised: %s v%s (tier=%s, status=%s)",
            metadata.model_id, metadata.version,
            metadata.risk_tier, metadata.status,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: Any, y: Any, **kwargs: Any) -> "BaseCreditRiskModel":
        """Train the model. Must set self._is_fitted = True."""

    @abstractmethod
    def predict_proba(self, X: Any) -> dict[str, Any]:
        """
        Core prediction logic (no governance overhead).
        Returns a dict — keys are model-specific but must include the
        primary risk estimate (e.g. 'pd_estimate', 'lgd_estimate').
        """

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Return {feature_name: importance_score} sorted descending.
        Used by model cards and explainability modules.
        """

    # ------------------------------------------------------------------
    # Governed prediction (production entry point)
    # ------------------------------------------------------------------

    def predict(
        self,
        X: Any,
        application_id: str | None = None,
        environment: str = "production",
    ) -> dict[str, Any]:
        """
        Governance-wrapped prediction.

        1. Validates model is fitted.
        2. Runs input validation; raises on hard errors.
        3. Calls predict_proba (subclass logic).
        4. Writes immutable audit record.
        5. Attaches _governance block to the returned dict.
        """
        self._check_fitted()

        # --- input validation ---
        validation = self.validate_inputs(X)
        if not validation.passed:
            raise ValueError(
                f"Input validation failed for {self.metadata.model_id}: "
                + "; ".join(validation.errors)
            )
        if validation.warnings:
            for w in validation.warnings:
                logger.warning("[%s] input warning: %s", self.metadata.model_id, w)

        # --- core prediction ---
        result = self.predict_proba(X)

        # --- audit trail ---
        record_id = f"{self.metadata.model_id}-{self._prediction_count:010d}-{uuid.uuid4().hex[:6]}"
        record = PredictionRecord(
            record_id=record_id,
            model_id=self.metadata.model_id,
            model_version=self.metadata.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash=self._hash_inputs(X),
            output=result,
            environment=environment,
            application_id=application_id,
        )
        self._audit_log.append(record)
        self._prediction_count += 1

        if self._prediction_count % 1000 == 0:
            logger.info(
                "[%s] milestone: %d predictions made",
                self.metadata.model_id, self._prediction_count,
            )

        # --- attach governance block ---
        result["_governance"] = {
            "model_id": self.metadata.model_id,
            "model_version": self.metadata.version,
            "record_id": record_id,
            "timestamp": record.timestamp,
            "application_id": application_id,
            "regulatory_use": self.metadata.regulatory_use,
            "risk_tier": self.metadata.risk_tier,
            "environment": environment,
        }

        return result

    # ------------------------------------------------------------------
    # Input validation (default; override to add model-specific checks)
    # ------------------------------------------------------------------

    def validate_inputs(self, X: Any) -> ValidationResult:
        """
        Default validation: checks X is not None.
        Override in subclasses for feature-range and type checks.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if X is None:
            errors.append("Input X is None.")

        return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)

    # ------------------------------------------------------------------
    # Model card generation (SR 11-7 §III.C)
    # ------------------------------------------------------------------

    def generate_model_card(self) -> dict[str, Any]:
        """
        Auto-generate a SR 11-7 compliant model card.

        Covers: overview, regulatory alignment, governance lifecycle,
        feature importance, assumptions, limitations, and production stats.
        """
        feature_imp: dict[str, float] = {}
        if self._is_fitted:
            try:
                feature_imp = self.get_feature_importance()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not retrieve feature importance: %s", exc)

        return {
            "model_card_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_overview": {
                "id": self.metadata.model_id,
                "name": self.metadata.model_name,
                "version": self.metadata.version,
                "purpose": self.metadata.purpose,
                "asset_class": self.metadata.asset_class,
                "risk_tier": self.metadata.risk_tier,
                "status": self.metadata.status,
            },
            "regulatory_alignment": self.metadata.regulatory_use,
            "governance_lifecycle": {
                "developer": self.metadata.developer,
                "development_date": self.metadata.development_date,
                "model_owner": self.metadata.model_owner,
                "approved_by": self.metadata.approved_by,
                "approval_date": self.metadata.approval_date,
                "last_validation_date": self.metadata.last_validation_date,
                "next_review_date": self.metadata.next_review_date,
            },
            "feature_importance": feature_imp,
            "model_assumptions": self.metadata.assumptions,
            "known_limitations": self.metadata.known_limitations,
            "production_statistics": {
                "total_predictions": self._prediction_count,
                "audit_log_size": len(self._audit_log),
            },
        }

    # ------------------------------------------------------------------
    # Audit log access
    # ------------------------------------------------------------------

    def get_audit_log(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Return the prediction audit log as a list of dicts."""
        log = self._audit_log[-last_n:] if last_n else self._audit_log
        return [record.__dict__ for record in log]

    def clear_audit_log(self) -> int:
        """
        Flush the in-memory audit log (after persisting to DB/S3).
        Returns the number of records cleared.
        """
        n = len(self._audit_log)
        self._audit_log = []
        logger.info("[%s] audit log cleared (%d records)", self.metadata.model_id, n)
        return n

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"Model {self.metadata.model_id} has not been fitted. "
                "Call .fit() before .predict()."
            )

    @staticmethod
    def _hash_inputs(X: Any) -> str:
        """
        SHA-256 hex digest of serialised inputs.
        Privacy-preserving: hash enables forensic reproducibility
        without storing raw applicant data.
        """
        try:
            serialised = json.dumps(
                X if isinstance(X, (dict, list)) else str(X),
                sort_keys=True,
                default=str,
            ).encode()
            return hashlib.sha256(serialised).hexdigest()
        except Exception:  # noqa: BLE001
            return "hash_unavailable"

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"id={self.metadata.model_id!r} "
            f"v={self.metadata.version!r} "
            f"fitted={self._is_fitted}>"
        )
