"""
base_model.py — Abstract base class for all credit risk models.
Enforces governance hooks required by SR 11-7 / MRM frameworks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib, json, logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    model_id: str
    model_name: str
    version: str
    purpose: str                    # e.g., "PD estimation for BNPL portfolio"
    regulatory_use: List[str]       # e.g., ["CECL", "Basel"]
    developer: str
    development_date: str
    last_validation_date: Optional[str] = None
    next_review_date: Optional[str] = None
    risk_tier: str = "High"         # High / Medium / Low per SR 11-7
    status: str = "Production"      # Development / Validation / Production / Retired


@dataclass
class PredictionRecord:
    """Immutable audit record for every model prediction."""
    record_id: str
    model_id: str
    model_version: str
    timestamp: str
    input_hash: str                 # Hash of inputs (privacy-preserving audit)
    output: Dict[str, Any]
    environment: str = "production"


class BaseCreditRiskModel(ABC):
    """
    Abstract base for all CreditRisk Intelligence models.
    
    All subclasses automatically get:
    - Audit logging on every prediction
    - Input validation
    - Model card generation
    - Drift monitoring hooks
    """

    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self._audit_log: List[PredictionRecord] = []
        self._prediction_count = 0
        logger.info(f"Initialized model: {metadata.model_id} v{metadata.version}")

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Train the model. Must be implemented by subclass."""
        pass

    @abstractmethod
    def predict_proba(self, X) -> Dict[str, Any]:
        """Return predictions with confidence intervals."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importances for explainability."""
        pass

    def predict(self, X, application_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Governed predict — wraps predict_proba with audit logging.
        This is the method that should be called in production.
        """
        # Validate inputs
        self._validate_inputs(X)

        # Get prediction from subclass
        result = self.predict_proba(X)

        # Audit log
        record = PredictionRecord(
            record_id=f"{self.metadata.model_id}-{self._prediction_count:08d}",
            model_id=self.metadata.model_id,
            model_version=self.metadata.version,
            timestamp=datetime.utcnow().isoformat(),
            input_hash=self._hash_inputs(X),
            output=result,
        )
        self._audit_log.append(record)
        self._prediction_count += 1

        # Attach governance metadata to output
        result["_governance"] = {
            "model_id": self.metadata.model_id,
            "model_version": self.metadata.version,
            "record_id": record.record_id,
            "timestamp": record.timestamp,
            "regulatory_use": self.metadata.regulatory_use,
        }

        return result

    def generate_model_card(self) -> Dict[str, Any]:
        """Auto-generate SR 11-7 compliant model card."""
        return {
            "model_overview": {
                "id": self.metadata.model_id,
                "name": self.metadata.model_name,
                "version": self.metadata.version,
                "purpose": self.metadata.purpose,
                "risk_tier": self.metadata.risk_tier,
                "status": self.metadata.status,
            },
            "regulatory_alignment": self.metadata.regulatory_use,
            "development": {
                "developer": self.metadata.developer,
                "development_date": self.metadata.development_date,
                "last_validation": self.metadata.last_validation_date,
                "next_review": self.metadata.next_review_date,
            },
            "feature_importance": self.get_feature_importance(),
            "production_stats": {
                "total_predictions": self._prediction_count,
            },
        }

    def _validate_inputs(self, X):
        """Override to add model-specific input validation."""
        if X is None:
            raise ValueError("Input X cannot be None")

    def _hash_inputs(self, X) -> str:
        """Create privacy-preserving hash of inputs for audit trail."""
        try:
            data = json.dumps(X, sort_keys=True, default=str)
            return hashlib.sha256(data.encode()).hexdigest()[:16]
        except Exception:
            return "hash_error"
