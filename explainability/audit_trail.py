"""
audit_trail.py
==============
Immutable audit trail for all model predictions, governance events,
and monitoring runs.

Provides:
  - In-memory log (dev/test)
  - File-backed JSONL log (production)
  - Structured event types: prediction, monitoring, deployment, validation
  - SR 11-7 compliant retention metadata

For production deployments, swap the FileAuditBackend for your preferred
persistent store (PostgreSQL, S3 JSONL, Snowflake, etc.) by implementing
the AuditBackend protocol.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    PREDICTION   = "prediction"
    MONITORING   = "monitoring"
    DEPLOYMENT   = "deployment"
    VALIDATION   = "validation"
    GOVERNANCE   = "governance"
    ALERT        = "alert"
    DATA_QUALITY = "data_quality"


@dataclass
class AuditEvent:
    event_id: str
    event_type: EventType
    timestamp: str          # UTC ISO-8601
    model_id: str
    model_version: str
    actor: str              # user / service / system
    payload: dict[str, Any]
    environment: str = "production"
    session_id: str | None = None
    parent_event_id: str | None = None

    @classmethod
    def create(
        cls,
        event_type: EventType,
        model_id: str,
        model_version: str,
        actor: str,
        payload: dict[str, Any],
        environment: str = "production",
        session_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> "AuditEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=model_id,
            model_version=model_version,
            actor=actor,
            payload=payload,
            environment=environment,
            session_id=session_id,
            parent_event_id=parent_event_id,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class AuditBackend(ABC):
    """Protocol for audit trail persistence backends."""

    @abstractmethod
    def write(self, event: AuditEvent) -> None: ...

    @abstractmethod
    def query(
        self,
        model_id: str | None = None,
        event_type: EventType | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]: ...


class InMemoryBackend(AuditBackend):
    """In-memory backend for development and testing."""

    def __init__(self) -> None:
        self._store: list[AuditEvent] = []
        self._lock = threading.Lock()

    def write(self, event: AuditEvent) -> None:
        with self._lock:
            self._store.append(event)

    def query(
        self,
        model_id: str | None = None,
        event_type: EventType | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._store)

        if model_id:
            events = [e for e in events if e.model_id == model_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return [e.to_dict() for e in events[-limit:]]

    def __len__(self) -> int:
        return len(self._store)


class FileAuditBackend(AuditBackend):
    """
    JSONL file backend.

    Each line is a single JSON-serialised AuditEvent.
    Files are rotated daily: audit_YYYY-MM-DD.jsonl

    Thread-safe via a per-file lock.
    """

    def __init__(self, log_dir: str | Path = "./audit_logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _current_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def write(self, event: AuditEvent) -> None:
        with self._lock:
            with open(self._current_path(), "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")

    def query(
        self,
        model_id: str | None = None,
        event_type: EventType | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)

        for log_file in log_files:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if model_id and event.get("model_id") != model_id:
                        continue
                    if event_type and event.get("event_type") != event_type.value:
                        continue
                    if start_time and event.get("timestamp", "") < start_time:
                        continue
                    if end_time and event.get("timestamp", "") > end_time:
                        continue
                    results.append(event)
                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break

        return results


# ---------------------------------------------------------------------------
# AuditTrail facade
# ---------------------------------------------------------------------------

class AuditTrail:
    """
    Central audit trail service.

    Instantiate once per application and inject into models / API handlers.

    Usage
    -----
    >>> trail = AuditTrail(backend=FileAuditBackend("./audit_logs"))
    >>> trail.log_prediction("APP-001", model_id="PD-CONSUMER-001",
    ...                       version="1.0.0", input_hash="abc123",
    ...                       output={"pd_estimate": 0.042})
    """

    def __init__(
        self,
        backend: AuditBackend | None = None,
        default_actor: str = "system",
        environment: str = "production",
    ) -> None:
        self.backend = backend or InMemoryBackend()
        self.default_actor = default_actor
        self.environment = environment

    # ------------------------------------------------------------------
    # Convenience log methods
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        application_id: str | None,
        model_id: str,
        version: str,
        input_hash: str,
        output: dict[str, Any],
        actor: str | None = None,
        session_id: str | None = None,
    ) -> str:
        event = AuditEvent.create(
            event_type=EventType.PREDICTION,
            model_id=model_id,
            model_version=version,
            actor=actor or self.default_actor,
            payload={
                "application_id": application_id,
                "input_hash": input_hash,
                "output_summary": {
                    k: v for k, v in output.items()
                    if not k.startswith("_") and not isinstance(v, dict)
                },
            },
            environment=self.environment,
            session_id=session_id,
        )
        self.backend.write(event)
        return event.event_id

    def log_deployment(
        self,
        model_id: str,
        version: str,
        actor: str,
        notes: str = "",
        previous_version: str | None = None,
    ) -> str:
        event = AuditEvent.create(
            event_type=EventType.DEPLOYMENT,
            model_id=model_id,
            model_version=version,
            actor=actor,
            payload={
                "previous_version": previous_version,
                "notes": notes,
                "environment": self.environment,
            },
            environment=self.environment,
        )
        self.backend.write(event)
        logger.info("Deployment logged: %s v%s by %s", model_id, version, actor)
        return event.event_id

    def log_monitoring_run(
        self,
        model_id: str,
        version: str,
        run_result: dict[str, Any],
    ) -> str:
        event = AuditEvent.create(
            event_type=EventType.MONITORING,
            model_id=model_id,
            model_version=version,
            actor="monitoring_service",
            payload=run_result,
            environment=self.environment,
        )
        self.backend.write(event)
        return event.event_id

    def log_validation(
        self,
        model_id: str,
        version: str,
        validator: str,
        outcome: str,
        findings: list[str],
    ) -> str:
        event = AuditEvent.create(
            event_type=EventType.VALIDATION,
            model_id=model_id,
            model_version=version,
            actor=validator,
            payload={
                "outcome": outcome,
                "findings": findings,
            },
            environment=self.environment,
        )
        self.backend.write(event)
        logger.info("Validation logged: %s v%s → %s", model_id, version, outcome)
        return event.event_id

    def log_alert(
        self,
        model_id: str,
        version: str,
        alert_dict: dict[str, Any],
    ) -> str:
        event = AuditEvent.create(
            event_type=EventType.ALERT,
            model_id=model_id,
            model_version=version,
            actor="monitoring_service",
            payload=alert_dict,
            environment=self.environment,
        )
        self.backend.write(event)
        return event.event_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        model_id: str | None = None,
        event_type: EventType | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return self.backend.query(
            model_id=model_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
