"""
test_governance.py
==================
Unit tests for governance modules:
  - DriftMonitor (PSI, KS, Gini alerts)
  - AuditTrail (InMemoryBackend)
  - ModelCardGenerator
  - Threshold registry
"""

import numpy as np
import pytest

from src.monitoring.drift_detector import (
    DriftMonitor,
    compute_psi,
    compute_ks,
    compute_gini,
    PSI_STABLE,
    PSI_MODERATE,
)
from src.governance.audit_trail import AuditTrail, InMemoryBackend, EventType
from src.governance.thresholds import check_threshold, ALL_THRESHOLDS


# ─────────────────────────────────────────────────────────────────────────────
# PSI / KS / Gini pure functions
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricFunctions:

    def test_psi_identical_distributions_is_zero(self):
        scores = np.random.uniform(0, 1, 500)
        psi = compute_psi(scores, scores)
        assert psi < 0.01

    def test_psi_very_different_distributions_is_high(self):
        expected = np.random.uniform(0.0, 0.3, 500)
        actual   = np.random.uniform(0.7, 1.0, 500)
        psi = compute_psi(expected, actual)
        assert psi > PSI_MODERATE

    def test_psi_is_non_negative(self):
        a = np.random.uniform(0, 1, 300)
        b = np.random.uniform(0, 1, 300)
        assert compute_psi(a, b) >= 0

    def test_ks_perfect_separation(self):
        y_true  = np.array([1] * 100 + [0] * 100)
        # Perfect separation: defaults score 1.0, non-defaults score 0.0
        y_score = np.array([1.0] * 100 + [0.0] * 100)
        ks = compute_ks(y_true, y_score)
        assert ks > 0.95

    def test_ks_no_separation(self):
        rng = np.random.default_rng(0)
        y_true  = rng.integers(0, 2, 200)
        y_score = rng.uniform(0, 1, 200)
        ks = compute_ks(y_true, y_score)
        assert ks < 0.30   # near-random

    def test_gini_range(self):
        rng = np.random.default_rng(1)
        y_true  = rng.integers(0, 2, 300)
        y_score = rng.uniform(0, 1, 300)
        gini = compute_gini(y_true, y_score)
        assert -1.0 <= gini <= 1.0

    def test_gini_good_model(self):
        rng = np.random.default_rng(2)
        n = 500
        y_true = rng.integers(0, 2, n)
        # Good model: defaults get high scores
        y_score = np.where(y_true == 1,
                           rng.uniform(0.6, 1.0, n),
                           rng.uniform(0.0, 0.4, n))
        gini = compute_gini(y_true, y_score)
        assert gini > 0.40


# ─────────────────────────────────────────────────────────────────────────────
# DriftMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftMonitor:

    @pytest.fixture
    def stable_monitor(self):
        rng = np.random.default_rng(42)
        baseline = rng.uniform(0.0, 0.4, 1000)
        return DriftMonitor(
            model_id="PD-CONSUMER-001",
            baseline_scores=baseline,
            baseline_ks=0.42,
            baseline_gini=0.63,
        )

    def test_healthy_run_status(self, stable_monitor):
        """Scores from same distribution → healthy status."""
        rng = np.random.default_rng(99)
        current = rng.uniform(0.0, 0.4, 500)
        y_true  = rng.integers(0, 2, 500)
        result = stable_monitor.run(current, y_true)
        assert result.status in ("healthy", "warning")

    def test_critical_psi_triggers_alert(self, stable_monitor):
        """Scores shifted dramatically → critical PSI alert."""
        rng = np.random.default_rng(7)
        current = rng.uniform(0.7, 1.0, 500)   # very different from baseline
        y_true  = rng.integers(0, 2, 500)
        result = stable_monitor.run(current, y_true)
        assert result.status in ("warning", "critical")
        assert result.alert_count > 0

    def test_run_returns_all_metrics(self, stable_monitor):
        rng = np.random.default_rng(5)
        current = rng.uniform(0.0, 0.4, 300)
        y_true  = rng.integers(0, 2, 300)
        result = stable_monitor.run(current, y_true)
        for metric in ("psi", "ks_statistic", "gini_coefficient", "auc_roc"):
            assert metric in result.metrics

    def test_run_id_is_unique(self, stable_monitor):
        rng = np.random.default_rng(3)
        scores = rng.uniform(0, 0.4, 200)
        y = rng.integers(0, 2, 200)
        r1 = stable_monitor.run(scores, y)
        r2 = stable_monitor.run(scores, y)
        assert r1.run_id != r2.run_id

    def test_to_dict_serialisable(self, stable_monitor):
        import json
        rng = np.random.default_rng(4)
        scores = rng.uniform(0, 0.4, 200)
        y = rng.integers(0, 2, 200)
        result = stable_monitor.run(scores, y)
        # Should not raise
        serialised = json.dumps(result.to_dict())
        assert len(serialised) > 0


# ─────────────────────────────────────────────────────────────────────────────
# AuditTrail
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditTrail:

    @pytest.fixture
    def trail(self):
        return AuditTrail(backend=InMemoryBackend())

    def test_log_prediction_returns_event_id(self, trail):
        eid = trail.log_prediction(
            application_id="APP-001",
            model_id="PD-CONSUMER-001",
            version="1.2.0",
            input_hash="abc123",
            output={"pd_estimate": 0.042},
        )
        assert isinstance(eid, str)
        assert len(eid) > 0

    def test_logged_event_queryable(self, trail):
        trail.log_prediction(
            application_id="APP-002",
            model_id="PD-CONSUMER-001",
            version="1.2.0",
            input_hash="def456",
            output={"pd_estimate": 0.08},
        )
        results = trail.query(model_id="PD-CONSUMER-001")
        assert len(results) == 1

    def test_event_type_filter(self, trail):
        trail.log_prediction("APP-003", "PD-CONSUMER-001", "1.0",
                             "hash1", {"pd": 0.03})
        trail.log_deployment("PD-CONSUMER-001", "1.1.0", actor="rajan")
        preds = trail.query(event_type=EventType.PREDICTION)
        deploys = trail.query(event_type=EventType.DEPLOYMENT)
        assert len(preds) == 1
        assert len(deploys) == 1

    def test_log_deployment(self, trail):
        eid = trail.log_deployment(
            model_id="PD-CONSUMER-001",
            version="1.2.0",
            actor="rajan.puri",
            notes="Bumped XGBoost version",
            previous_version="1.1.0",
        )
        assert eid is not None
        results = trail.query(event_type=EventType.DEPLOYMENT)
        assert results[0]["payload"]["previous_version"] == "1.1.0"

    def test_log_validation(self, trail):
        trail.log_validation(
            model_id="PD-CONSUMER-001",
            version="1.2.0",
            validator="model_validator",
            outcome="Approved",
            findings=["No material issues found.", "Gini 0.63 — within threshold."],
        )
        results = trail.query(event_type=EventType.VALIDATION)
        assert results[0]["payload"]["outcome"] == "Approved"

    def test_multiple_models_isolated(self, trail):
        trail.log_prediction("A1", "PD-CONSUMER-001", "1.0", "h1", {"pd": 0.02})
        trail.log_prediction("B1", "LGD-CONSUMER-001", "1.0", "h2", {"lgd": 0.45})
        pd_events  = trail.query(model_id="PD-CONSUMER-001")
        lgd_events = trail.query(model_id="LGD-CONSUMER-001")
        assert len(pd_events) == 1
        assert len(lgd_events) == 1

    def test_in_memory_backend_len(self, trail):
        for i in range(5):
            trail.log_prediction(f"APP-{i}", "M1", "1.0", f"h{i}", {})
        assert len(trail.backend) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Threshold registry
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholds:

    def test_all_thresholds_registered(self):
        for key in ("psi", "gini", "ks", "auc_roc"):
            assert key in ALL_THRESHOLDS

    def test_psi_healthy_no_flag(self):
        flagged, severity = check_threshold("psi", 0.05)
        assert flagged is False

    def test_psi_warning_flagged(self):
        flagged, severity = check_threshold("psi", 0.15)
        assert flagged is True
        assert severity.value == "warning"

    def test_psi_critical_flagged(self):
        flagged, severity = check_threshold("psi", 0.30)
        assert flagged is True
        assert severity.value == "critical"

    def test_gini_healthy_no_flag(self):
        flagged, _ = check_threshold("gini", 0.55)
        assert flagged is False

    def test_gini_critical_flagged(self):
        flagged, severity = check_threshold("gini", 0.15)
        assert flagged is True
        assert severity.value == "critical"

    def test_unknown_metric_no_flag(self):
        flagged, severity = check_threshold("nonexistent_metric", 999.0)
        assert flagged is False

    @pytest.mark.parametrize("metric,value,expect_flag", [
        ("psi",    0.05,  False),
        ("psi",    0.12,  True),
        ("psi",    0.30,  True),
        ("gini",   0.50,  False),
        ("gini",   0.28,  True),
        ("ks",     0.40,  False),
        ("ks",     0.18,  True),
        ("auc_roc",0.75,  False),
        ("auc_roc",0.58,  True),
    ])
    def test_threshold_parametrised(self, metric, value, expect_flag):
        flagged, _ = check_threshold(metric, value)
        assert flagged == expect_flag
