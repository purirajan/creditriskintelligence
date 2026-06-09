"""
test_api.py
===========
Integration tests for all FastAPI endpoints.

Tests cover:
  - Health check
  - Single application scoring
  - Batch portfolio scoring
  - Full ECL calculation
  - Explainability / adverse action
  - Model monitoring endpoints
  - Model inventory
  - Audit log (authenticated)
  - Input validation (422 errors)
  - Authentication guard
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)
AUTH = {"X-API-Key": "dev-key-replace-in-production"}


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_status_healthy(self):
        r = client.get("/health")
        assert r.json()["status"] == "healthy"

    def test_health_returns_version(self):
        r = client.get("/health")
        assert "version" in r.json()

    def test_health_no_auth_required(self):
        """Health endpoint must be publicly accessible (load balancer probes)."""
        r = client.get("/health")
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Single scoring  POST /v1/score
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreEndpoint:

    VALID_PAYLOAD = {
        "application_id":   "TEST-001",
        "fico_score":       720,
        "dti_ratio":        0.28,
        "utilization_rate": 0.30,
        "months_on_book":   36,
        "delinquency_count":0,
        "income_verified":  1,
    }

    def test_score_returns_200(self):
        r = client.post("/v1/score", json=self.VALID_PAYLOAD)
        assert r.status_code == 200

    def test_score_returns_pd_estimate(self):
        r = client.post("/v1/score", json=self.VALID_PAYLOAD)
        body = r.json()
        assert "pd_estimate" in body
        assert 0.0 <= body["pd_estimate"] <= 1.0

    def test_score_returns_confidence_interval(self):
        r = client.post("/v1/score", json=self.VALID_PAYLOAD)
        body = r.json()
        assert body["pd_lower_95"] <= body["pd_estimate"] <= body["pd_upper_95"]

    def test_score_returns_risk_grade(self):
        r = client.post("/v1/score", json=self.VALID_PAYLOAD)
        body = r.json()
        assert body["risk_grade"] in ("AAA","AA","A","A-","BBB","BB","B","CCC","D")

    def test_score_returns_decision(self):
        r = client.post("/v1/score", json=self.VALID_PAYLOAD)
        body = r.json()
        assert body["decision_recommendation"] in ("APPROVE", "REVIEW", "DECLINE")

    def test_score_returns_governance_block(self):
        r = client.post("/v1/score", json=self.VALID_PAYLOAD)
        body = r.json()
        assert "governance" in body
        gov = body["governance"]
        assert gov["model_id"] == "PD-CONSUMER-001"
        assert "timestamp" in gov
        assert "record_id" in gov

    def test_score_application_id_echoed(self):
        r = client.post("/v1/score", json=self.VALID_PAYLOAD)
        assert r.json()["application_id"] == "TEST-001"

    def test_good_profile_approves(self):
        r = client.post("/v1/score", json=self.VALID_PAYLOAD)
        assert r.json()["decision_recommendation"] == "APPROVE"

    def test_risky_profile_declines(self):
        risky = {
            "fico_score": 550, "dti_ratio": 0.65,
            "utilization_rate": 0.90, "months_on_book": 2,
            "delinquency_count": 3, "income_verified": 0,
        }
        r = client.post("/v1/score", json=risky)
        assert r.json()["decision_recommendation"] == "DECLINE"

    def test_missing_required_field_returns_422(self):
        bad = {"fico_score": 700}   # missing dti_ratio, etc.
        r = client.post("/v1/score", json=bad)
        assert r.status_code == 422

    def test_fico_below_300_returns_422(self):
        bad = dict(self.VALID_PAYLOAD)
        bad["fico_score"] = 200
        r = client.post("/v1/score", json=bad)
        assert r.status_code == 422

    def test_fico_above_850_returns_422(self):
        bad = dict(self.VALID_PAYLOAD)
        bad["fico_score"] = 900
        r = client.post("/v1/score", json=bad)
        assert r.status_code == 422

    def test_dti_above_1_returns_422(self):
        bad = dict(self.VALID_PAYLOAD)
        bad["dti_ratio"] = 1.5
        r = client.post("/v1/score", json=bad)
        assert r.status_code == 422

    def test_score_without_application_id(self):
        payload = dict(self.VALID_PAYLOAD)
        del payload["application_id"]
        r = client.post("/v1/score", json=payload)
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Batch scoring  POST /v1/score/batch
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchScoreEndpoint:

    def _make_batch(self, n: int = 5) -> dict:
        apps = [
            {
                "application_id":   f"APP-{i:03d}",
                "fico_score":       600 + i * 10,
                "dti_ratio":        0.30 + i * 0.02,
                "utilization_rate": 0.40,
                "months_on_book":   12 + i * 3,
                "delinquency_count":0,
                "income_verified":  1,
            }
            for i in range(n)
        ]
        return {"portfolio_id": "PORT-001", "applications": apps}

    def test_batch_returns_200(self):
        r = client.post("/v1/score/batch", json=self._make_batch())
        assert r.status_code == 200

    def test_batch_total_matches_input(self):
        r = client.post("/v1/score/batch", json=self._make_batch(5))
        assert r.json()["total"] == 5

    def test_batch_results_have_pd(self):
        r = client.post("/v1/score/batch", json=self._make_batch(3))
        for result in r.json()["results"]:
            assert "pd_estimate" in result
            assert 0 <= result["pd_estimate"] <= 1

    def test_batch_summary_contains_rates(self):
        r = client.post("/v1/score/batch", json=self._make_batch(10))
        summary = r.json()["summary"]
        assert "approve_rate" in summary
        assert "decline_rate" in summary
        assert "mean_pd" in summary

    def test_batch_rates_sum_to_one(self):
        r = client.post("/v1/score/batch", json=self._make_batch(20))
        s = r.json()["summary"]
        total = s["approve_rate"] + s["review_rate"] + s["decline_rate"]
        assert abs(total - 1.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# ECL scoring  POST /v1/score/ecl
# ─────────────────────────────────────────────────────────────────────────────

class TestECLEndpoint:

    ECL_PAYLOAD = {
        "application_id": "ECL-001",
        "pd_inputs": {
            "fico_score": 680, "dti_ratio": 0.35,
            "utilization_rate": 0.42, "months_on_book": 18,
            "delinquency_count": 0, "income_verified": 1,
        },
        "lgd_inputs": {
            "collateral_value": 0, "loan_amount": 5000,
            "months_past_due": 0, "secured_flag": 0,
            "product_type_code": 2, "origination_ltv": 0,
            "time_in_default_months": 0,
        },
        "ead_inputs": {
            "current_balance": 1200, "credit_limit": 5000,
            "utilization_rate": 0.24, "months_to_maturity": 0,
            "product_type_code": 1, "months_since_last_draw": 2,
            "payment_behaviour_score": 72,
        },
    }

    def test_ecl_returns_200(self):
        r = client.post("/v1/score/ecl", json=self.ECL_PAYLOAD)
        assert r.status_code == 200

    def test_ecl_has_all_components(self):
        r = client.post("/v1/score/ecl", json=self.ECL_PAYLOAD)
        body = r.json()
        for field in ("pd_estimate", "lgd_estimate", "ead_estimate",
                      "ecl_12_month", "ecl_lifetime"):
            assert field in body

    def test_ecl_12m_equals_pd_lgd_ead(self):
        r = client.post("/v1/score/ecl", json=self.ECL_PAYLOAD)
        body = r.json()
        expected = body["pd_estimate"] * body["lgd_estimate"] * body["ead_estimate"]
        assert abs(body["ecl_12_month"] - round(expected, 2)) < 0.10

    def test_ecl_lifetime_gte_12month(self):
        r = client.post("/v1/score/ecl", json=self.ECL_PAYLOAD)
        body = r.json()
        assert body["ecl_lifetime"] >= body["ecl_12_month"]


# ─────────────────────────────────────────────────────────────────────────────
# Explainability  POST /v1/explain
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainEndpoint:

    VALID_PAYLOAD = {
        "application_id":   "EXP-001",
        "fico_score":       620,
        "dti_ratio":        0.48,
        "utilization_rate": 0.72,
        "months_on_book":   6,
        "delinquency_count":1,
        "income_verified":  0,
    }

    def test_explain_returns_200(self):
        r = client.post("/v1/explain", json=self.VALID_PAYLOAD)
        assert r.status_code == 200

    def test_explain_has_shap_contributions(self):
        r = client.post("/v1/explain", json=self.VALID_PAYLOAD)
        body = r.json()
        assert "feature_contributions" in body
        assert len(body["feature_contributions"]) == 6

    def test_explain_has_top_risk_drivers(self):
        r = client.post("/v1/explain", json=self.VALID_PAYLOAD)
        body = r.json()
        assert "top_risk_drivers" in body
        assert len(body["top_risk_drivers"]) > 0

    def test_risky_profile_triggers_adverse_action(self):
        r = client.post("/v1/explain", json=self.VALID_PAYLOAD)
        body = r.json()
        assert body["is_adverse_action"] is True

    def test_adverse_action_has_reg_b_codes(self):
        r = client.post("/v1/explain", json=self.VALID_PAYLOAD)
        reasons = r.json()["adverse_action_reasons"]
        assert len(reasons) >= 1
        assert len(reasons) <= 4          # Reg B max 4
        assert all("code" in r for r in reasons)
        assert all(r["code"].startswith("AA-") for r in reasons)

    def test_good_profile_no_adverse_action(self):
        good = {
            "fico_score": 780, "dti_ratio": 0.20,
            "utilization_rate": 0.15, "months_on_book": 60,
            "delinquency_count": 0, "income_verified": 1,
        }
        r = client.post("/v1/explain", json=good)
        body = r.json()
        assert body["is_adverse_action"] is False
        assert body["adverse_action_reasons"] == []

    def test_explain_has_regulatory_note(self):
        r = client.post("/v1/explain", json=self.VALID_PAYLOAD)
        assert "regulatory_note" in r.json()

    def test_explain_has_governance_block(self):
        r = client.post("/v1/explain", json=self.VALID_PAYLOAD)
        assert "governance" in r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring  GET /v1/monitor
# ─────────────────────────────────────────────────────────────────────────────

class TestMonitorEndpoint:

    def test_monitor_list_returns_200(self):
        r = client.get("/v1/monitor")
        assert r.status_code == 200

    def test_monitor_list_has_summary(self):
        r = client.get("/v1/monitor")
        body = r.json()
        assert "summary" in body
        assert "total_models" in body["summary"]

    def test_monitor_single_model_returns_200(self):
        r = client.get("/v1/monitor/PD-CONSUMER-001")
        assert r.status_code == 200

    def test_monitor_single_has_metrics(self):
        r = client.get("/v1/monitor/PD-CONSUMER-001")
        body = r.json()
        assert "metrics" in body
        assert "psi" in body["metrics"]
        assert "gini_coefficient" in body["metrics"]

    def test_monitor_unknown_model_returns_404(self):
        r = client.get("/v1/monitor/NONEXISTENT-999")
        assert r.status_code == 404

    def test_monitor_ead_model_has_warning_status(self):
        r = client.get("/v1/monitor/EAD-CONSUMER-001")
        assert r.json()["status"] == "warning"

    def test_monitor_pd_model_is_healthy(self):
        r = client.get("/v1/monitor/PD-CONSUMER-001")
        assert r.json()["status"] == "healthy"


# ─────────────────────────────────────────────────────────────────────────────
# Model inventory  GET /v1/models
# ─────────────────────────────────────────────────────────────────────────────

class TestModelsEndpoint:

    def test_models_returns_200(self):
        r = client.get("/v1/models")
        assert r.status_code == 200

    def test_models_returns_list(self):
        r = client.get("/v1/models")
        assert "models" in r.json()
        assert len(r.json()["models"]) >= 3

    def test_models_have_required_fields(self):
        r = client.get("/v1/models")
        for model in r.json()["models"]:
            assert "model_id" in model
            assert "version" in model
            assert "status" in model
            assert "regulatory_use" in model


# ─────────────────────────────────────────────────────────────────────────────
# Audit log  GET /v1/audit/{model_id}
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditEndpoint:

    def test_audit_without_key_returns_401(self):
        r = client.get("/v1/audit/PD-CONSUMER-001")
        assert r.status_code == 401

    def test_audit_with_key_returns_200(self):
        r = client.get("/v1/audit/PD-CONSUMER-001", headers=AUTH)
        assert r.status_code == 200

    def test_audit_has_entries(self):
        r = client.get("/v1/audit/PD-CONSUMER-001", headers=AUTH)
        body = r.json()
        assert "entries" in body
        assert "total" in body

    def test_audit_wrong_key_returns_401(self):
        r = client.get("/v1/audit/PD-CONSUMER-001",
                       headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401
