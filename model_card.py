"""
model_card.py
=============
Auto-generates SR 11-7 compliant model documentation from a fitted model.

Produces:
  - Structured model card (dict / JSON)
  - Markdown model card (human-readable)
  - Regulatory mapping table (CECL / Basel / CCAR)
  - Model limitations and assumptions section
  - Version change log support

SR 11-7 requires documented model purpose, methodology, assumptions,
limitations, performance benchmarks, and a clearly defined governance
lifecycle (development → validation → approval → production → retirement).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regulatory mapping reference
# ---------------------------------------------------------------------------

REGULATORY_DESCRIPTIONS: dict[str, str] = {
    "CECL": (
        "Current Expected Credit Loss (ASC 326). Requires lifetime PD/LGD curves "
        "for allowance estimation. Model outputs feed the allowance calculation "
        "for all financial instruments in scope."
    ),
    "Basel II IRB": (
        "Internal Ratings-Based approach under Basel II Pillar 1. "
        "PD, LGD, and EAD estimates drive Risk-Weighted Asset calculation."
    ),
    "Basel III IRB": (
        "Updated IRB framework under Basel III / CRR II. "
        "Includes output floors, revised LGD/EAD definitions, and stricter PD calibration."
    ),
    "CCAR": (
        "Comprehensive Capital Analysis and Review. Model supports stress-scenario "
        "loss estimation (baseline, adverse, severely adverse) for DFAST/CCAR submissions."
    ),
    "IFRS 9": (
        "International Financial Reporting Standard 9. "
        "Supports Stage 1/2/3 classification and 12-month / lifetime ECL estimation."
    ),
    "ECOA": (
        "Equal Credit Opportunity Act. Model outputs trigger SHAP-based adverse "
        "action reason codes (Reg B) for all declined applicants."
    ),
}


# ---------------------------------------------------------------------------
# ModelCardGenerator
# ---------------------------------------------------------------------------

class ModelCardGenerator:
    """
    Generates structured and markdown model cards for any BaseCreditRiskModel.

    Usage
    -----
    >>> gen = ModelCardGenerator(fitted_pd_model)
    >>> card = gen.generate()
    >>> print(gen.to_markdown())
    """

    def __init__(
        self,
        model: Any,
        performance_metrics: dict[str, float] | None = None,
        validation_findings: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model                : fitted BaseCreditRiskModel instance
        performance_metrics  : out-of-sample metrics dict (AUC, Gini, KS, MAE, etc.)
        validation_findings  : list of strings from independent model validation
        """
        self.model = model
        self.performance_metrics = performance_metrics or {}
        self.validation_findings = validation_findings or []

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self) -> dict[str, Any]:
        """
        Return the full model card as a structured dict.
        Merges the base model card with performance and validation sections.
        """
        base = self.model.generate_model_card()
        meta = self.model.metadata

        return {
            **base,
            "performance_metrics": self._build_performance_section(),
            "regulatory_alignment_detail": self._build_regulatory_section(meta.regulatory_use),
            "validation_summary": self._build_validation_section(),
            "governance_controls": self._build_governance_controls(),
            "intended_use": {
                "primary_use": meta.purpose,
                "out_of_scope_uses": [
                    "Do not use for non-consumer credit segments without recalibration.",
                    "Do not apply to sovereigns, corporates, or financial institutions.",
                    "Point-in-time estimates require macro-economic adjustments before use in CECL.",
                ],
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the model card as a formatted JSON string."""
        return json.dumps(self.generate(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """
        Render the model card as a Markdown document suitable for GitHub
        or internal documentation portals.
        """
        card = self.generate()
        meta = self.model.metadata
        lines: list[str] = []

        lines += [
            f"# Model Card: {meta.model_name}",
            f"",
            f"**Model ID**: `{meta.model_id}`  ",
            f"**Version**: `{meta.version}`  ",
            f"**Status**: {meta.status}  ",
            f"**Risk Tier**: {meta.risk_tier} (SR 11-7)  ",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"",
            f"---",
            f"",
            f"## 1. Model Purpose",
            f"",
            meta.purpose,
            f"",
            f"**Asset Class**: {meta.asset_class}  ",
            f"**Regulatory Use**: {', '.join(meta.regulatory_use)}",
            f"",
            f"---",
            f"",
            f"## 2. Model Development",
            f"",
            f"| Field | Value |",
            f"|---|---|",
            f"| Developer | {meta.developer} |",
            f"| Development Date | {meta.development_date} |",
            f"| Model Owner | {meta.model_owner or 'TBD'} |",
            f"| Approved By | {meta.approved_by or 'TBD'} |",
            f"| Approval Date | {meta.approval_date or 'TBD'} |",
            f"| Last Validation | {meta.last_validation_date or 'TBD'} |",
            f"| Next Review | {meta.next_review_date or 'TBD'} |",
            f"",
            f"---",
            f"",
            f"## 3. Performance Metrics",
            f"",
        ]

        pm = card.get("performance_metrics", {})
        if pm.get("out_of_sample"):
            lines += [f"### Out-of-sample (hold-out test set)", f""]
            for k, v in pm["out_of_sample"].items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        if pm.get("in_sample"):
            lines += [f"### In-sample (training set)", f""]
            for k, v in pm["in_sample"].items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        lines += [
            f"---",
            f"",
            f"## 4. Feature Importance",
            f"",
            f"| Feature | Importance |",
            f"|---|---|",
        ]
        for feat, imp in card.get("feature_importance", {}).items():
            lines.append(f"| {feat} | {imp:.5f} |")
        lines.append("")

        lines += [
            f"---",
            f"",
            f"## 5. Regulatory Alignment",
            f"",
        ]
        for reg, desc in card.get("regulatory_alignment_detail", {}).items():
            lines += [f"### {reg}", f"", desc, ""]

        lines += [
            f"---",
            f"",
            f"## 6. Assumptions",
            f"",
        ]
        for i, assumption in enumerate(meta.assumptions, 1):
            lines.append(f"{i}. {assumption}")
        lines.append("")

        lines += [
            f"---",
            f"",
            f"## 7. Known Limitations",
            f"",
        ]
        for i, limitation in enumerate(meta.known_limitations, 1):
            lines.append(f"{i}. {limitation}")
        lines.append("")

        if self.validation_findings:
            lines += [
                f"---",
                f"",
                f"## 8. Independent Validation Findings",
                f"",
            ]
            for finding in self.validation_findings:
                lines.append(f"- {finding}")
            lines.append("")

        lines += [
            f"---",
            f"",
            f"## 9. Governance Controls",
            f"",
        ]
        for control in card.get("governance_controls", []):
            lines.append(f"- {control}")
        lines.append("")

        lines += [
            f"---",
            f"",
            f"*This model card was auto-generated by CreditRisk Intelligence.*  ",
            f"*All model decisions require human review per SR 11-7 §III.D.*",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_performance_section(self) -> dict[str, Any]:
        oos = self.performance_metrics
        in_sample = getattr(self.model, "_train_metrics", {})
        return {
            "out_of_sample": oos,
            "in_sample": in_sample,
            "benchmark": {
                "minimum_acceptable_gini": 0.30,
                "minimum_acceptable_ks":   0.20,
                "minimum_acceptable_auc":  0.65,
            },
        }

    @staticmethod
    def _build_regulatory_section(regulatory_use: list[str]) -> dict[str, str]:
        return {
            reg: REGULATORY_DESCRIPTIONS.get(reg, "See regulatory guidance document.")
            for reg in regulatory_use
        }

    def _build_validation_section(self) -> dict[str, Any]:
        meta = self.model.metadata
        return {
            "last_validation_date": meta.last_validation_date,
            "next_review_date":     meta.next_review_date,
            "validator":            meta.approved_by,
            "findings":             self.validation_findings,
            "outcome":              "Approved for production use" if meta.status == "Production" else meta.status,
        }

    @staticmethod
    def _build_governance_controls() -> list[str]:
        return [
            "All predictions are audit-logged with SHA-256 input hash and UTC timestamp.",
            "Model card is auto-regenerated on each deployment.",
            "PSI and Gini monitored daily; alerts routed to model owner and risk team.",
            "Annual independent validation required (SR 11-7 §IV).",
            "Model changes ≥ minor version require validation sign-off before deployment.",
            "SHAP adverse action reasons generated for all declined applications (Reg B).",
            "Disparate impact analysis run quarterly; findings reviewed by fair lending team.",
        ]
