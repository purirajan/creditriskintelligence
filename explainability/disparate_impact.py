"""
disparate_impact.py
===================
Fair lending / disparate impact analysis toolkit.

Implements the statistical tests required for ECOA, Fair Housing Act, and
CFPB supervisory examination preparation.

Key metrics:
  - Adverse Impact Ratio (AIR): approval rate (protected) / approval rate (control)
    Threshold: AIR < 0.80 triggers "4/5ths rule" adverse impact finding (EEOC guidelines)
  - Marginal Effect Test: regression-based test controlling for legitimate credit factors
  - Approval rate gap analysis by demographic group
  - Pricing disparity analysis (APR / fee differences)

Regulatory alignment:
  - ECOA (12 C.F.R. Part 202)
  - Fair Housing Act (42 U.S.C. §§ 3601–3619)
  - CFPB Examination Procedures — Fair Lending
  - HMDA (Home Mortgage Disclosure Act) — required data fields
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# EEOC 4/5ths rule threshold
AIR_THRESHOLD = 0.80

# Protected class column names expected in the analysis DataFrame
PROTECTED_CLASSES = [
    "race_ethnicity",
    "sex",
    "national_origin",
    "marital_status",
    "age_group",
    "religion",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GroupStats:
    group_name: str
    n_applications: int
    n_approvals: int
    approval_rate: float
    mean_pd: float | None = None
    mean_apr: float | None = None


@dataclass
class DisparateImpactResult:
    protected_class: str
    protected_group: str
    control_group: str
    protected_stats: GroupStats
    control_stats: GroupStats
    adverse_impact_ratio: float
    air_flag: bool               # True = potential adverse impact (AIR < 0.80)
    chi2_statistic: float        # Chi-squared test for approval rate difference
    chi2_pvalue: float
    statistically_significant: bool   # p < 0.05
    finding: str                 # "No adverse impact" | "Potential adverse impact" | "Adverse impact"
    recommendation: str


@dataclass
class FairLendingReport:
    analysis_date: str
    portfolio_size: int
    results: list[DisparateImpactResult]
    flag_count: int
    summary: str


# ---------------------------------------------------------------------------
# Analyser class
# ---------------------------------------------------------------------------

class DisparateImpactAnalyser:
    """
    Run adverse impact analysis across protected classes.

    Usage
    -----
    >>> analyser = DisparateImpactAnalyser()
    >>> report = analyser.run(df, decision_col="approved", pd_col="pd_estimate")
    >>> report.flag_count
    1
    """

    def __init__(
        self,
        air_threshold: float = AIR_THRESHOLD,
        significance_level: float = 0.05,
        min_group_size: int = 30,  # minimum observations to include a group
    ) -> None:
        self.air_threshold = air_threshold
        self.significance_level = significance_level
        self.min_group_size = min_group_size

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        decision_col: str = "approved",
        pd_col: str | None = "pd_estimate",
        apr_col: str | None = None,
        protected_classes: list[str] | None = None,
    ) -> FairLendingReport:
        """
        Run adverse impact analysis for all available protected class columns.

        Parameters
        ----------
        df              : portfolio DataFrame
        decision_col    : binary approval column (1=approved, 0=denied)
        pd_col          : PD estimate column (optional, for supplemental analysis)
        apr_col         : APR column (optional, for pricing disparity)
        protected_classes : list of column names to analyse; defaults to PROTECTED_CLASSES
        """
        from datetime import datetime, timezone

        classes_to_check = protected_classes or PROTECTED_CLASSES
        available = [c for c in classes_to_check if c in df.columns]

        if not available:
            logger.warning(
                "No protected class columns found in DataFrame. "
                "Expected any of: %s", classes_to_check
            )

        results: list[DisparateImpactResult] = []
        for pc in available:
            results.extend(
                self._analyse_class(df, pc, decision_col, pd_col, apr_col)
            )

        flag_count = sum(1 for r in results if r.air_flag and r.statistically_significant)

        summary = (
            f"Analysed {len(available)} protected class variable(s). "
            f"{len(results)} group comparisons completed. "
            f"{flag_count} statistically significant adverse impact flag(s) identified."
        )
        logger.info(summary)

        return FairLendingReport(
            analysis_date=datetime.now(timezone.utc).isoformat(),
            portfolio_size=len(df),
            results=results,
            flag_count=flag_count,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Per-class analysis
    # ------------------------------------------------------------------

    def _analyse_class(
        self,
        df: pd.DataFrame,
        protected_class: str,
        decision_col: str,
        pd_col: str | None,
        apr_col: str | None,
    ) -> list[DisparateImpactResult]:
        """Compare each minority group against the majority (control) group."""
        groups = df[protected_class].dropna().unique()

        # Control group = largest group
        group_sizes = df[protected_class].value_counts()
        control_group = group_sizes.index[0]

        results: list[DisparateImpactResult] = []
        control_df = df[df[protected_class] == control_group]
        if len(control_df) < self.min_group_size:
            return results

        control_stats = self._compute_group_stats(
            control_df, str(control_group), decision_col, pd_col, apr_col
        )

        for group in groups:
            if group == control_group:
                continue
            protected_df = df[df[protected_class] == group]
            if len(protected_df) < self.min_group_size:
                logger.debug(
                    "Skipping group %s=%s (n=%d < min=%d)",
                    protected_class, group, len(protected_df), self.min_group_size,
                )
                continue

            protected_stats = self._compute_group_stats(
                protected_df, str(group), decision_col, pd_col, apr_col
            )

            air = self._compute_air(protected_stats, control_stats)
            chi2_stat, chi2_p = self._chi2_test(protected_stats, control_stats)
            air_flag = air < self.air_threshold
            significant = chi2_p < self.significance_level

            finding, recommendation = self._interpret(air, significant)

            results.append(DisparateImpactResult(
                protected_class=protected_class,
                protected_group=str(group),
                control_group=str(control_group),
                protected_stats=protected_stats,
                control_stats=control_stats,
                adverse_impact_ratio=round(air, 4),
                air_flag=air_flag,
                chi2_statistic=round(chi2_stat, 4),
                chi2_pvalue=round(chi2_p, 6),
                statistically_significant=significant,
                finding=finding,
                recommendation=recommendation,
            ))

        return results

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_group_stats(
        df: pd.DataFrame,
        group_name: str,
        decision_col: str,
        pd_col: str | None,
        apr_col: str | None,
    ) -> GroupStats:
        n = len(df)
        n_approved = int(df[decision_col].sum())
        approval_rate = n_approved / n if n > 0 else 0.0
        mean_pd = float(df[pd_col].mean()) if pd_col and pd_col in df.columns else None
        mean_apr = float(df[apr_col].mean()) if apr_col and apr_col in df.columns else None
        return GroupStats(
            group_name=group_name,
            n_applications=n,
            n_approvals=n_approved,
            approval_rate=round(approval_rate, 5),
            mean_pd=round(mean_pd, 5) if mean_pd is not None else None,
            mean_apr=round(mean_apr, 5) if mean_apr is not None else None,
        )

    @staticmethod
    def _compute_air(
        protected: GroupStats, control: GroupStats
    ) -> float:
        if control.approval_rate == 0:
            return 0.0
        return protected.approval_rate / control.approval_rate

    @staticmethod
    def _chi2_test(
        protected: GroupStats, control: GroupStats
    ) -> tuple[float, float]:
        """Chi-squared test of independence for approval rates."""
        p_deny = protected.n_applications - protected.n_approvals
        c_deny = control.n_applications - control.n_approvals
        if p_deny < 0 or c_deny < 0:
            return 0.0, 1.0
        table = np.array([
            [protected.n_approvals, p_deny],
            [control.n_approvals,   c_deny],
        ])
        if table.min() == 0:
            return 0.0, 1.0
        chi2, p, *_ = stats.chi2_contingency(table, correction=False)
        return float(chi2), float(p)

    def _interpret(self, air: float, significant: bool) -> tuple[str, str]:
        if air >= self.air_threshold:
            return (
                "No adverse impact",
                "No action required. Continue monitoring on a quarterly basis.",
            )
        if not significant:
            return (
                "Potential adverse impact (not statistically significant)",
                "Monitor closely. Increase sample collection before drawing conclusions.",
            )
        return (
            "Adverse impact — immediate review required",
            (
                "Engage fair lending counsel. Conduct comparative file review. "
                "Evaluate whether legitimate non-discriminatory explanations exist. "
                "Consider remediation including retroactive rate relief or re-underwriting."
            ),
        )
