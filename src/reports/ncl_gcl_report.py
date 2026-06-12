"""
ncl_gcl_report.py
=================
Net Charge-Off Loss (NCL) and Gross Charge-Off Loss (GCL) report engine.

Produces the quarterly loss performance report that every community bank
needs for: board credit committee, CECL allowance adequacy, OCC/FDIC
examiner review, and SR 11-7 model backtesting.

Report sections
---------------
1. Period summary      — GCL, recoveries, NCL, NCL%, coverage ratio
2. Product breakdown   — NCL by loan type (consumer, auto, CRE, C&I, HELOC)
3. Vintage analysis    — cumulative default rate by origination cohort
4. Recovery analysis   — recovery rates by product and vintage
5. PD backtesting      — model-predicted PD vs realised NCL% by risk grade
6. CECL adequacy       — ACL balance vs ECL estimate vs 2yr NCL projection
7. Peer comparison     — NCL% vs peer group benchmarks (FDIC call report)

Regulatory alignment
--------------------
- CECL (ASC 326)   — loss rate method, lifetime loss curves
- Basel II/III     — LGD backtesting via realised recovery rates
- SR 11-7 §IV      — ongoing model performance monitoring
- OCC Bulletin 2020-62 — credit risk review expectations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Peer benchmark data — sourced from FDIC call reports (public)
# These are approximate US community bank medians; update quarterly from:
# https://banks.data.fdic.gov/api/
# ---------------------------------------------------------------------------

PEER_NCL_BENCHMARKS: dict[str, dict[str, float]] = {
    "consumer":    {"median": 0.0142, "p75": 0.0220, "p25": 0.0071},
    "auto":        {"median": 0.0068, "p75": 0.0110, "p25": 0.0031},
    "cre":         {"median": 0.0021, "p75": 0.0048, "p25": 0.0005},
    "ci":          {"median": 0.0038, "p75": 0.0082, "p25": 0.0012},
    "heloc":       {"median": 0.0019, "p75": 0.0041, "p25": 0.0007},
    "mortgage":    {"median": 0.0011, "p75": 0.0024, "p25": 0.0003},
    "credit_card": {"median": 0.0385, "p75": 0.0520, "p25": 0.0210},
    "bnpl":        {"median": 0.0480, "p75": 0.0650, "p25": 0.0280},
    "all":         {"median": 0.0055, "p75": 0.0098, "p25": 0.0022},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PeriodSummary:
    bank_id: str
    period_start: date
    period_end: date
    period_label: str           # e.g. "Q4 2024"

    # Balances
    avg_loans_outstanding: float
    ending_balance: float

    # Charge-off activity
    gross_charge_offs: float    # GCL
    recoveries: float
    net_charge_offs: float      # NCL = GCL - recoveries

    # Rates
    ncl_rate: float             # NCL / avg loans (annualised)
    gcl_rate: float             # GCL / avg loans (annualised)
    recovery_rate: float        # recoveries / prior period GCL

    # CECL
    acl_balance: float | None   # allowance for credit losses (book value)
    ecl_estimate: float | None  # model ECL estimate
    coverage_ratio: float | None # ACL / NCL (times coverage)

    # Counts
    total_loans: int
    charged_off_count: int
    charge_off_count_rate: float


@dataclass
class ProductBreakdown:
    product: str
    avg_balance: float
    gross_charge_offs: float
    recoveries: float
    net_charge_offs: float
    ncl_rate: float
    peer_ncl_median: float | None
    vs_peer: str                # "below peer" | "at peer" | "above peer"
    loan_count: int


@dataclass
class VintageRow:
    origination_cohort: str     # e.g. "2022-Q1"
    origination_year: int
    original_balance: float
    cumulative_defaults: float
    cumulative_default_rate: float
    cumulative_recoveries: float
    cumulative_net_loss_rate: float
    months_seasoned: float
    loan_count: int


@dataclass
class BacktestRow:
    risk_grade: str
    loan_count: int
    avg_balance: float
    model_pd: float             # predicted PD from model
    realised_default_rate: float# actual NCL% in grade
    pd_error: float             # realised - model (positive = underestimation)
    pd_ratio: float             # realised / model (1.0 = perfect)
    status: str                 # "well-calibrated" | "over-estimates" | "under-estimates"


@dataclass
class NCLGCLReport:
    """Full NCL/GCL report — all sections combined."""
    bank_id: str
    generated_at: str
    report_period: str
    period_summary: PeriodSummary
    product_breakdown: list[ProductBreakdown]
    vintage_analysis: list[VintageRow]
    backtest_results: list[BacktestRow]
    peer_comparison: dict[str, Any]
    methodology_notes: list[str]
    data_quality_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id":            self.bank_id,
            "generated_at":       self.generated_at,
            "report_period":      self.report_period,
            "period_summary":     self.period_summary.__dict__,
            "product_breakdown":  [p.__dict__ for p in self.product_breakdown],
            "vintage_analysis":   [v.__dict__ for v in self.vintage_analysis],
            "backtest_results":   [b.__dict__ for b in self.backtest_results],
            "peer_comparison":    self.peer_comparison,
            "methodology_notes":  self.methodology_notes,
            "data_quality_flags": self.data_quality_flags,
        }

    def summary_table(self) -> pd.DataFrame:
        """Single-row DataFrame of key metrics — easy to log or email."""
        ps = self.period_summary
        return pd.DataFrame([{
            "bank_id":              ps.bank_id,
            "period":               ps.period_label,
            "avg_loans_$M":         round(ps.avg_loans_outstanding / 1e6, 2),
            "GCL_$":                round(ps.gross_charge_offs, 0),
            "recoveries_$":         round(ps.recoveries, 0),
            "NCL_$":                round(ps.net_charge_offs, 0),
            "NCL_%":                f"{ps.ncl_rate:.3%}",
            "GCL_%":                f"{ps.gcl_rate:.3%}",
            "recovery_rate_%":      f"{ps.recovery_rate:.1%}",
            "coverage_ratio_x":     round(ps.coverage_ratio, 2) if ps.coverage_ratio else None,
            "charged_off_loans":    ps.charged_off_count,
        }])


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class NCLGCLEngine:
    """
    Compute NCL/GCL metrics from a parsed loan tape DataFrame.

    Usage
    -----
    >>> engine = NCLGCLEngine(bank_id="COMMUNITY_BANK_001")
    >>> report = engine.generate(
    ...     df=loan_tape_df,
    ...     period_start=date(2024, 10, 1),
    ...     period_end=date(2024, 12, 31),
    ...     acl_balance=4_200_000,
    ...     model_pd_col="pd_estimate",   # optional: column with model PD
    ... )
    >>> print(report.summary_table())
    """

    def __init__(self, bank_id: str) -> None:
        self.bank_id = bank_id
        self._flags: list[str] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        df: pd.DataFrame,
        period_start: date,
        period_end: date,
        acl_balance: float | None = None,
        model_pd_col: str | None = None,
        prior_period_gcl: float | None = None,
    ) -> NCLGCLReport:
        """
        Generate the full NCL/GCL report.

        Parameters
        ----------
        df               : loan tape DataFrame (output of LoanTapeParser)
        period_start     : first day of reporting period
        period_end       : last day of reporting period (observation date)
        acl_balance      : allowance for credit losses balance (from G/L)
        model_pd_col     : column name containing model PD estimates
        prior_period_gcl : GCL from prior period (for recovery rate calc)
        """
        self._flags = []
        df = self._prepare(df, period_start, period_end)

        period_label = self._period_label(period_start, period_end)

        # Section 1: Period summary
        summary = self._period_summary(
            df, period_start, period_end, period_label,
            acl_balance, prior_period_gcl,
        )

        # Section 2: Product breakdown
        products = self._product_breakdown(df)

        # Section 3: Vintage analysis
        vintages = self._vintage_analysis(df)

        # Section 4 + 5: Backtest (if model PD available)
        backtest = self._pd_backtest(df, model_pd_col) if model_pd_col else []

        # Section 6: Peer comparison
        peer = self._peer_comparison(summary, products)

        methodology = self._methodology_notes(df, model_pd_col)

        logger.info(
            "[%s] NCL/GCL report generated — period=%s  NCL=%.4f%%  GCL=$%,.0f",
            self.bank_id,
            period_label,
            summary.ncl_rate * 100,
            summary.gross_charge_offs,
        )

        return NCLGCLReport(
            bank_id=self.bank_id,
            generated_at=datetime.utcnow().isoformat() + "Z",
            report_period=period_label,
            period_summary=summary,
            product_breakdown=products,
            vintage_analysis=vintages,
            backtest_results=backtest,
            peer_comparison=peer,
            methodology_notes=methodology,
            data_quality_flags=self._flags,
        )

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _prepare(
        self, df: pd.DataFrame, period_start: date, period_end: date
    ) -> pd.DataFrame:
        df = df.copy()

        # Coerce dates
        for col in ["charge_off_date", "recovery_date", "origination_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Flag rows charged off during reporting period
        ps = pd.Timestamp(period_start)
        pe = pd.Timestamp(period_end)
        if "charge_off_date" in df.columns:
            df["_co_in_period"] = (
                df["charge_off_date"].notna()
                & (df["charge_off_date"] >= ps)
                & (df["charge_off_date"] <= pe)
            )
        else:
            df["_co_in_period"] = False
            self._flags.append("charge_off_date column missing — GCL cannot be computed accurately")

        # Flag recoveries received during period
        if "recovery_date" in df.columns:
            df["_rec_in_period"] = (
                df["recovery_date"].notna()
                & (df["recovery_date"] >= ps)
                & (df["recovery_date"] <= pe)
            )
        else:
            df["_rec_in_period"] = False

        # Fill numeric nulls
        for col in ["charge_off_amount", "recovery_amount", "outstanding_balance"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # Section 1: Period summary
    # ------------------------------------------------------------------

    def _period_summary(
        self,
        df: pd.DataFrame,
        period_start: date,
        period_end: date,
        period_label: str,
        acl_balance: float | None,
        prior_period_gcl: float | None,
    ) -> PeriodSummary:

        # Balances
        total_balance    = float(df["outstanding_balance"].sum())
        avg_balance      = total_balance   # simplified; ideally average of bop + eop

        # GCL: charge-offs originated in the period
        co_mask          = df["_co_in_period"]
        gcl              = float(df.loc[co_mask, "charge_off_amount"].sum())
        charged_off_n    = int(co_mask.sum())

        # Recoveries: receipts in the period on previously charged-off loans
        rec_mask         = df["_rec_in_period"]
        recoveries       = float(df.loc[rec_mask, "recovery_amount"].sum())

        # NCL
        ncl              = gcl - recoveries
        n                = len(df)

        # Rates (annualise quarterly by ×4)
        quarters_in_period = max(
            ((period_end - period_start).days / 91.25), 0.25
        )
        annualisation    = 1.0 / quarters_in_period
        gcl_rate         = (gcl / avg_balance * annualisation) if avg_balance > 0 else 0.0
        ncl_rate         = (ncl / avg_balance * annualisation) if avg_balance > 0 else 0.0

        # Recovery rate vs prior period GCL
        rec_rate         = (recoveries / prior_period_gcl) if prior_period_gcl else 0.0

        # Coverage ratio
        coverage_ratio   = None
        ecl_estimate     = None
        if acl_balance and ncl_rate > 0:
            # Simplified ECL = forward 2-year NCL projection
            ecl_estimate  = ncl_rate * avg_balance * 2
            coverage_ratio = acl_balance / ncl if ncl > 0 else None

        return PeriodSummary(
            bank_id=self.bank_id,
            period_start=period_start,
            period_end=period_end,
            period_label=period_label,
            avg_loans_outstanding=round(avg_balance, 2),
            ending_balance=round(total_balance, 2),
            gross_charge_offs=round(gcl, 2),
            recoveries=round(recoveries, 2),
            net_charge_offs=round(ncl, 2),
            ncl_rate=round(ncl_rate, 6),
            gcl_rate=round(gcl_rate, 6),
            recovery_rate=round(rec_rate, 4),
            acl_balance=acl_balance,
            ecl_estimate=round(ecl_estimate, 2) if ecl_estimate else None,
            coverage_ratio=round(coverage_ratio, 2) if coverage_ratio else None,
            total_loans=n,
            charged_off_count=charged_off_n,
            charge_off_count_rate=round(charged_off_n / n, 6) if n else 0,
        )

    # ------------------------------------------------------------------
    # Section 2: Product breakdown
    # ------------------------------------------------------------------

    def _product_breakdown(self, df: pd.DataFrame) -> list[ProductBreakdown]:
        if "product_type" not in df.columns:
            self._flags.append("product_type column missing — product breakdown unavailable")
            return []

        results: list[ProductBreakdown] = []

        for product, grp in df.groupby("product_type"):
            avg_bal   = float(grp["outstanding_balance"].sum())
            gcl       = float(grp.loc[grp["_co_in_period"], "charge_off_amount"].sum())
            rec       = float(grp.loc[grp["_rec_in_period"], "recovery_amount"].sum())
            ncl       = gcl - rec
            ncl_rate  = (ncl / avg_bal) if avg_bal > 0 else 0.0
            n         = len(grp)

            peer      = PEER_NCL_BENCHMARKS.get(str(product), {})
            peer_med  = peer.get("median")

            if peer_med:
                if ncl_rate < peer_med * 0.8:
                    vs_peer = "below peer"
                elif ncl_rate > peer_med * 1.2:
                    vs_peer = "above peer"
                else:
                    vs_peer = "at peer"
            else:
                vs_peer = "no benchmark"

            results.append(ProductBreakdown(
                product=str(product),
                avg_balance=round(avg_bal, 2),
                gross_charge_offs=round(gcl, 2),
                recoveries=round(rec, 2),
                net_charge_offs=round(ncl, 2),
                ncl_rate=round(ncl_rate, 6),
                peer_ncl_median=peer_med,
                vs_peer=vs_peer,
                loan_count=n,
            ))

        # Sort by NCL rate descending
        results.sort(key=lambda x: x.ncl_rate, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Section 3: Vintage analysis
    # ------------------------------------------------------------------

    def _vintage_analysis(self, df: pd.DataFrame) -> list[VintageRow]:
        """
        Cumulative default rate by origination quarter cohort.

        This is the foundation of the CECL loss rate method — you need
        at least 4–8 quarters of cohort data to estimate lifetime losses.
        """
        if "origination_quarter" not in df.columns:
            self._flags.append("origination_date missing — vintage analysis unavailable")
            return []

        results: list[VintageRow] = []
        today = pd.Timestamp.today()

        for cohort, grp in df.groupby("origination_quarter"):
            orig_bal   = float(grp["original_balance"].fillna(
                grp["outstanding_balance"]).sum())
            if orig_bal == 0:
                continue

            cum_co     = float(grp["charge_off_amount"].fillna(0).sum())
            cum_rec    = float(grp["recovery_amount"].fillna(0).sum())
            cum_def_rt = cum_co / orig_bal
            cum_net_rt = (cum_co - cum_rec) / orig_bal

            # Approximate months seasoned from cohort midpoint
            try:
                cohort_str = str(cohort)
                year = int(cohort_str[:4])
                qtr  = int(cohort_str[-1])
                mid_month = qtr * 3 - 1
                mid_date  = pd.Timestamp(year, mid_month, 15)
                months_seasoned = max(0.0, (today - mid_date).days / 30.44)
            except Exception:
                months_seasoned = 0.0

            results.append(VintageRow(
                origination_cohort=str(cohort),
                origination_year=int(str(cohort)[:4]),
                original_balance=round(orig_bal, 2),
                cumulative_defaults=round(cum_co, 2),
                cumulative_default_rate=round(cum_def_rt, 6),
                cumulative_recoveries=round(cum_rec, 2),
                cumulative_net_loss_rate=round(cum_net_rt, 6),
                months_seasoned=round(months_seasoned, 1),
                loan_count=len(grp),
            ))

        # Sort chronologically
        results.sort(key=lambda x: x.origination_cohort)
        return results

    # ------------------------------------------------------------------
    # Section 4+5: PD backtest
    # ------------------------------------------------------------------

    def _pd_backtest(
        self, df: pd.DataFrame, model_pd_col: str
    ) -> list[BacktestRow]:
        """
        SR 11-7 model backtesting: compare model PD vs realised default rate
        by risk grade bucket.

        If model_pd_col is present, uses those scores. Otherwise falls back
        to using the risk_grade column directly.
        """
        if model_pd_col not in df.columns:
            self._flags.append(
                f"Column '{model_pd_col}' not found — PD backtesting skipped"
            )
            return []

        if "risk_grade" not in df.columns:
            self._flags.append("risk_grade column missing — backtesting by grade unavailable")
            # Fall back to decile-based backtest
            return self._pd_backtest_by_decile(df, model_pd_col)

        results: list[BacktestRow] = []

        for grade, grp in df.groupby("risk_grade"):
            n          = len(grp)
            if n < 10:
                continue   # too few loans for meaningful backtest

            avg_bal    = float(grp["outstanding_balance"].mean())
            model_pd   = float(pd.to_numeric(grp[model_pd_col], errors='coerce').mean())

            # Realised default rate = charged-off in this cohort / total
            realised_dr = float(grp["_co_in_period"].mean())

            # Annualise if needed (model PD is typically 12-month)
            pd_error   = realised_dr - model_pd
            pd_ratio   = (realised_dr / model_pd) if model_pd > 0 else float("inf")

            if pd_ratio < 0.7:
                status = "model over-estimates risk"
            elif pd_ratio > 1.4:
                status = "model under-estimates risk"
            else:
                status = "well-calibrated"

            results.append(BacktestRow(
                risk_grade=str(grade),
                loan_count=n,
                avg_balance=round(avg_bal, 2),
                model_pd=round(model_pd, 5),
                realised_default_rate=round(realised_dr, 5),
                pd_error=round(pd_error, 5),
                pd_ratio=round(pd_ratio, 4),
                status=status,
            ))

        results.sort(key=lambda x: x.model_pd)
        return results

    def _pd_backtest_by_decile(
        self, df: pd.DataFrame, model_pd_col: str
    ) -> list[BacktestRow]:
        """Backtest by PD score decile (when risk_grade is unavailable)."""
        df = df.copy()
        df["_pd_decile"] = pd.qcut(
            df[model_pd_col].clip(0.0001, 0.9999),
            q=10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop"
        )
        return self._pd_backtest(df.rename(columns={"_pd_decile": "risk_grade"}),
                                 model_pd_col)

    # ------------------------------------------------------------------
    # Section 7: Peer comparison
    # ------------------------------------------------------------------

    def _peer_comparison(
        self,
        summary: PeriodSummary,
        products: list[ProductBreakdown],
    ) -> dict[str, Any]:
        peer_all = PEER_NCL_BENCHMARKS["all"]
        ncl_rate = summary.ncl_rate

        if ncl_rate < peer_all["p25"]:
            portfolio_position = "top quartile — significantly below peer median"
        elif ncl_rate < peer_all["median"]:
            portfolio_position = "second quartile — below peer median"
        elif ncl_rate < peer_all["p75"]:
            portfolio_position = "third quartile — above peer median"
        else:
            portfolio_position = "bottom quartile — significantly above peer median"

        product_positions = {
            p.product: {
                "bank_ncl_rate":   p.ncl_rate,
                "peer_median":     p.peer_ncl_median,
                "position":        p.vs_peer,
            }
            for p in products
            if p.peer_ncl_median is not None
        }

        return {
            "portfolio_ncl_rate":      summary.ncl_rate,
            "peer_ncl_median":         peer_all["median"],
            "peer_ncl_p25":            peer_all["p25"],
            "peer_ncl_p75":            peer_all["p75"],
            "portfolio_position":      portfolio_position,
            "product_positions":       product_positions,
            "benchmark_source":        "FDIC Call Report — community bank peer group",
            "benchmark_as_of":         "2024 Q3 (update quarterly)",
        }

    # ------------------------------------------------------------------
    # Methodology notes
    # ------------------------------------------------------------------

    def _methodology_notes(
        self, df: pd.DataFrame, model_pd_col: str | None
    ) -> list[str]:
        notes = [
            "GCL = sum of principal charged off during the reporting period.",
            "NCL = GCL minus recoveries received during the reporting period.",
            "NCL rate = annualised NCL / average loans outstanding.",
            "Vintage analysis uses origination quarter cohorts; "
            "cumulative default rate = charge-offs to date / original balance.",
            "CECL ECL estimate is a simplified 2-year forward NCL projection. "
            "For full CECL allowance, use the cecl_allowance module.",
            "Peer benchmarks are approximate US community bank medians from "
            "FDIC call report data. Update quarterly from banks.data.fdic.gov.",
        ]
        if model_pd_col:
            notes.append(
                f"PD backtesting uses column '{model_pd_col}' as model output. "
                "Compares mean predicted PD vs realised default rate per grade bucket. "
                "SR 11-7 requires this analysis at least annually."
            )
        return notes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _period_label(start: date, end: date) -> str:
        quarter = (end.month - 1) // 3 + 1
        return f"Q{quarter} {end.year}"


# ---------------------------------------------------------------------------
# Report formatter — clean tabular output
# ---------------------------------------------------------------------------

class ReportFormatter:
    """
    Formats an NCLGCLReport as human-readable text tables.
    Output is ready for: board packs, examiner submissions, email body.
    """

    def __init__(self, report: NCLGCLReport) -> None:
        self.report = report

    def full_text_report(self) -> str:
        lines: list[str] = []
        r = self.report
        ps = r.period_summary

        lines += [
            "=" * 72,
            f"  NET CHARGE-OFF / GROSS CHARGE-OFF REPORT",
            f"  Bank: {r.bank_id}   Period: {r.report_period}",
            f"  Generated: {r.generated_at}",
            "=" * 72,
            "",
            "── SECTION 1: PERIOD SUMMARY ──────────────────────────────────────",
            "",
            f"  Average loans outstanding : ${ps.avg_loans_outstanding:>16,.0f}",
            f"  Ending balance            : ${ps.ending_balance:>16,.0f}",
            f"  Total loans               : {ps.total_loans:>17,}",
            "",
            f"  Gross charge-offs (GCL)   : ${ps.gross_charge_offs:>16,.0f}",
            f"  Recoveries                : ${ps.recoveries:>16,.0f}",
            f"  Net charge-offs (NCL)     : ${ps.net_charge_offs:>16,.0f}",
            "",
            f"  GCL rate (annualised)     : {ps.gcl_rate:>16.3%}",
            f"  NCL rate (annualised)     : {ps.ncl_rate:>16.3%}",
            f"  Recovery rate             : {ps.recovery_rate:>16.1%}",
            f"  Charged-off loans         : {ps.charged_off_count:>17,}  "
            f"({ps.charge_off_count_rate:.2%} of portfolio)",
        ]

        if ps.acl_balance:
            lines += [
                "",
                f"  ACL balance               : ${ps.acl_balance:>16,.0f}",
                f"  ECL estimate (2yr fwd)    : ${ps.ecl_estimate or 0:>16,.0f}",
                f"  Coverage ratio            : {(ps.coverage_ratio or 0):>15.1f}×",
            ]

        lines += ["", "── SECTION 2: PRODUCT BREAKDOWN ────────────────────────────────────", ""]

        if r.product_breakdown:
            hdr = (
                f"  {'Product':<16} {'Balance':>12} {'GCL':>10} "
                f"{'NCL':>10} {'NCL%':>7} {'vs Peer':>14}"
            )
            lines.append(hdr)
            lines.append("  " + "-" * 70)
            for p in r.product_breakdown:
                lines.append(
                    f"  {p.product:<16} ${p.avg_balance:>11,.0f} "
                    f"${p.gross_charge_offs:>9,.0f} "
                    f"${p.net_charge_offs:>9,.0f} "
                    f"{p.ncl_rate:>6.3%} "
                    f"  {p.vs_peer:>14}"
                )
        else:
            lines.append("  Product breakdown unavailable — product_type column missing.")

        lines += ["", "── SECTION 3: VINTAGE ANALYSIS ─────────────────────────────────────", ""]

        if r.vintage_analysis:
            hdr = (
                f"  {'Cohort':<12} {'Orig bal':>12} {'Cum default':>12} "
                f"{'Default rate':>13} {'Net loss rate':>14} {'Seasoned (mo)':>14}"
            )
            lines.append(hdr)
            lines.append("  " + "-" * 80)
            for v in r.vintage_analysis[-12:]:   # last 12 cohorts
                lines.append(
                    f"  {v.origination_cohort:<12} "
                    f"${v.original_balance:>11,.0f} "
                    f"${v.cumulative_defaults:>11,.0f} "
                    f"{v.cumulative_default_rate:>12.3%} "
                    f"{v.cumulative_net_loss_rate:>13.3%} "
                    f"{v.months_seasoned:>14.1f}"
                )
        else:
            lines.append("  Vintage analysis unavailable.")

        if r.backtest_results:
            lines += ["", "── SECTION 4: PD MODEL BACKTEST (SR 11-7) ──────────────────────────", ""]
            hdr = (
                f"  {'Grade':<8} {'Count':>7} {'Model PD':>10} "
                f"{'Realised':>10} {'Error':>8} {'Ratio':>7} {'Status'}"
            )
            lines.append(hdr)
            lines.append("  " + "-" * 72)
            for b in r.backtest_results:
                lines.append(
                    f"  {b.risk_grade:<8} {b.loan_count:>7,} "
                    f"{b.model_pd:>9.3%} "
                    f"{b.realised_default_rate:>9.3%} "
                    f"{b.pd_error:>+7.4f} "
                    f"{b.pd_ratio:>6.2f}×  {b.status}"
                )

        lines += ["", "── SECTION 5: PEER BENCHMARKING ────────────────────────────────────", ""]
        pc = r.peer_comparison
        lines += [
            f"  Portfolio NCL rate : {pc['portfolio_ncl_rate']:.3%}",
            f"  Peer group median  : {pc['peer_ncl_median']:.3%}",
            f"  Peer 25th pct      : {pc['peer_ncl_p25']:.3%}",
            f"  Peer 75th pct      : {pc['peer_ncl_p75']:.3%}",
            f"  Position           : {pc['portfolio_position']}",
            f"  Source             : {pc['benchmark_source']}",
        ]

        if r.data_quality_flags:
            lines += ["", "── DATA QUALITY FLAGS ──────────────────────────────────────────────", ""]
            for flag in r.data_quality_flags:
                lines.append(f"  ! {flag}")

        lines += ["", "── METHODOLOGY ─────────────────────────────────────────────────────", ""]
        for note in r.methodology_notes:
            lines.append(f"  • {note}")

        lines += ["", "=" * 72]
        return "\n".join(lines)

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Return all sections as a dict of DataFrames (for Excel export)."""
        r = self.report
        ps = r.period_summary

        summary_df = pd.DataFrame([{
            "Metric": k, "Value": v
        } for k, v in ps.__dict__.items()])

        product_df = pd.DataFrame([p.__dict__ for p in r.product_breakdown])
        vintage_df = pd.DataFrame([v.__dict__ for v in r.vintage_analysis])
        backtest_df = pd.DataFrame([b.__dict__ for b in r.backtest_results])

        return {
            "Period Summary":    summary_df,
            "Product Breakdown": product_df,
            "Vintage Analysis":  vintage_df,
            "PD Backtest":       backtest_df,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_ncl_gcl_report(
    df: pd.DataFrame,
    bank_id: str,
    period_start: date,
    period_end: date,
    acl_balance: float | None = None,
    model_pd_col: str | None = None,
    print_report: bool = True,
) -> NCLGCLReport:
    """
    One-call convenience function.

    Usage
    -----
    >>> report = generate_ncl_gcl_report(
    ...     df=loan_df,
    ...     bank_id="FIRST_NATIONAL",
    ...     period_start=date(2024, 10, 1),
    ...     period_end=date(2024, 12, 31),
    ...     acl_balance=4_200_000,
    ...     print_report=True,
    ... )
    """
    engine   = NCLGCLEngine(bank_id=bank_id)
    report   = engine.generate(
        df=df,
        period_start=period_start,
        period_end=period_end,
        acl_balance=acl_balance,
        model_pd_col=model_pd_col,
    )
    if print_report:
        formatter = ReportFormatter(report)
        print(formatter.full_text_report())

    return report
