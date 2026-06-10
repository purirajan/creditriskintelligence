"""
loan_tape.py
============
CSV loan tape parser and validator for community bank loan exports.

Handles the messy reality of real-world bank data:
  - FiServ Prologue column names
  - Jack Henry Symitar column names
  - Generic field aliases used by smaller core systems
  - Missing columns, nulls, type mismatches, currency formatting

Output is a clean, typed, validated DataFrame using the
CreditRisk Intelligence internal schema — ready for NCL/GCL
calculation, CECL modelling, and PD/LGD/EAD scoring.

Internal schema
---------------
Field                   Type      Description
------                  ------    -----------
loan_id                 str       Unique loan identifier
bank_id                 str       Tenant identifier (added by ingestion layer)
origination_date        date      Loan origination date
maturity_date           date      Scheduled maturity date
product_type            str       Normalised product: consumer / auto / cre /
                                  ci / heloc / bnpl / mortgage
outstanding_balance     float     Current outstanding principal (USD)
original_balance        float     Original loan amount at origination (USD)
risk_grade              str       Internal risk grade (A–G or numeric)
interest_rate           float     Annual interest rate (decimal, e.g. 0.065)
days_past_due           int       Days past due at observation date
delinquency_status      str       current / 30dpd / 60dpd / 90dpd+ / nonaccrual
charge_off_date         date      Date charged off (null if not charged off)
charge_off_amount       float     Principal charged off (USD, null if not)
recovery_date           date      Date of recovery payment (null if none)
recovery_amount         float     Recovery received post charge-off (USD, null)
collateral_value        float     Current collateral value (USD, 0 for unsecured)
secured_flag            int       1 = secured, 0 = unsecured
fico_score              float     FICO score at origination (null if unavailable)
dti_ratio               float     Debt-to-income at origination (0-1)
observation_date        date      Date of the snapshot (reporting period end)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column alias maps
# Each key is our internal name; values are aliases found in real exports
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, list[str]] = {
    "loan_id": [
        "loan_id", "loanid", "loan_number", "loannumber", "account_number",
        "acct_num", "account_no", "loan_no", "id", "loan_ref",
        # FiServ Prologue
        "LOAN_NBR", "LN_NBR",
        # Jack Henry Symitar
        "ACCOUNT_ID", "SHARE_ID",
    ],
    "origination_date": [
        "origination_date", "orig_date", "open_date", "opendate",
        "loan_date", "funding_date", "booking_date",
        "ORIG_DT", "OPEN_DT", "FUND_DT",
    ],
    "maturity_date": [
        "maturity_date", "matdate", "mat_date", "due_date", "expiry_date",
        "final_payment_date", "term_date",
        "MAT_DT", "MATURITY_DT",
    ],
    "product_type": [
        "product_type", "product", "loan_type", "loantype", "product_code",
        "loan_category", "category", "type",
        "PROD_TYPE", "LOAN_TYPE_CD",
    ],
    "outstanding_balance": [
        "outstanding_balance", "balance", "current_balance", "prin_balance",
        "principal_balance", "outstanding", "loan_balance", "unpaid_balance",
        "upb", "current_upb", "ending_balance",
        "CUR_BAL", "PRIN_BAL", "OUT_BAL",
    ],
    "original_balance": [
        "original_balance", "orig_balance", "original_amount", "loan_amount",
        "principal", "face_amount", "approved_amount",
        "ORIG_AMT", "ORIG_BAL",
    ],
    "risk_grade": [
        "risk_grade", "grade", "risk_rating", "internal_rating", "credit_grade",
        "loan_grade", "rating",
        "RISK_GRD", "LOAN_GRD", "CREDIT_RTG",
    ],
    "interest_rate": [
        "interest_rate", "rate", "note_rate", "coupon_rate", "apr",
        "annual_rate", "int_rate",
        "INT_RATE", "NOTE_RATE",
    ],
    "days_past_due": [
        "days_past_due", "dpd", "days_delinquent", "delinquency_days",
        "past_due_days", "days_overdue",
        "DAYS_DLQ", "DPD",
    ],
    "delinquency_status": [
        "delinquency_status", "status", "loan_status", "acct_status",
        "payment_status", "delinquency_bucket",
        "DLQ_STATUS", "LOAN_STAT",
    ],
    "charge_off_date": [
        "charge_off_date", "chargeoff_date", "co_date", "writeoff_date",
        "charge_off_dt", "charged_off_date",
        "CO_DT", "CHARGEOFF_DT",
    ],
    "charge_off_amount": [
        "charge_off_amount", "chargeoff_amount", "co_amount", "writeoff_amount",
        "charged_off_amount", "co_principal",
        "CO_AMT", "CHARGEOFF_AMT",
    ],
    "recovery_date": [
        "recovery_date", "recovery_dt", "recovered_date",
        "RECOVERY_DT",
    ],
    "recovery_amount": [
        "recovery_amount", "recovery", "recovered_amount", "recovery_amt",
        "collections_received",
        "RECOVERY_AMT",
    ],
    "collateral_value": [
        "collateral_value", "collateral", "appraised_value", "appraisal_value",
        "property_value", "vehicle_value", "col_value",
        "COLLAT_VAL", "APPR_VAL",
    ],
    "secured_flag": [
        "secured_flag", "secured", "is_secured", "collateralised",
        "collateralized", "secured_ind",
        "SECURED_IND",
    ],
    "fico_score": [
        "fico_score", "fico", "credit_score", "beacon_score", "vantage_score",
        "origination_fico", "orig_fico",
        "FICO_SCR", "CREDIT_SCR",
    ],
    "dti_ratio": [
        "dti_ratio", "dti", "debt_to_income", "debt_income_ratio",
        "DTI", "DEBT_INC_RATIO",
    ],
    "observation_date": [
        "observation_date", "report_date", "as_of_date", "snapshot_date",
        "period_end_date", "reporting_date",
        "RPT_DT", "AS_OF_DT",
    ],
}

# Normalised product type map (raw value → internal canonical name)
PRODUCT_MAP: dict[str, str] = {
    # Consumer
    "consumer": "consumer", "personal": "consumer", "personal loan": "consumer",
    "installment": "consumer", "unsecured": "consumer", "pl": "consumer",
    # Auto
    "auto": "auto", "vehicle": "auto", "auto loan": "auto",
    "car": "auto", "auto_loan": "auto",
    # CRE
    "cre": "cre", "commercial real estate": "cre", "commercial_re": "cre",
    "commercial re": "cre", "non-residential": "cre",
    # C&I
    "ci": "ci", "c&i": "ci", "commercial": "ci",
    "commercial industrial": "ci", "business": "ci",
    # HELOC
    "heloc": "heloc", "home equity": "heloc", "home_equity": "heloc",
    "equity line": "heloc", "hel": "heloc",
    # Mortgage
    "mortgage": "mortgage", "residential": "mortgage",
    "1-4 family": "mortgage", "single family": "mortgage",
    # BNPL
    "bnpl": "bnpl", "buy now pay later": "bnpl", "point of sale": "bnpl",
    "pos": "bnpl",
    # Credit card
    "credit card": "credit_card", "card": "credit_card",
    "revolving": "credit_card",
}

# Required columns — ingestion fails if these are absent
REQUIRED_COLUMNS = [
    "loan_id",
    "origination_date",
    "outstanding_balance",
]

# Columns with default values when absent
COLUMN_DEFAULTS: dict[str, Any] = {
    "charge_off_date":   None,
    "charge_off_amount": 0.0,
    "recovery_date":     None,
    "recovery_amount":   0.0,
    "collateral_value":  0.0,
    "secured_flag":      0,
    "days_past_due":     0,
    "risk_grade":        "Ungraded",
    "delinquency_status":"current",
    "fico_score":        np.nan,
    "dti_ratio":         np.nan,
    "interest_rate":     np.nan,
    "original_balance":  np.nan,
}


# ---------------------------------------------------------------------------
# Result / validation types
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    severity: str          # "error" | "warning" | "info"
    field: str
    message: str
    row_count: int = 0


@dataclass
class IngestionResult:
    """Returned by LoanTapeParser.parse()."""
    success: bool
    df: pd.DataFrame | None
    bank_id: str
    observation_date: date | None
    row_count: int
    column_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    unmapped_columns: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"IngestionResult [{status}] "
            f"bank={self.bank_id} "
            f"rows={self.row_count:,} "
            f"errors={self.error_count} "
            f"warnings={self.warning_count}"
        )


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class LoanTapeParser:
    """
    Parse and validate a community bank loan tape CSV.

    Handles FiServ Prologue, Jack Henry Symitar, and generic exports.
    Returns a clean DataFrame using the CreditRisk Intelligence internal schema.

    Usage
    -----
    >>> parser = LoanTapeParser(bank_id="FIRST_NATIONAL_001")
    >>> result = parser.parse("path/to/loan_tape_q4_2024.csv")
    >>> if result.success:
    ...     df = result.df
    ...     print(f"Loaded {result.row_count:,} loans")
    """

    def __init__(
        self,
        bank_id: str,
        observation_date: date | str | None = None,
        strict_mode: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        bank_id          : unique bank identifier (tenant key)
        observation_date : reporting period end date; inferred from filename
                           or set to today if not provided
        strict_mode      : if True, warnings become errors
        """
        self.bank_id = bank_id
        self.observation_date = self._coerce_date(observation_date)
        self.strict_mode = strict_mode
        self._issues: list[ValidationIssue] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(
        self,
        source: str | Path | pd.DataFrame,
        encoding: str = "utf-8",
        **read_csv_kwargs: Any,
    ) -> IngestionResult:
        """
        Parse a loan tape from CSV path or DataFrame.

        Steps:
          1. Load raw data
          2. Map column names to internal schema
          3. Clean and cast types
          4. Validate ranges and business rules
          5. Derive computed fields
          6. Return IngestionResult
        """
        self._issues = []

        # --- load ---
        try:
            raw = self._load(source, encoding, **read_csv_kwargs)
        except Exception as exc:
            return IngestionResult(
                success=False, df=None,
                bank_id=self.bank_id,
                observation_date=self.observation_date,
                row_count=0, column_count=0,
                issues=[ValidationIssue("error", "file", str(exc))],
            )

        logger.info("[%s] Loaded %d rows × %d columns", self.bank_id, *raw.shape)

        # --- map columns ---
        df, unmapped = self._map_columns(raw)

        # --- check required columns ---
        missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            self._issues.append(ValidationIssue(
                "error", "schema",
                f"Required columns missing after mapping: {missing_required}"
            ))
            return IngestionResult(
                success=False, df=None,
                bank_id=self.bank_id,
                observation_date=self.observation_date,
                row_count=len(df), column_count=len(df.columns),
                issues=self._issues, unmapped_columns=unmapped,
            )

        # --- fill defaults for optional columns ---
        for col, default in COLUMN_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        # --- clean types ---
        df = self._clean_types(df)

        # --- validate ---
        self._validate(df)

        # --- derive fields ---
        df = self._derive_fields(df)

        # --- attach tenant + observation date ---
        df["bank_id"] = self.bank_id
        if "observation_date" not in df.columns or df["observation_date"].isna().all():
            obs = self.observation_date or date.today()
            df["observation_date"] = obs
            if self.observation_date is None:
                logger.warning(
                    "[%s] observation_date not found — defaulting to today (%s)",
                    self.bank_id, obs,
                )

        success = self.error_count == 0
        summary = self._build_summary(df)

        logger.info(
            "[%s] ingestion %s — %d rows  errors=%d  warnings=%d",
            self.bank_id,
            "SUCCESS" if success else "FAILED",
            len(df),
            self.error_count,
            self.warning_count,
        )

        return IngestionResult(
            success=success,
            df=df if success else None,
            bank_id=self.bank_id,
            observation_date=df["observation_date"].iloc[0] if "observation_date" in df.columns else self.observation_date,
            row_count=len(df),
            column_count=len(df.columns),
            issues=self._issues,
            unmapped_columns=unmapped,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(
        self,
        source: str | Path | pd.DataFrame,
        encoding: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Loan tape not found: {path}")

        # Try to infer observation date from filename
        # e.g. loan_tape_2024Q4.csv  or  loans_20241231.csv
        if self.observation_date is None:
            self.observation_date = self._date_from_filename(path.stem)

        # Detect delimiter
        sample = path.read_text(encoding=encoding, errors="replace")[:4096]
        delimiter = "," if sample.count(",") > sample.count("|") else "|"

        df = pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            low_memory=False,
            dtype=str,
            **kwargs,
        )
        # Strip BOM and whitespace from column names
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Column mapping
    # ------------------------------------------------------------------

    def _map_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Map raw column names → internal schema.
        Case-insensitive. Logs unmapped columns as info.
        """
        rename: dict[str, str] = {}
        raw_cols_lower = {c.lower().strip(): c for c in df.columns}

        for internal_name, aliases in COLUMN_ALIASES.items():
            if internal_name in df.columns:
                continue   # already correct
            for alias in aliases:
                key = alias.lower().strip()
                if key in raw_cols_lower:
                    rename[raw_cols_lower[key]] = internal_name
                    break

        df = df.rename(columns=rename)

        mapped = set(rename.values()) | (set(df.columns) & set(COLUMN_ALIASES.keys()))
        unmapped = [
            c for c in df.columns
            if c not in COLUMN_ALIASES and c not in mapped
        ]
        if unmapped:
            logger.info("[%s] unmapped columns (ignored): %s", self.bank_id, unmapped)

        return df, unmapped

    # ------------------------------------------------------------------
    # Type cleaning
    # ------------------------------------------------------------------

    def _clean_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- currency / numeric ---
        money_cols = [
            "outstanding_balance", "original_balance",
            "charge_off_amount", "recovery_amount", "collateral_value",
        ]
        for col in money_cols:
            if col in df.columns:
                df[col] = self._parse_currency(df[col])

        rate_cols = ["interest_rate", "dti_ratio"]
        for col in rate_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Convert percentage to decimal (e.g. 6.5 → 0.065)
                if col == "interest_rate":
                    mask = df[col] > 1.0
                    df.loc[mask, col] = df.loc[mask, col] / 100.0

        int_cols = ["days_past_due", "secured_flag"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        float_cols = ["fico_score"]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- dates ---
        date_cols = [
            "origination_date", "maturity_date",
            "charge_off_date", "recovery_date", "observation_date",
        ]
        for col in date_cols:
            if col in df.columns:
                df[col] = self._parse_dates(df[col])

        # --- product type normalisation ---
        if "product_type" in df.columns:
            df["product_type"] = (
                df["product_type"]
                .str.lower()
                .str.strip()
                .map(PRODUCT_MAP)
                .fillna("other")
            )

        # --- delinquency status normalisation ---
        if "delinquency_status" in df.columns:
            df["delinquency_status"] = self._normalise_delinquency(
                df["delinquency_status"], df.get("days_past_due")
            )

        # --- string cleanup ---
        df["loan_id"] = df["loan_id"].astype(str).str.strip()
        if "risk_grade" in df.columns:
            df["risk_grade"] = df["risk_grade"].astype(str).str.upper().str.strip()

        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        n = len(df)

        # Required non-null
        null_loan_ids = df["loan_id"].isna().sum()
        if null_loan_ids > 0:
            self._add_issue("error", "loan_id",
                f"{null_loan_ids} rows have null loan_id", null_loan_ids)

        null_balances = df["outstanding_balance"].isna().sum()
        if null_balances > 0:
            self._add_issue("warning", "outstanding_balance",
                f"{null_balances} rows have null outstanding_balance", null_balances)

        # Duplicate loan IDs
        dupes = df["loan_id"].duplicated().sum()
        if dupes > 0:
            self._add_issue("warning", "loan_id",
                f"{dupes} duplicate loan_id values found", dupes)

        # Balance ranges
        if "outstanding_balance" in df.columns:
            neg = (df["outstanding_balance"] < 0).sum()
            if neg > 0:
                self._add_issue("warning", "outstanding_balance",
                    f"{neg} loans have negative outstanding balance", neg)

            very_large = (df["outstanding_balance"] > 50_000_000).sum()
            if very_large > 0:
                self._add_issue("info", "outstanding_balance",
                    f"{very_large} loans exceed $50M — verify these are correct",
                    very_large)

        # FICO range
        if "fico_score" in df.columns:
            fico_valid = df["fico_score"].dropna()
            bad_fico = ((fico_valid < 300) | (fico_valid > 850)).sum()
            if bad_fico > 0:
                self._add_issue("warning", "fico_score",
                    f"{bad_fico} FICO scores outside [300, 850]", bad_fico)

        # Origination date sanity
        if "origination_date" in df.columns:
            future_orig = (df["origination_date"] > pd.Timestamp.today()).sum()
            if future_orig > 0:
                self._add_issue("warning", "origination_date",
                    f"{future_orig} loans have future origination dates", future_orig)

            very_old = (df["origination_date"] < pd.Timestamp("1980-01-01")).sum()
            if very_old > 0:
                self._add_issue("info", "origination_date",
                    f"{very_old} loans originated before 1980 — verify", very_old)

        # Charge-off consistency
        if "charge_off_date" in df.columns and "charge_off_amount" in df.columns:
            has_date_no_amt = (
                df["charge_off_date"].notna() & (df["charge_off_amount"] == 0)
            ).sum()
            if has_date_no_amt > 0:
                self._add_issue("warning", "charge_off_amount",
                    f"{has_date_no_amt} loans have charge-off date but zero amount",
                    has_date_no_amt)

        # Recovery without charge-off
        if "recovery_amount" in df.columns and "charge_off_date" in df.columns:
            recovery_no_co = (
                (df["recovery_amount"] > 0) & df["charge_off_date"].isna()
            ).sum()
            if recovery_no_co > 0:
                self._add_issue("warning", "recovery_amount",
                    f"{recovery_no_co} loans show recovery but no charge-off date",
                    recovery_no_co)

        # DTI range
        if "dti_ratio" in df.columns:
            dti = df["dti_ratio"].dropna()
            bad_dti = ((dti < 0) | (dti > 2.0)).sum()
            if bad_dti > 0:
                self._add_issue("warning", "dti_ratio",
                    f"{bad_dti} DTI values outside [0, 2.0]", bad_dti)

        # Portfolio completeness
        charge_off_pct = (
            df["charge_off_date"].notna().sum() / n * 100
            if "charge_off_date" in df.columns else 0
        )
        if charge_off_pct > 30:
            self._add_issue("info", "charge_off_date",
                f"High charge-off rate in tape: {charge_off_pct:.1f}% of loans — "
                "confirm this is a defaulted portfolio extract, not a full loan tape.")

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------

    def _derive_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Months on books
        if "origination_date" in df.columns:
            obs = pd.Timestamp(self.observation_date or date.today())
            df["months_on_book"] = (
                (obs - df["origination_date"])
                .dt.days
                .div(30.44)
                .clip(lower=0)
                .round(1)
            )

        # Loan-to-value at origination (for secured loans)
        if "original_balance" in df.columns and "collateral_value" in df.columns:
            df["origination_ltv"] = np.where(
                df["collateral_value"] > 0,
                (df["original_balance"] / df["collateral_value"]).clip(0, 5),
                0.0,
            ).round(4)

        # Current utilisation (for revolving — approximated)
        if "outstanding_balance" in df.columns and "original_balance" in df.columns:
            df["utilization_rate"] = np.where(
                df["original_balance"] > 0,
                (df["outstanding_balance"] / df["original_balance"]).clip(0, 1),
                0.0,
            ).round(4)

        # Charged-off flag
        df["charged_off"] = df["charge_off_date"].notna().astype(int)

        # Net loss (charge-off minus recovery)
        df["net_loss"] = (
            df.get("charge_off_amount", 0).fillna(0)
            - df.get("recovery_amount", 0).fillna(0)
        ).clip(lower=0)

        # Origination year / quarter for vintage analysis
        if "origination_date" in df.columns:
            df["origination_year"]    = df["origination_date"].dt.year
            df["origination_quarter"] = df["origination_date"].dt.to_period("Q").astype(str)

        return df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        n = len(df)
        total_balance = df["outstanding_balance"].sum() if "outstanding_balance" in df.columns else 0
        charged_off_n = df.get("charged_off", pd.Series(0, index=df.index)).sum()
        total_co_amt  = df.get("charge_off_amount", pd.Series(0, index=df.index)).fillna(0).sum()
        total_rec_amt = df.get("recovery_amount",   pd.Series(0, index=df.index)).fillna(0).sum()

        by_product: dict[str, int] = {}
        if "product_type" in df.columns:
            by_product = df["product_type"].value_counts().to_dict()

        return {
            "total_loans":          n,
            "total_balance_usd":    round(float(total_balance), 2),
            "charged_off_loans":    int(charged_off_n),
            "charge_off_rate":      round(float(charged_off_n / n), 5) if n else 0,
            "gross_charge_off_usd": round(float(total_co_amt), 2),
            "total_recovery_usd":   round(float(total_rec_amt), 2),
            "net_charge_off_usd":   round(float(total_co_amt - total_rec_amt), 2),
            "loans_by_product":     by_product,
            "observation_date":     str(self.observation_date or ""),
        }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @property
    def error_count(self) -> int:
        return sum(1 for i in self._issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self._issues
                   if i.severity in ("warning", "error" if self.strict_mode else "warning"))

    def _add_issue(
        self, severity: str, field: str, message: str, row_count: int = 0
    ) -> None:
        self._issues.append(ValidationIssue(severity, field, message, row_count))
        if severity == "error":
            logger.error("[%s] %s: %s", self.bank_id, field, message)
        elif severity == "warning":
            logger.warning("[%s] %s: %s", self.bank_id, field, message)
        else:
            logger.info("[%s] %s: %s", self.bank_id, field, message)

    @staticmethod
    def _parse_currency(series: pd.Series) -> pd.Series:
        """Strip $, commas, parentheses from currency strings then coerce to float."""
        cleaned = (
            series.astype(str)
            .str.replace(r"[\$,\s]", "", regex=True)
            .str.replace(r"\(([0-9.]+)\)", r"-\1", regex=True)  # (1234) → -1234
        )
        return pd.to_numeric(cleaned, errors="coerce")

    @staticmethod
    def _parse_dates(series: pd.Series) -> pd.Series:
        """Flexible date parser that handles multiple formats."""
        return pd.to_datetime(series, errors="coerce")

    @staticmethod
    def _normalise_delinquency(
        status_col: pd.Series,
        dpd_col: pd.Series | None,
    ) -> pd.Series:
        """Map raw delinquency status strings to canonical buckets."""
        mapping = {
            "current": "current", "ok": "current", "0": "current",
            "30": "30dpd", "30 dpd": "30dpd", "30-59": "30dpd",
            "60": "60dpd", "60 dpd": "60dpd", "60-89": "60dpd",
            "90": "90dpd+", "90+": "90dpd+", "90 dpd": "90dpd+",
            "nonaccrual": "nonaccrual", "non-accrual": "nonaccrual",
            "charged off": "charged_off", "charged-off": "charged_off",
            "co": "charged_off",
        }
        normalised = (
            status_col.astype(str)
            .str.lower()
            .str.strip()
            .map(mapping)
        )
        # Fill from DPD if status unmapped
        if dpd_col is not None:
            mask = normalised.isna()
            dpd = pd.to_numeric(dpd_col, errors="coerce").fillna(0)
            def _dpd_bucket(d):
                if d <= 0:  return "current"
                if d <= 29: return "current"
                if d <= 59: return "30dpd"
                if d <= 89: return "60dpd"
                return "90dpd+"
            normalised = normalised.where(~mask, dpd.map(_dpd_bucket))
        return normalised.fillna("current")

    @staticmethod
    def _coerce_date(d: date | str | None) -> date | None:
        if d is None:
            return None
        if isinstance(d, date):
            return d
        try:
            return datetime.strptime(str(d), "%Y-%m-%d").date()
        except ValueError:
            return None

    @staticmethod
    def _date_from_filename(stem: str) -> date | None:
        """Try to extract a date from a filename like loan_tape_20241231."""
        patterns = [
            r"(\d{4})(Q[1-4])",     # 2024Q4
            r"(\d{4})-(\d{2})",     # 2024-12
            r"(\d{8})",             # 20241231
            r"(\d{4})",             # 2024
        ]
        for pat in patterns:
            m = re.search(pat, stem)
            if m:
                try:
                    s = m.group(0)
                    if "Q" in s:
                        year, q = int(s[:4]), int(s[-1])
                        month = q * 3
                        return date(year, month, 1)
                    if len(s) == 8:
                        return datetime.strptime(s, "%Y%m%d").date()
                    if len(s) == 7 and "-" in s:
                        return datetime.strptime(s + "-01", "%Y-%m-%d").date()
                    if len(s) == 4:
                        return date(int(s), 12, 31)
                except ValueError:
                    continue
        return None


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def load_loan_tape(
    path: str | Path | pd.DataFrame,
    bank_id: str,
    observation_date: date | str | None = None,
    strict_mode: bool = False,
) -> IngestionResult:
    """
    One-call convenience wrapper for LoanTapeParser.

    Usage
    -----
    >>> result = load_loan_tape("q4_2024_loans.csv", bank_id="COMMUNITY_BANK_001")
    >>> df = result.df
    """
    parser = LoanTapeParser(
        bank_id=bank_id,
        observation_date=observation_date,
        strict_mode=strict_mode,
    )
    return parser.parse(path)
