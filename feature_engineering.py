"""
utils/feature_engineering.py
=============================
Feature engineering utilities for credit risk models.

Provides:
  - Scorecard-style binning (Weight of Evidence / Information Value)
  - Derived feature construction (payment burden, credit depth, etc.)
  - Missing value imputation strategies for credit data
  - Feature validation and range clipping

Used in training pipelines and as pre-processors upstream of model inference.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WoE / IV binning  (standard scorecard technique)
# ---------------------------------------------------------------------------

def compute_woe_iv(
    feature: pd.Series,
    target: pd.Series,
    n_bins: int = 10,
    min_bin_pct: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    """
    Compute Weight of Evidence (WoE) and Information Value (IV).

    WoE = ln( %Events / %Non-Events )
    IV  = Σ ( %Events − %Non-Events ) × WoE

    IV benchmarks:
      < 0.02  → useless predictor
      0.02–0.10 → weak predictor
      0.10–0.30 → medium predictor
      > 0.30  → strong predictor

    Returns
    -------
    woe_table : DataFrame with bin, woe, iv_bin columns
    iv_total  : total Information Value
    """
    df = pd.DataFrame({"feature": feature, "target": target})

    # Quantile-based binning (handles skew better than equal-width)
    try:
        df["bin"] = pd.qcut(df["feature"], q=n_bins, duplicates="drop")
    except Exception:  # noqa: BLE001
        df["bin"] = pd.cut(df["feature"], bins=n_bins)

    total_events     = float(target.sum())
    total_non_events = float((1 - target).sum())

    if total_events == 0 or total_non_events == 0:
        logger.warning("compute_woe_iv: target is all-one or all-zero.")
        return pd.DataFrame(), 0.0

    stats = df.groupby("bin", observed=True)["target"].agg(
        events=("sum"),
        non_events=(lambda x: (1 - x).sum()),
        count="count",
    ).reset_index()

    stats["pct_total"] = stats["count"] / len(df)
    # Drop bins below minimum size
    stats = stats[stats["pct_total"] >= min_bin_pct].copy()

    eps = 1e-6
    stats["pct_events"]     = stats["events"]     / total_events
    stats["pct_non_events"] = stats["non_events"] / total_non_events
    stats["woe"] = np.log(
        (stats["pct_events"] + eps) / (stats["pct_non_events"] + eps)
    )
    stats["iv_bin"] = (stats["pct_events"] - stats["pct_non_events"]) * stats["woe"]

    iv_total = float(stats["iv_bin"].sum())
    logger.debug("IV for %s: %.4f", feature.name, iv_total)

    return stats[["bin", "woe", "iv_bin", "count", "pct_total"]], iv_total


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct derived features commonly used in consumer credit models.

    Input DataFrame should have raw bureau/application variables.
    Returns a copy with additional columns appended.
    """
    result = df.copy()

    # Payment burden: monthly debt / monthly income proxy
    if "dti_ratio" in df.columns and "loan_amount" in df.columns:
        result["payment_burden"] = (
            df["dti_ratio"] * df["loan_amount"] / (df["loan_amount"] + 1)
        ).round(6)

    # Credit depth: length of history × number of accounts (proxy)
    if "months_on_book" in df.columns:
        result["credit_depth_log"] = np.log1p(df["months_on_book"]).round(6)

    # Utilisation × delinquency interaction
    if "utilization_rate" in df.columns and "delinquency_count" in df.columns:
        result["util_delinq_interaction"] = (
            df["utilization_rate"] * np.log1p(df["delinquency_count"])
        ).round(6)

    # Risk premium proxy: FICO distance from prime threshold
    if "fico_score" in df.columns:
        result["subprime_indicator"] = (df["fico_score"] < 620).astype(int)
        result["fico_distance_from_prime"] = (680 - df["fico_score"]).clip(lower=0)

    # Income quality flag
    if "income_verified" in df.columns and "dti_ratio" in df.columns:
        result["high_risk_income"] = (
            (df["income_verified"] == 0) & (df["dti_ratio"] > 0.40)
        ).astype(int)

    return result


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute_credit_features(
    df: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    """
    Impute missing values in credit feature columns.

    strategy options: "median" | "mean" | "conservative"
    "conservative" uses the 75th percentile for risk features (DTI, utilization)
    to apply a cautious treatment of missing data.
    """
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if result[col].isna().any():
            n_missing = result[col].isna().sum()
            if strategy == "conservative" and col in ("dti_ratio", "utilization_rate", "delinquency_count"):
                fill_value = result[col].quantile(0.75)
            elif strategy == "mean":
                fill_value = result[col].mean()
            else:
                fill_value = result[col].median()

            result[col].fillna(fill_value, inplace=True)
            logger.info(
                "Imputed %d missing values in '%s' with %s=%.4f",
                n_missing, col, strategy, fill_value,
            )

    return result


# ---------------------------------------------------------------------------
# Feature selection by IV
# ---------------------------------------------------------------------------

def select_features_by_iv(
    df: pd.DataFrame,
    target: pd.Series,
    min_iv: float = 0.02,
    max_iv: float = 0.50,   # IV > 0.5 may indicate data leakage
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Select numeric features based on Information Value.

    Returns {feature: iv} for features with IV ∈ [min_iv, max_iv], sorted desc.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    iv_scores: dict[str, float] = {}

    for col in numeric_cols:
        try:
            _, iv = compute_woe_iv(df[col], target, n_bins=n_bins)
            if min_iv <= iv <= max_iv:
                iv_scores[col] = round(iv, 5)
        except Exception as exc:  # noqa: BLE001
            logger.debug("IV computation failed for %s: %s", col, exc)

    return dict(sorted(iv_scores.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Clipping / bounds enforcement
# ---------------------------------------------------------------------------

def clip_features(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """
    Hard-clip feature values to defined bounds.

    Used to handle out-of-range inputs in production (sensor errors,
    data pipeline issues) without crashing the scoring pipeline.
    """
    result = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in result.columns:
            n_clipped = ((result[col] < lo) | (result[col] > hi)).sum()
            if n_clipped > 0:
                logger.warning(
                    "Clipping %d out-of-range values in '%s' to [%s, %s].",
                    n_clipped, col, lo, hi,
                )
            result[col] = result[col].clip(lower=lo, upper=hi)
    return result
