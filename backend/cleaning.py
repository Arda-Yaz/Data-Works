import streamlit as st
import pandas as pd
import numpy as np


# ── Missing-value helpers ────────────────────────────────────────────────────

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing count, percentage, and suggested strategy per column."""
    missing = df.isna().sum().to_frame("count")
    missing["%"] = (100 * missing["count"] / len(df)).round(2)
    missing = missing[missing["count"] > 0]

    suggestions = []
    for col in missing.index:
        pct: float = missing.loc[col, "%"]  # type: ignore[assignment]
        if pct < 5:
            suggestions.append("drop_rows")
        elif pct < 20:
            if df[col].dtype in ("int64", "float64", "Int64", "Float64"):
                suggestions.append("median")
            else:
                suggestions.append("mode")
        else:
            suggestions.append("drop_column")
    missing["suggested"] = suggestions
    return missing


def handle_missing(
    df: pd.DataFrame,
    col: str,
    strategy: str,
    fill_value=None,
) -> pd.DataFrame:
    """Apply a missing-value strategy to a single column.

    Strategies: drop_rows, mean, median, mode, constant, interpolate, drop_column
    """
    df = df.copy()
    if strategy == "drop_rows":
        df = df.dropna(subset=[col])
    elif strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])
    elif strategy == "constant":
        df[col] = df[col].fillna(fill_value)
    elif strategy == "interpolate":
        df[col] = df[col].interpolate()
    elif strategy == "drop_column":
        df = df.drop(columns=[col])
    return df


# ── Duplicate helpers ────────────────────────────────────────────────────────

def get_duplicate_summary(
    df: pd.DataFrame, subset: list[str] | None = None
) -> tuple[int, pd.DataFrame]:
    dupes = df.duplicated(subset=subset, keep=False)
    n_dupes = dupes.sum()
    sample = df[dupes].head(20)
    return int(n_dupes), sample


def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str | bool = "first",
) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)  # type: ignore[arg-type]
    return df, before - len(df)


# ── Outlier helpers ──────────────────────────────────────────────────────────

def find_outliers(df):
    """Find outliers using the IQR method. Returns dict of col → list of indices."""
    outliers = {}
    for col in df.select_dtypes(include="number").columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
    return outliers


def handle_outliers(
    df: pd.DataFrame,
    col: str,
    method: str = "iqr",
    action: str = "remove",
) -> pd.DataFrame:
    """Handle outliers for a single column.

    method: iqr | zscore
    action: remove | cap | replace_nan
    """
    df = df.copy()
    series = df[col]

    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    else:  # zscore
        mean, std = series.mean(), series.std()
        lower, upper = mean - 3 * std, mean + 3 * std

    mask = (series < lower) | (series > upper)

    if action == "remove":
        df = df[~mask]
    elif action == "cap":
        df.loc[series < lower, col] = lower
        df.loc[series > upper, col] = upper
    elif action == "replace_nan":
        df.loc[mask, col] = np.nan

    return df


# ── Mismatch helpers (existing, kept) ────────────────────────────────────────

def find_mismatches_by_allowed_values(df, allowed_values_dict):
    mismatches = {}
    for col, allowed_values in allowed_values_dict.items():
        col_mismatches = []
        if not allowed_values:
            continue
        allowed_set = set(allowed_values)
        for idx, value in df[col].items():
            if pd.isna(value):
                continue
            if str(value) not in allowed_set:
                col_mismatches.append(idx)
        mismatches[col] = col_mismatches
    return mismatches


# ── Legacy automatic cleaning (kept for backward compat) ─────────────────────

def automatic_cleaning(df, missing):
    for col in missing.index:
        if missing.loc[col, "%"] < 5:
            df = df.dropna(subset=[col])
        elif missing.loc[col, "%"] < 20:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df = df.drop(columns=[col])
    return df