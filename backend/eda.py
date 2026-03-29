import pandas as pd
import numpy as np


def univariate_stats(df: pd.DataFrame, col: str) -> dict:
    """Return descriptive statistics for a single column."""
    series = df[col].dropna()
    if pd.api.types.is_numeric_dtype(series):
        return {
            "type": "numeric",
            "count": int(series.count()),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q25": round(float(series.quantile(0.25)), 4),
            "q75": round(float(series.quantile(0.75)), 4),
            "skew": round(float(series.skew(skipna=True)), 4),  # type: ignore[arg-type]
            "kurtosis": round(float(series.kurtosis(skipna=True)), 4),  # type: ignore[arg-type]
        }
    # Categorical / object
    vc = series.value_counts()
    return {
        "type": "categorical",
        "count": int(series.count()),
        "unique": int(series.nunique()),
        "mode": str(vc.index[0]) if not vc.empty else None,
        "top_5": vc.head(5).to_dict(),
    }


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Return the correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(method=method).round(4)  # type: ignore[arg-type]


def value_counts_summary(df: pd.DataFrame, col: str, top_n: int = 20) -> pd.DataFrame:
    vc = df[col].value_counts().head(top_n).reset_index()
    vc.columns = [col, "count"]
    return vc


def group_by_summary(
    df: pd.DataFrame,
    group_col: str,
    agg_col: str,
    agg_func: str = "mean",
) -> pd.DataFrame:
    """Group by one column and aggregate another."""
    result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    result.columns = [group_col, f"{agg_func}({agg_col})"]
    return result
