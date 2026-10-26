import warnings

import numpy as np
import pandas as pd


_NUMERIC_STATS = ("mean", "median", "std", "min", "max", "q25", "q75", "skew", "kurtosis")
_GROUP_FUNCTIONS = ("mean", "sum", "count", "min", "max", "median")


def _column(df: pd.DataFrame, name) -> pd.Series:
    if name not in df.columns:
        raise ValueError(f"Column {name!r} is not present in the dataset.")
    column = df[name]
    if not isinstance(column, pd.Series):
        raise ValueError("Column names must be unique for analysis.")
    return column


def _real_numeric(series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype)
    )


def _output_name(preferred: str, occupied) -> str:
    name = preferred
    number = 2
    while name in occupied:
        name = f"{preferred} [{number}]"
        number += 1
    return name


def _number_or_none(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    return round(number, 4) if np.isfinite(number) else None


def _observed_counts(series: pd.Series) -> pd.Series:
    counts = series.value_counts()
    return counts[counts > 0]


def _category_labels(values):
    labels = [str(value) for value in values]
    duplicates = {label for label in labels if labels.count(label) > 1}
    return [
        f"{type(value).__name__}:{value!r}" if label in duplicates else label
        for value, label in zip(values, labels)
    ]


def univariate_stats(df: pd.DataFrame, col) -> dict:
    """Describe observed values; undefined numeric statistics are JSON nulls.

    Numeric summaries use finite observations only. count still counts all
    non-missing observations; finite_count/non_finite make exclusions explicit.
    """
    series = _column(df, col).dropna()
    if _real_numeric(series):
        finite = np.isfinite(series.to_numpy(dtype=float))
        observed = series.iloc[np.flatnonzero(finite)]
        result = {
            "type": "numeric",
            "count": len(series),
            "finite_count": len(observed),
            "non_finite": int((~finite).sum()),
            **{name: None for name in _NUMERIC_STATS},
        }
        if observed.empty:
            return result
        # Extreme but finite inputs can overflow higher moments. Those summaries
        # are unavailable rather than being exposed as invalid JSON infinities.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = {
                "mean": observed.mean(),
                "median": observed.median(),
                "std": observed.std() if len(observed) > 1 else None,
                "min": observed.min(),
                "max": observed.max(),
                "q25": observed.quantile(0.25),
                "q75": observed.quantile(0.75),
                "skew": observed.skew() if len(observed) > 2 else None,
                "kurtosis": observed.kurtosis() if len(observed) > 3 else None,
            }
        result.update({name: _number_or_none(value) for name, value in values.items()})
        return result

    counts = _observed_counts(series)
    top = counts.head(5)
    return {
        "type": "categorical",
        "count": int(series.count()),
        "unique": int(series.nunique()),
        "mode": str(counts.index[0]) if not counts.empty else None,
        "top_5": dict(zip(_category_labels(top.index), map(int, top))),
    }


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlate varying real numeric columns using finite pairs."""
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError("Choose pearson, spearman, or kendall correlation.")
    names = [name for name in df.columns if _real_numeric(_column(df, name))]
    numeric = df[names].replace([np.inf, -np.inf], np.nan)
    names = [name for name in names if numeric[name].nunique(dropna=True) > 1]
    if len(names) < 2:
        return pd.DataFrame()
    return numeric[names].corr(method=method).round(4)


def value_counts_summary(df: pd.DataFrame, col, top_n: int = 20) -> pd.DataFrame:
    """Return observed categorical counts with non-colliding column labels."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")
    counts = _observed_counts(_column(df, col)).head(top_n)
    label = _output_name("count", [col])
    return counts.rename(label).rename_axis(col).reset_index()


def group_by_summary(
    df: pd.DataFrame,
    group_col,
    agg_col,
    agg_func: str = "mean",
) -> pd.DataFrame:
    """Aggregate observed groups and name the output before resetting the index."""
    _column(df, group_col)
    values = _column(df, agg_col)
    if agg_func not in _GROUP_FUNCTIONS:
        raise ValueError(f"Unsupported aggregation {agg_func!r}; choose one of {_GROUP_FUNCTIONS}.")
    if agg_func != "count" and not _real_numeric(values):
        raise ValueError(
            f"Cannot apply {agg_func!r} to non-numeric column {agg_col!r}. "
            f"Column dtype is {values.dtype!r}; use count instead."
        )
    label = _output_name(f"{agg_func}({agg_col})", [group_col])
    grouped = df.groupby(group_col, observed=True)[agg_col]
    result = grouped.sum(min_count=1) if agg_func == "sum" else grouped.agg(agg_func)
    return result.rename(label).reset_index()
