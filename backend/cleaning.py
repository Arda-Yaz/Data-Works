from decimal import Decimal, InvalidOperation
from fractions import Fraction
from numbers import Real

import pandas as pd
import numpy as np


# ── Missing-value helpers ────────────────────────────────────────────────────

def _is_real_numeric(series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype)
    )


def get_missing_strategies(series: pd.Series) -> list[str]:
    """Return only missing-value strategies compatible with the column's dtype."""
    strategies = ["drop_rows"]
    if _is_real_numeric(series):
        strategies.extend(["mean", "median"])
    strategies.extend(["mode", "constant"])
    if "mean" in strategies:
        strategies.append("interpolate")
    strategies.append("drop_column")
    return strategies


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
            if "median" in get_missing_strategies(df[col]):
                suggestions.append("median")
            else:
                suggestions.append("mode")
        else:
            suggestions.append("drop_column")
    missing["suggested"] = suggestions
    return missing


def _parse_number(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Enter a number, not a boolean.")
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError("Enter a finite numeric fill value, for example 0 or 1.5.") from exc
    if not number.is_finite():
        raise ValueError("The fill value must be finite.")
    if number == number.to_integral_value():
        if -(2**63) <= number <= 2**64 - 1:
            return int(number)
        result = float(number)
        if not np.isfinite(result) or Decimal.from_float(result) != number:
            raise ValueError("The whole-number fill value cannot be represented exactly.")
        return result
    result = float(number)
    if not np.isfinite(result) or result == 0 or result.is_integer():
        raise ValueError("The fill value cannot be represented without losing its numeric meaning.")
    return result


def _parse_constant(series: pd.Series, value):
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        raise ValueError("Supply a non-missing scalar fill value.")
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = pd.Series(series.cat.categories)
        if categories.dtype == object and not categories.empty and all(
            isinstance(item, Real) and not isinstance(item, (bool, np.bool_)) for item in categories
        ):
            return _parse_number(value)
        parsed = _parse_constant(categories, value)
        if pd.api.types.is_float_dtype(categories.dtype):
            _, parsed = _numeric_destination(categories, parsed)
        return parsed
    if pd.api.types.is_bool_dtype(series.dtype):
        tokens = {"true": True, "false": False, "yes": True, "no": False,
                  "t": True, "f": False, "1": True, "0": False}
        token = str(value).strip().lower()
        if token not in tokens:
            raise ValueError("Use true/false, yes/no, t/f, or 1/0 for a boolean fill.")
        return tokens[token]
    if _is_real_numeric(series):
        return _parse_number(value)
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        timestamp = pd.to_datetime(value, format="ISO8601", errors="raise")
        if pd.isna(timestamp):
            raise ValueError("Enter a non-missing ISO date or datetime.")
        timezone = series.dt.tz
        if timezone is not None:
            try:
                return timestamp.tz_localize(timezone) if timestamp.tzinfo is None else timestamp.tz_convert(timezone)
            except Exception as exc:
                raise ValueError(
                    "This local datetime is ambiguous or invalid for the column's timezone. "
                    "Use an ISO datetime with an explicit UTC offset."
                ) from exc
        if timestamp.tzinfo is not None:
            raise ValueError("This column has no timezone; use a datetime without a timezone offset.")
        return timestamp
    if pd.api.types.is_timedelta64_dtype(series.dtype):
        duration = pd.to_timedelta(value, errors="raise")
        if pd.isna(duration):
            raise ValueError("Enter a non-missing duration, for example '2 days'.")
        return duration
    if pd.api.types.is_complex_dtype(series.dtype):
        number = complex(value)
        if not np.isfinite(number):
            raise ValueError("The fill value must be finite.")
        return number
    return value


def _float_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series.dtype):
        if any(int(value) != int(float(value)) for value in series.dropna()):
            raise ValueError(
                "A fractional result would round existing large integers. "
                "Use a whole-number constant, mode, or a row/column removal strategy."
            )
    dtype = "Float64" if pd.api.types.is_extension_array_dtype(series.dtype) else "float64"
    return series.astype(dtype)


def _numeric_destination(series: pd.Series, value) -> tuple[pd.Series, int | float]:
    """Choose a safe destination dtype before assigning a numeric result."""
    if not np.isfinite(float(value)):
        raise ValueError("The numeric result is not finite; choose a different fill or cleaning strategy.")
    if pd.api.types.is_integer_dtype(series.dtype):
        if isinstance(value, Fraction):
            if value.denominator == 1:
                value = value.numerator
            else:
                value = float(value)
                if value.is_integer():
                    raise ValueError("The fractional result would be rounded to an integer.")
        if not float(value).is_integer():
            return _float_series(series), value
        value = int(value)
        limits = np.iinfo(getattr(series.dtype, "numpy_dtype", series.dtype))
        if not limits.min <= value <= limits.max:
            observed = series.dropna()
            low = min(value, int(observed.min())) if not observed.empty else value
            high = max(value, int(observed.max())) if not observed.empty else value
            if -(2**63) <= low and high < 2**63:
                dtype = "Int64"
            elif 0 <= low and high < 2**64:
                dtype = "UInt64"
            else:
                raise ValueError("The fill value and existing integers cannot share a safe 64-bit dtype.")
            return series.astype(dtype), value
    elif isinstance(value, (int, np.integer)) and int(float(value)) != value:
        raise ValueError("This whole-number fill value would lose precision in a floating-point column.")
    if pd.api.types.is_float_dtype(series.dtype):
        storage = getattr(series.dtype, "numpy_dtype", series.dtype)
        with np.errstate(over="ignore", under="ignore"):
            stored_value = np.asarray(value, dtype=storage).item()
        if stored_value != value:
            return _float_series(series), value
    return series.copy(), value


def _statistic(series: pd.Series, strategy: str):
    if not pd.api.types.is_integer_dtype(series.dtype):
        numeric = _float_series(series)
        return numeric.mean() if strategy == "mean" else numeric.median()
    present = series.dropna()
    if strategy == "mean":
        value = Fraction(sum(map(int, present)), len(present))
    else:
        ordered = present.sort_values()
        midpoint = len(ordered) // 2
        value = Fraction(int(ordered.iloc[midpoint])) if len(ordered) % 2 else Fraction(
            int(ordered.iloc[midpoint - 1]) + int(ordered.iloc[midpoint]), 2,
        )
    if value.denominator == 1:
        return value.numerator
    result = float(value)
    if result.is_integer():
        raise ValueError("The fractional statistic is too large to represent without rounding to an integer.")
    return result


def _interpolate_integers(series: pd.Series) -> pd.Series:
    # Compute missing positions exactly before deciding whether a float dtype is needed.
    present = np.flatnonzero(series.notna().to_numpy())
    replacements = []
    for left, right in zip(present[:-1], present[1:]):
        if right == left + 1:
            continue
        start, stop = int(series.iloc[left]), int(series.iloc[right])
        for position in range(left + 1, right):
            replacements.append((position, Fraction(start) + Fraction(stop - start, int(right - left)) * int(position - left)))
    replacements.extend((position, Fraction(int(series.iloc[present[-1]]))) for position in range(present[-1] + 1, len(series)))
    fractional = any(value.denominator != 1 for _, value in replacements)
    result = _float_series(series) if fractional else series.copy()
    for position, value in replacements:
        number = float(value) if fractional else value.numerator
        if fractional and value.denominator == 1 and int(number) != value.numerator:
            raise ValueError("Interpolation would round a newly calculated whole number during float conversion.")
        if value.denominator != 1 and number.is_integer():
            raise ValueError("Interpolation would round a fractional result to an integer.")
        result.iloc[position] = number
    return result


def _interpolate_floats(series: pd.Series) -> pd.Series:
    result = _float_series(series).interpolate()
    storage = getattr(series.dtype, "numpy_dtype", series.dtype)
    if np.dtype(storage).itemsize < 8:
        filled = result.loc[series.isna()].dropna().to_numpy(dtype=float)
        with np.errstate(over="ignore", under="ignore"):
            narrowed = filled.astype(storage).astype(float)
        if np.array_equal(filled, narrowed):
            return result.astype(series.dtype)
    return result


def handle_missing(
    df: pd.DataFrame,
    col: str,
    strategy: str,
    fill_value=None,
) -> pd.DataFrame:
    """Apply a missing-value strategy to a single column.

    Strategies: drop_rows, mean, median, mode, constant, interpolate, drop_column
    """
    if col not in df.columns:
        raise ValueError(f"Column {col!r} is not present in the dataset.")
    series = df[col]
    if strategy not in get_missing_strategies(series):
        raise ValueError(f"Column {col!r}: strategy {strategy!r} is incompatible with dtype {series.dtype}.")
    result = df.copy()
    if strategy == "drop_rows":
        return result.dropna(subset=[col])
    if strategy == "drop_column":
        return result.drop(columns=[col])
    if not series.isna().any():
        return result
    try:
        if strategy != "constant" and series.dropna().empty:
            raise ValueError("There are no observed values to use. Choose a constant or drop rows/the column.")
        if strategy in ("mean", "median", "interpolate"):
            if not np.isfinite(series.dropna().to_numpy(dtype=float)).all():
                raise ValueError("Replace non-finite values before computing statistics or interpolating.")
        if strategy == "interpolate":
            result[col] = (
                _interpolate_integers(series) if pd.api.types.is_integer_dtype(series.dtype)
                else _interpolate_floats(series)
            )
            return result
        if strategy in ("mean", "median"):
            value = _statistic(series, strategy)
        elif strategy == "mode":
            value = series.mode().iloc[0]
        else:
            value = _parse_constant(series, fill_value)
        destination = series.copy()
        if _is_real_numeric(series):
            destination, value = _numeric_destination(series, value)
        elif isinstance(series.dtype, pd.CategoricalDtype) and value not in series.cat.categories:
            categories = series.cat.categories
            mixed_numeric = pd.api.types.is_integer_dtype(categories.dtype) and isinstance(value, float)
            if categories.dtype == object or mixed_numeric:
                # add_categories may infer a float index and round integer labels.
                categorical = pd.Categorical.from_codes(
                    series.cat.codes.to_numpy(copy=True),
                    categories=pd.Index([*categories, value], dtype=object),
                    ordered=series.cat.ordered,
                )
                destination = pd.Series(categorical, index=series.index, name=series.name)
            else:
                destination = series.cat.add_categories([value])
        destination.loc[destination.isna()] = value
        result[col] = destination
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Column {col!r}: {exc}") from exc


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

def _integer_quantile(ordered: pd.Series, quantile: Fraction) -> Fraction:
    position = (len(ordered) - 1) * quantile
    left = position.numerator // position.denominator
    value = Fraction(int(ordered.iloc[left]))
    if position.denominator != 1:
        value += (int(ordered.iloc[left + 1]) - int(ordered.iloc[left])) * (position - left)
    return value


def _bound_masks(series: pd.Series, lower, upper):
    if pd.api.types.is_integer_dtype(series.dtype):
        # Integer comparisons against rounded float bounds also lose precision.
        lower, upper = Fraction(lower), Fraction(upper)
        below = -(-lower.numerator // lower.denominator)
        above = upper.numerator // upper.denominator
        limits = np.iinfo(getattr(series.dtype, "numpy_dtype", series.dtype))
        false = pd.Series(False, index=series.index)
        low_mask = false if below <= limits.min else series.notna() if below > limits.max else series < below
        high_mask = false if above >= limits.max else series.notna() if above < limits.min else series > above
        return low_mask.fillna(False), high_mask.fillna(False)
    return (series < lower).fillna(False), (series > upper).fillna(False)


def _outlier_mask(series: pd.Series, method: str):
    if method not in ("iqr", "zscore"):
        raise ValueError(f"Column {series.name!r}: unknown outlier method {method!r}.")
    if not _is_real_numeric(series):
        raise ValueError(f"Column {series.name!r}: outlier handling requires a real numeric column.")
    observed = series.dropna()
    empty_mask = pd.Series(False, index=series.index)
    if observed.empty:
        return empty_mask, None, None
    if not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ValueError(f"Column {series.name!r}: replace non-finite values before detecting outliers.")
    integer = pd.api.types.is_integer_dtype(series.dtype)
    if method == "iqr":
        if integer:
            ordered = observed.sort_values()
            q1, q3 = _integer_quantile(ordered, Fraction(1, 4)), _integer_quantile(ordered, Fraction(3, 4))
        else:
            numeric = _float_series(observed)
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        spread = q3 - q1
        multiplier = Fraction(3, 2) if integer else 1.5
        lower, upper = q1 - multiplier * spread, q3 + multiplier * spread
    else:
        if len(observed) < 2:
            return empty_mask, None, None
        if integer:
            mean = Fraction(sum(map(int, observed)), len(observed))
            origin = int(observed.iloc[0])
            std = pd.Series([int(value) - origin for value in observed], dtype=float).std()
            distance = Fraction.from_float(3 * float(std))
        else:
            numeric = _float_series(observed)
            mean, std = numeric.mean(), numeric.std()
            distance = 3 * std
        lower, upper = mean - distance, mean + distance
    if not np.isfinite(float(lower)) or not np.isfinite(float(upper)):
        raise ValueError(f"Column {series.name!r}: outlier bounds are not finite; rescale the values first.")
    low_mask, high_mask = _bound_masks(series, lower, upper)
    return low_mask | high_mask, lower, upper


def find_outliers(df):
    """Find outliers using the IQR method. Returns dict of col → list of indices."""
    outliers = {}
    for col in df.select_dtypes(include="number").columns:
        if _is_real_numeric(df[col]):
            mask, _, _ = _outlier_mask(df[col], "iqr")
            outliers[col] = df.index[mask].tolist()
    return outliers


def handle_outliers(
    df: pd.DataFrame,
    col: str,
    method: str = "iqr",
    action: str = "remove",
) -> pd.DataFrame:
    """Handle outliers for a single column.

    method: iqr | zscore
    action: remove | cap | replace_nan | ignore
    """
    if col not in df.columns:
        raise ValueError(f"Column {col!r} is not present in the dataset.")
    if action not in ("remove", "cap", "replace_nan", "ignore"):
        raise ValueError(f"Column {col!r}: unknown outlier action {action!r}.")
    series = df[col]
    mask, lower, upper = _outlier_mask(series, method)
    result = df.copy()
    if action == "ignore" or not mask.any():
        return result
    if action == "remove":
        return result.loc[~mask].copy()
    try:
        if action == "cap":
            capped = series.copy()
            low_mask, high_mask = _bound_masks(series, lower, upper)
            for bound, selected in (
                (lower, low_mask),
                (upper, high_mask),
            ):
                if selected.any():
                    capped, value = _numeric_destination(capped, bound)
                    capped.loc[selected] = value
            result[col] = capped
        elif action == "replace_nan":
            destination = series.convert_dtypes() if pd.api.types.is_integer_dtype(series.dtype) else series.copy()
            destination.loc[mask] = pd.NA
            result[col] = destination
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Column {col!r}: {exc}") from exc


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
    result = df.copy()
    for col in missing.index:
        if missing.loc[col, "%"] < 5:
            strategy = "drop_rows"
        elif missing.loc[col, "%"] < 20:
            strategy = "median" if "median" in get_missing_strategies(result[col]) else "mode"
        else:
            strategy = "drop_column"
        result = handle_missing(result, col, strategy)
    return result
