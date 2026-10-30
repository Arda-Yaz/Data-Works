"""Owned dataset copies and schema reconciliation without UI or session state."""

import copy

import pandas as pd

from backend import resources
from backend.cleaning import validate_allowed_values
from backend.ingestion import infer_column_meta, validate_numeric_format


def empty_column_meta() -> dict:
    return {
        "original_dtype": "object",
        "inferred_dtype": "object",
        "current_dtype": "object",
        "confidence": 0.0,
        "user_dtype": None,
        "role": "feature",
        "cardinality": 0,
        "is_categorical": False,
        "allowed_values": None,
        "date_format": None,
        "numeric_decimal": ".",
        "numeric_thousands": None,
    }


def _schema_dtype(series: pd.Series) -> str:
    """Map nullable storage types to the schema editor's equivalent choices."""
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        if not pd.api.types.is_unsigned_integer_dtype(series):
            return "int64"
        if series.dropna().empty or int(series.max()) <= 2**63 - 1:
            return "int64"
    if pd.api.types.is_float_dtype(series):
        return "float64"
    if pd.api.types.is_string_dtype(series.dtype):
        return "object"
    return str(series.dtype)


def prepare_dataset(df, metadata, previous_df=None, explicit_schema=False):
    """Prepare owned copies and synchronized metadata before any state mutation."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The dataset must be a pandas DataFrame.")
    resources.validate_dataset(df)
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique to maintain an unambiguous schema.")
    if not isinstance(metadata, dict) or any(not isinstance(value, dict) for value in metadata.values()):
        raise ValueError("Column metadata must map column names to metadata dictionaries.")
    candidate = df.copy(deep=True)
    missing = [name for name in df.columns if name not in metadata]
    inferred = infer_column_meta(df[missing]) if missing else {}
    synchronized = {}
    for name in candidate.columns:
        series = candidate[name]
        supplied = metadata.get(name, inferred.get(name, {}))
        meta = empty_column_meta()
        meta.update(copy.deepcopy(supplied))
        if meta["role"] not in ("feature", "id", "datetime", "ignore"):
            raise ValueError(f"Column {name!r}: choose a supported column role.")
        if not isinstance(meta["inferred_dtype"], str) or not meta["inferred_dtype"]:
            raise ValueError(f"Column {name!r}: the inferred type must be a non-empty string.")
        if meta["user_dtype"] is not None and (
            not isinstance(meta["user_dtype"], str) or not meta["user_dtype"]
        ):
            raise ValueError(f"Column {name!r}: the requested type must be a non-empty string.")
        allowed = meta["allowed_values"]
        if allowed is not None and (
            not isinstance(allowed, list) or any(not isinstance(value, str) for value in allowed)
        ):
            raise ValueError(f"Column {name!r}: allowed values must be a list of strings.")
        if meta["date_format"] is not None and not isinstance(meta["date_format"], str):
            raise ValueError(f"Column {name!r}: the date format must be a string.")
        # Validate project/schema settings before accepting any dataset mutation.
        validate_numeric_format(meta["numeric_decimal"], meta["numeric_thousands"])
        if allowed:
            validate_allowed_values(series, allowed)
        if "original_dtype" not in supplied:
            meta["original_dtype"] = str(series.dtype)
        if (
            not explicit_schema and previous_df is not None
            and name in previous_df.columns and previous_df[name].dtype != series.dtype
        ):
            meta["user_dtype"] = None
            meta["inferred_dtype"] = _schema_dtype(series)
        # Schema choices are logical conversion targets. Preparation preserves
        # existing storage (including StringDtype's missing-value semantics).
        meta["current_dtype"] = str(series.dtype)
        meta["cardinality"] = int(series.nunique())
        meta["is_categorical"] = (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or meta["cardinality"] / max(len(candidate), 1) < 0.05
            or (meta["cardinality"] < 30 and not pd.api.types.is_datetime64_any_dtype(series.dtype))
        )
        synchronized[name] = meta
    return candidate, synchronized
