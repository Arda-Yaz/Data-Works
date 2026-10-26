"""Validated transformation descriptions and deterministic replay.

Recipes contain data and a fixed set of operation names. They never contain
Python expressions, imports, callables, or executable serialized objects.
"""

import copy
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

from backend.cleaning import (
    build_missing_candidate,
    build_outlier_candidate,
    find_mismatches_by_allowed_values,
    remove_duplicates,
    validate_allowed_values,
)
from backend.ingestion import apply_inferred_dtypes
from backend.text_transform import (
    map_categories, normalize_text, transformed_metadata,
    validate_category_mapping, validate_normalization,
)


_MISSING = {"keep", "drop_rows", "mean", "median", "mode", "constant", "interpolate", "drop_column"}
_KINDS = {"schema", "missing", "duplicates", "outliers", "rules", "mismatches", "text_normalize", "category_map"}
_NAMES = {
    "schema": "Confirm schema", "missing": "Handle missing values",
    "duplicates": "Remove duplicates", "outliers": "Handle outliers",
    "rules": "Update allowed values", "mismatches": "Remove mismatch rows",
    "text_normalize": "Normalize text", "category_map": "Map categories",
}


def _safe_value(value, depth=0):
    if depth > 30:
        raise ValueError("Operation parameters are nested too deeply.")
    if value is None or value is pd.NA or value is pd.NaT:
        return
    if isinstance(value, (str, bool, int, float, complex, Decimal, Fraction,
                          date, datetime, time, timedelta, pd.Timestamp, pd.Timedelta,
                          np.datetime64, np.timedelta64)):
        return
    if isinstance(value, np.generic):
        _safe_value(value.item(), depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _safe_value(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _label(key)
            _safe_value(item, depth + 1)
        return
    raise ValueError(f"Unsupported operation parameter type: {type(value).__name__}.")


def _label(value, depth=0):
    if depth > 30:
        raise ValueError("Column labels are nested too deeply.")
    if isinstance(value, tuple):
        for item in value:
            _label(item, depth + 1)
    elif not (
        value is None or value is pd.NA or value is pd.NaT
        or isinstance(value, (str, bool, int, float, complex, Decimal, Fraction,
                             date, datetime, time, timedelta, pd.Timestamp, pd.Timedelta,
                             np.integer, np.floating, np.complexfloating, np.bool_,
                             np.datetime64, np.timedelta64))
    ):
        raise ValueError("Column references must be scalar labels or tuples of scalar labels.")
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError("Column references must be hashable labels.") from exc


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must map column names to values.")
    for key in value:
        _label(key)
    return value


def _parameters(operation, required, optional=()):
    parameters = operation.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("An operation needs a parameters dictionary.")
    unknown = set(parameters) - set(required) - set(optional)
    if unknown or not set(required).issubset(parameters):
        raise ValueError("The operation has missing or unsupported parameters.")
    _safe_value(parameters)
    return copy.deepcopy(parameters)


def _rules(values):
    values = _mapping(values, "Allowed values")
    for column, rules in values.items():
        if rules is not None and (
            not isinstance(rules, list) or any(not isinstance(rule, str) for rule in rules)
        ):
            raise ValueError(f"Column {column!r}: allowed values must be a list of strings or None.")
    return {column: rules or None for column, rules in values.items()}


def validate_operation(operation: dict) -> dict:
    """Return an owned, normalized descriptor, or raise before any work occurs.

    Honest custom/legacy audit records may be stored with replayable=False.
    They are never executable recipes and replay rejects them explicitly.
    Other audit fields such as id/name/timestamps are intentionally not executed.
    """
    if not isinstance(operation, dict):
        raise ValueError("An operation description must be a dictionary.")
    kind = operation.get("kind")
    if not isinstance(kind, str):
        raise ValueError("An operation kind must be a string.")
    if kind in ("custom", "legacy"):
        if operation.get("replayable") is not False:
            raise ValueError("Custom or legacy operations must be marked non-replayable.")
        parameters = operation.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("An operation needs a parameters dictionary.")
        _safe_value(parameters)
        parameters = copy.deepcopy(parameters)
        return {"kind": kind, "parameters": parameters, "replayable": False}
    if kind not in _KINDS:
        raise ValueError(f"Unsupported recipe operation: {kind!r}.")
    if operation.get("replayable", True) is not True:
        raise ValueError("Supported recipe operations must be marked replayable.")

    if kind == "schema":
        parameters = _parameters(operation, ("column_meta",), ("schema_confirmed", "apply_types"))
        metadata = _mapping(parameters["column_meta"], "Schema metadata")
        if any(not isinstance(value, dict) for value in metadata.values()):
            raise ValueError("Schema metadata must contain one metadata dictionary per column.")
        confirmed = parameters.get("schema_confirmed", True)
        if not isinstance(confirmed, bool):
            raise ValueError("Schema confirmation must be a boolean.")
        parameters["schema_confirmed"] = confirmed
        apply_types = parameters.get("apply_types", True)
        if not isinstance(apply_types, bool):
            raise ValueError("Schema type application must be a boolean.")
        parameters["apply_types"] = apply_types
    elif kind == "missing":
        parameters = _parameters(operation, ("strategies",), ("fill_values",))
        strategies = _mapping(parameters["strategies"], "Missing-value strategies")
        if any(not isinstance(value, str) or value not in _MISSING for value in strategies.values()):
            raise ValueError("Choose supported missing-value strategies.")
        fills = _mapping(parameters.get("fill_values", {}), "Fill values")
        for column, value in fills.items():
            if strategies.get(column) != "constant" or not pd.api.types.is_scalar(value):
                raise ValueError("Fill values must be scalars for columns using the constant strategy.")
        parameters["fill_values"] = fills
    elif kind == "duplicates":
        parameters = _parameters(operation, (), ("subset", "keep"))
        subset = parameters.get("subset")
        if subset is not None:
            if not isinstance(subset, list):
                raise ValueError("A duplicate subset must be a list of column labels or None.")
            for column in subset:
                _label(column)
            if len(set(subset)) != len(subset):
                raise ValueError("Duplicate subset column names must be unique.")
        keep = parameters.get("keep", "first")
        if keep is not False and keep not in ("first", "last"):
            raise ValueError("Duplicate retention must be first, last, or False.")
        parameters = {"subset": subset or None, "keep": keep}
    elif kind == "outliers":
        parameters = _parameters(operation, ("actions",), ("include_protected",))
        actions = _mapping(parameters["actions"], "Outlier actions")
        for column, choice in actions.items():
            if (
                not isinstance(choice, (list, tuple)) or len(choice) != 2
                or choice[0] not in ("iqr", "zscore")
                or choice[1] not in ("ignore", "cap", "replace_nan", "remove")
            ):
                raise ValueError(f"Column {column!r}: choose a supported outlier method and action.")
        protected = parameters.get("include_protected", False)
        if not isinstance(protected, bool):
            raise ValueError("Protected-column inclusion must be a boolean.")
        parameters = {"actions": {key: tuple(value) for key, value in actions.items()},
                      "include_protected": protected}
    elif kind == "text_normalize":
        parameters = _parameters(operation, ("columns",), ("strip", "collapse_whitespace", "case", "unicode_form"))
        columns = parameters["columns"]
        if not isinstance(columns, list) or not columns:
            raise ValueError("Choose at least one text column.")
        for column in columns:
            _label(column)
        if len(set(columns)) != len(columns):
            raise ValueError("Choose each text column once.")
        parameters = {"columns": columns, **validate_normalization(**{
            key: value for key, value in parameters.items() if key != "columns"
        })}
    elif kind == "category_map":
        parameters = _parameters(operation, ("column", "mapping"))
        _label(parameters["column"])
        parameters["mapping"] = validate_category_mapping(parameters["mapping"])
    else:
        parameters = _parameters(operation, ("allowed_values",))
        parameters["allowed_values"] = _rules(parameters["allowed_values"])
    return {"kind": kind, "parameters": parameters, "replayable": True}


def _references(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Recipe references absent columns: {missing}.")


def _updated_rules(df, metadata, values):
    _references(df, values)
    result = copy.deepcopy(metadata)
    for column, rules in values.items():
        if rules:
            validate_allowed_values(df[column], rules)
        result.setdefault(column, {})["allowed_values"] = copy.deepcopy(rules)
    return result


def apply_recipe_operation(df, metadata, operation):
    """Calculate an operation without mutating the frame, schema, or session.

    Returns (candidate_df, explicit_metadata_or_None, confirmation_or_None,
    learns_from_data). StateManager remains responsible for state commitment.
    """
    descriptor = validate_operation(operation)
    if not descriptor["replayable"]:
        raise ValueError("This operation has no reproducible parameters and cannot be replayed.")
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError("Recipe replay requires a DataFrame with unique column names.")
    if not isinstance(metadata, dict) or any(not isinstance(value, dict) for value in metadata.values()):
        raise ValueError("Recipe replay requires valid column metadata.")
    kind, parameters = descriptor["kind"], descriptor["parameters"]
    if kind == "schema":
        schema = parameters["column_meta"]
        _references(df, schema)
        if any(column not in schema for column in df.columns):
            raise ValueError("A schema recipe must describe every current column.")
        candidate = apply_inferred_dtypes(df, schema) if parameters["apply_types"] else df.copy(deep=True)
        return candidate, schema, parameters["schema_confirmed"], False
    if kind == "missing":
        candidate = build_missing_candidate(df, parameters["strategies"], parameters["fill_values"])
        learns = any(value in ("mean", "median", "mode", "interpolate") for value in parameters["strategies"].values())
        return candidate, None, None, learns
    if kind == "duplicates":
        if parameters["subset"] is not None:
            _references(df, parameters["subset"])
        candidate, _ = remove_duplicates(df, subset=parameters["subset"], keep=parameters["keep"])
        return candidate, None, None, False
    if kind == "outliers":
        candidate = build_outlier_candidate(df, parameters["actions"])
        learns = any(choice[1] != "ignore" for choice in parameters["actions"].values())
        return candidate, None, None, learns
    if kind == "text_normalize":
        columns = parameters["columns"]
        options = {key: value for key, value in parameters.items() if key != "columns"}
        candidate = normalize_text(df, columns, **options)
        candidate_metadata = transformed_metadata(metadata, columns, normalization=options)
        return candidate, candidate_metadata, None, False
    if kind == "category_map":
        column, mapping = parameters["column"], parameters["mapping"]
        candidate = map_categories(df, column, mapping)
        candidate_metadata = transformed_metadata(metadata, [column], mapping=mapping)
        return candidate, candidate_metadata, None, False
    rules = parameters["allowed_values"]
    candidate_metadata = _updated_rules(df, metadata, rules)
    if kind == "rules":
        return df.copy(deep=True), candidate_metadata, None, False
    positional = df.reset_index(drop=True)
    mismatches = find_mismatches_by_allowed_values(positional, rules)
    bad = {position for positions in mismatches.values() for position in positions}
    candidate = positional.drop(index=list(bad))
    candidate.index = df.index.take(candidate.index.to_numpy(dtype=int))
    return candidate, candidate_metadata, None, False


def replay_operations(original_df, original_metadata, operation_log):
    """Replay a complete known recipe using the application's schema rules.

    Unknown legacy/custom records fail rather than being silently skipped.
    Audit names and other extra fields are never interpreted as code.
    """
    from backend.state import _prepare_dataset

    if not isinstance(operation_log, list):
        raise ValueError("A transformation history must be a list of operation records.")
    current, metadata = _prepare_dataset(original_df, original_metadata, explicit_schema=True)
    confirmed = False
    statistical = []
    for number, record in enumerate(operation_log, start=1):
        try:
            descriptor = validate_operation(record)
            name = record.get("name") or _NAMES.get(descriptor["kind"], "Unknown operation")
            if not isinstance(name, str):
                raise ValueError("Operation names must be strings.")
            candidate, explicit, new_confirmed, learns = apply_recipe_operation(current, metadata, descriptor)
            prepared, new_metadata = _prepare_dataset(
                candidate, metadata if explicit is None else explicit,
                previous_df=current, explicit_schema=explicit is not None,
            )
            next_confirmed = confirmed if new_confirmed is None else new_confirmed
            changed = not prepared.equals(current) or new_metadata != metadata or next_confirmed != confirmed
            if changed and learns and name not in statistical:
                statistical.append(name)
            current, metadata, confirmed = prepared, new_metadata, next_confirmed
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise ValueError(f"Cannot replay operation {number}: {exc}") from exc
    return {
        "df": current, "column_meta": metadata, "schema_confirmed": confirmed,
        "statistical_operations": statistical,
    }


def export_recipe(original_df, original_metadata, operation_log) -> bytes:
    """Export exact typed parameters and a digest identifying the source table."""
    import hashlib
    import json
    from backend.project_io import encode_value

    if not isinstance(operation_log, list):
        raise ValueError("Transformation history must be a list.")
    for operation in operation_log:
        validate_operation(operation)
    source = json.dumps(encode_value(original_df), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    document = {
        "format": "dataworks-transformation-recipe", "version": 1,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "original_column_meta": encode_value(original_metadata),
        "operations": encode_value(operation_log),
        "fully_replayable": all(item.get("replayable", False) for item in operation_log),
    }
    return json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
