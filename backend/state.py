import copy
from typing import Any

import pandas as pd
import streamlit as st

from backend.ingestion import infer_column_meta
from backend.cleaning import validate_allowed_values


MAX_HISTORY = 10
_WIDGET_PREFIXES = ("ms_", "fv_", "om_", "oa_", "av_", "ml_", "vz_", "eda_", "corr_", "gb_")
_WIDGET_KEYS = {
    "schema_editor", "data_upload_file", "preview_rows", "dup_subset", "dup_keep",
    "apply_missing", "apply_dupes", "apply_outliers", "find_mismatch",
    "apply_mismatch", "apply_allowed_values",
}


def _empty_column_meta() -> dict:
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
    }


def _defaults() -> dict:
    return {
        "original_df": None, "df": None, "column_meta": {}, "history": [],
        "target_column": None, "schema_confirmed": False, "viz_charts": [],
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


def _prepare_dataset(df, metadata, previous_df=None, explicit_schema=False):
    """Prepare owned copies and synchronized metadata before any state mutation."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The dataset must be a pandas DataFrame.")
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
        meta = _empty_column_meta()
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


class StateManager:
    """Own dataset mutations, synchronized schema, undo, and dependent state."""

    @staticmethod
    def init_session() -> None:
        # A widget cannot be reset after it has been instantiated in the same run.
        # Operations schedule only existing widget keys for the next rerun.
        pending = st.session_state.get("_dataset_widgets_pending_reset", [])
        for key in pending:
            st.session_state.pop(key, None)
        st.session_state["_dataset_widgets_pending_reset"] = []
        for key, value in _defaults().items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def _invalidate_dependent_state() -> None:
        pending = set(st.session_state.get("_dataset_widgets_pending_reset", []))
        pending.update(
            key for key in st.session_state
            if isinstance(key, str) and (key in _WIDGET_KEYS or key.startswith(_WIDGET_PREFIXES))
        )
        st.session_state["_dataset_widgets_pending_reset"] = sorted(pending)
        st.session_state.viz_charts = []

    @staticmethod
    def load_dataset(df: pd.DataFrame, column_meta: dict) -> None:
        candidate, metadata = _prepare_dataset(df, column_meta, explicit_schema=True)
        original = candidate.copy(deep=True)
        StateManager._invalidate_dependent_state()
        st.session_state.original_df = original
        st.session_state.df = candidate
        st.session_state.column_meta = metadata
        st.session_state.history = []
        st.session_state.schema_confirmed = False
        st.session_state.target_column = None

    @staticmethod
    def _snapshot(operation_name: str) -> dict:
        if not StateManager.has_dataset():
            raise ValueError("Load a dataset before recording an operation.")
        return {
            "name": operation_name,
            "df": st.session_state.df.copy(deep=True),
            "column_meta": copy.deepcopy(st.session_state.column_meta),
            "schema_confirmed": st.session_state.schema_confirmed,
            "target_column": st.session_state.target_column,
        }

    @staticmethod
    def apply_operation(
        operation_name: str,
        df: pd.DataFrame,
        *,
        column_meta: dict | None = None,
        schema_confirmed: bool | None = None,
    ) -> bool:
        """Atomically apply a prepared operation; return False for a true no-op.

        The caller calculates transformations first. Validation and copying finish
        before history, the current dataset, or dependent outputs are changed.
        """
        if not StateManager.has_dataset():
            raise ValueError("Load a dataset before applying an operation.")
        if schema_confirmed is not None and not isinstance(schema_confirmed, bool):
            raise ValueError("Schema confirmation must be a boolean.")
        metadata = st.session_state.column_meta if column_meta is None else column_meta
        candidate, metadata = _prepare_dataset(
            df, metadata, previous_df=st.session_state.df,
            explicit_schema=column_meta is not None,
        )
        confirmed = st.session_state.schema_confirmed if schema_confirmed is None else schema_confirmed
        target = st.session_state.target_column
        if target not in candidate.columns:
            target = None
        if (
            candidate.equals(st.session_state.df)
            and metadata == st.session_state.column_meta
            and confirmed == st.session_state.schema_confirmed
            and target == st.session_state.target_column
        ):
            return False
        snapshot = StateManager._snapshot(operation_name)
        history = [*st.session_state.history, snapshot][-MAX_HISTORY:]
        StateManager._invalidate_dependent_state()
        st.session_state.df = candidate
        st.session_state.column_meta = metadata
        st.session_state.schema_confirmed = confirmed
        st.session_state.target_column = target
        st.session_state.history = history
        return True

    @staticmethod
    def save_snapshot(operation_name: str) -> None:
        """Retain explicit checkpoints; normal mutations use apply_operation."""
        snapshot = StateManager._snapshot(operation_name)
        st.session_state.history = [*st.session_state.history, snapshot][-MAX_HISTORY:]

    @staticmethod
    def undo() -> bool:
        history = st.session_state.history
        if not history:
            return False
        snapshot = history[-1]
        candidate, metadata = _prepare_dataset(snapshot["df"], snapshot["column_meta"], explicit_schema=True)
        target = snapshot.get("target_column")
        if target not in candidate.columns:
            target = None
        confirmed = snapshot.get("schema_confirmed", False)
        StateManager._invalidate_dependent_state()
        st.session_state.df = candidate
        st.session_state.column_meta = metadata
        st.session_state.schema_confirmed = confirmed
        st.session_state.target_column = target
        st.session_state.history = history[:-1]
        return True

    @staticmethod
    def get_column_meta(col: str) -> dict:
        return copy.deepcopy(st.session_state.column_meta.get(col, _empty_column_meta()))

    @staticmethod
    def update_column_meta(col: str, **kwargs: Any) -> None:
        if not StateManager.has_dataset() or col not in st.session_state.df.columns:
            raise ValueError(f"Column {col!r} is not present in the dataset.")
        metadata = copy.deepcopy(st.session_state.column_meta)
        metadata[col].update(kwargs)
        StateManager.apply_operation(f"Update schema: {col}", st.session_state.df, column_meta=metadata)

    @staticmethod
    def get_columns_by_role(role: str) -> list[str]:
        if not StateManager.has_dataset():
            return []
        return [
            c for c, meta in st.session_state.column_meta.items()
            if c in st.session_state.df.columns and meta.get("role") == role
        ]

    @staticmethod
    def get_numeric_columns() -> list[str]:
        df = st.session_state.get("df")
        return [] if df is None else df.select_dtypes(include="number").columns.tolist()

    @staticmethod
    def get_categorical_columns() -> list[str]:
        df = st.session_state.get("df")
        return [] if df is None else df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    @staticmethod
    def has_dataset() -> bool:
        return st.session_state.get("df") is not None

    @staticmethod
    def clear() -> None:
        StateManager._invalidate_dependent_state()
        for key, value in _defaults().items():
            st.session_state[key] = value

