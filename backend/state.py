import copy
from typing import Any

import pandas as pd
import streamlit as st


# ── Column metadata defaults ────────────────────────────────────────────────
def _empty_column_meta() -> dict:
    return {
        "original_dtype": "object",
        "inferred_dtype": "object",
        "confidence": 0.0,
        "user_dtype": None,
        "role": "feature",
        "cardinality": 0,
        "is_categorical": False,
        "allowed_values": None,
    }


MAX_HISTORY = 10


class StateManager:
    """Thin wrapper around st.session_state for DataWorks."""

    # ── Initialisation ───────────────────────────────────────────────────
    @staticmethod
    def init_session() -> None:
        defaults: dict[str, Any] = {
            "original_df": None,
            "df": None,
            "column_meta": {},
            "history": [],
            "target_column": None,
            "schema_confirmed": False,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ── Dataset loading ──────────────────────────────────────────────────
    @staticmethod
    def load_dataset(df: pd.DataFrame, column_meta: dict) -> None:
        st.session_state.original_df = df.copy()
        st.session_state.df = df.copy()
        st.session_state.column_meta = column_meta
        st.session_state.history = []
        st.session_state.schema_confirmed = False
        st.session_state.target_column = None

    # ── Snapshot / undo ──────────────────────────────────────────────────
    @staticmethod
    def save_snapshot(operation_name: str) -> None:
        snapshot = {
            "name": operation_name,
            "df": st.session_state.df.copy(),
            "column_meta": copy.deepcopy(st.session_state.column_meta),
        }
        history: list = st.session_state.history
        history.append(snapshot)
        if len(history) > MAX_HISTORY:
            history.pop(0)

    @staticmethod
    def undo() -> bool:
        history: list = st.session_state.history
        if not history:
            return False
        snapshot = history.pop()
        st.session_state.df = snapshot["df"]
        st.session_state.column_meta = snapshot["column_meta"]
        return True

    # ── Column metadata helpers ──────────────────────────────────────────
    @staticmethod
    def get_column_meta(col: str) -> dict:
        return st.session_state.column_meta.get(col, _empty_column_meta())

    @staticmethod
    def update_column_meta(col: str, **kwargs: Any) -> None:
        meta = st.session_state.column_meta.setdefault(col, _empty_column_meta())
        meta.update(kwargs)

    @staticmethod
    def get_columns_by_role(role: str) -> list[str]:
        return [
            c for c, m in st.session_state.column_meta.items() if m.get("role") == role
        ]

    @staticmethod
    def get_numeric_columns() -> list[str]:
        df: pd.DataFrame = st.session_state.df
        if df is None:
            return []
        return df.select_dtypes(include="number").columns.tolist()

    @staticmethod
    def get_categorical_columns() -> list[str]:
        df: pd.DataFrame = st.session_state.df
        if df is None:
            return []
        return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ── Convenience predicates ───────────────────────────────────────────
    @staticmethod
    def has_dataset() -> bool:
        return st.session_state.get("df") is not None

    @staticmethod
    def clear() -> None:
        for key in ("original_df", "df", "column_meta", "history",
                     "target_column", "schema_confirmed"):
            st.session_state.pop(key, None)
