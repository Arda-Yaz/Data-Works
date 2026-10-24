import pandas as pd
import streamlit as st


def before_after_viewer(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    """Side-by-side comparison of two DataFrames with diff stats."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before**")
        st.write(f"Rows: {len(before_df)}, Columns: {before_df.shape[1]}")
        st.dataframe(before_df.head(10))
    with c2:
        st.markdown("**After**")
        st.write(f"Rows: {len(after_df)}, Columns: {after_df.shape[1]}")
        st.dataframe(after_df.head(10))

    row_diff = len(after_df) - len(before_df)
    col_diff = after_df.shape[1] - before_df.shape[1]
    parts = []
    if row_diff:
        parts.append(f"Rows: {row_diff:+d}")
    if col_diff:
        parts.append(f"Columns: {col_diff:+d}")
    if parts:
        st.info("Change: " + ", ".join(parts))


def column_multi_selector(
    label: str,
    df: pd.DataFrame,
    dtype_filter: str | None = None,
    default_all: bool = False,
    key: str | None = None,
) -> list[str]:
    """Multi-select widget filtered by dtype."""
    cols = df.columns.tolist()
    if dtype_filter == "number":
        cols = df.select_dtypes(include="number").columns.tolist()
    elif dtype_filter == "category":
        cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    default = cols if default_all else []
    return st.multiselect(label, cols, default=default, key=key)


def operation_log(history: list[dict]) -> None:
    """Display the operation history list."""
    if not history:
        st.caption("No operations yet.")
        return
    for i, entry in enumerate(reversed(history), 1):
        st.text(f"{i}. {entry['name']}")
