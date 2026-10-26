import pandas as pd
import streamlit as st


def before_after_viewer(
    before_df: pd.DataFrame, after_df: pd.DataFrame, *, summary: dict | None = None,
) -> None:
    """Side-by-side comparison of two DataFrames with diff stats."""
    summary = summary or {
        "rows_before": len(before_df), "rows_after": len(after_df),
        "columns_before": before_df.shape[1], "columns_after": after_df.shape[1],
    }
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before**")
        st.write(f"Rows: {summary['rows_before']}, Columns: {summary['columns_before']}")
        st.dataframe(before_df.head(20))
    with c2:
        st.markdown("**After**")
        st.write(f"Rows: {summary['rows_after']}, Columns: {summary['columns_after']}")
        st.dataframe(after_df.head(20))

    row_diff = summary["rows_after"] - summary["rows_before"]
    col_diff = summary["columns_after"] - summary["columns_before"]
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
