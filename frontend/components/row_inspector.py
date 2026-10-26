"""Read-only, literal search and bounded row browsing for arbitrary tables."""

import math

import pandas as pd
import streamlit as st


def _search_text(value):
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return ""
    return str(value)


def filter_rows(df, columns, query="", missing_only=False):
    """Return selected columns and matching rows, preserving positional order/index."""
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError("Row inspection requires a DataFrame with unique columns.")
    if not isinstance(columns, list) or any(column not in df.columns for column in columns):
        raise ValueError("Choose columns present in the dataset.")
    if not isinstance(query, str) or not isinstance(missing_only, bool):
        raise ValueError("Search text and missing-value filter have invalid types.")
    values = df.loc[:, columns]
    matches = pd.Series(True, index=pd.RangeIndex(len(df)))
    if query:
        needle = query.casefold()
        found = pd.Series(False, index=matches.index)
        for position in range(values.shape[1]):
            text = values.iloc[:, position].map(_search_text)
            found |= text.map(lambda value: needle in value.casefold()).to_numpy(dtype=bool)
        matches &= found
    if missing_only:
        matches &= values.isna().any(axis=1).to_numpy(dtype=bool)
    return values.iloc[matches.to_numpy(dtype=bool)].copy(deep=True)


def row_inspector(df, *, key_prefix="inspect"):
    """Display search/selection controls and at most 100 rows without editing data."""
    with st.expander("Inspect rows"):
        if df.shape[1] == 0:
            st.info("No columns are available to inspect.")
            return
        options = df.columns.tolist()
        columns_key = f"{key_prefix}_columns"
        if columns_key in st.session_state:
            st.session_state[columns_key] = [
                column for column in st.session_state[columns_key] if column in df.columns
            ]
        columns = st.multiselect(
            "Columns to inspect and search", options, default=options,
            format_func=repr, key=columns_key,
        )
        query = st.text_input("Find literal text (case-insensitive)", key=f"{key_prefix}_query")
        missing_only = st.checkbox("Only rows with missing values in these columns", key=f"{key_prefix}_missing")
        if not columns:
            st.info("Select at least one column to inspect.")
            return
        matches = filter_rows(df, columns, query, missing_only)
        st.caption(f"{len(matches):,} matching rows of {len(df):,} total rows.")
        if matches.empty:
            st.info("No rows match these filters.")
            return
        limit = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key=f"{key_prefix}_limit")
        pages = math.ceil(len(matches) / limit)
        page_key = f"{key_prefix}_page"
        if st.session_state.get(page_key, 1) > pages:
            st.session_state[page_key] = 1
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=page_key)
        start = (page - 1) * limit
        st.caption(f"Showing matches {start + 1:,}–{min(start + limit, len(matches)):,}; original row labels are retained.")
        st.dataframe(matches.iloc[start:start + limit], width="stretch")
