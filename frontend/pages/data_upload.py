import copy

import pandas as pd
import streamlit as st

from backend.ingestion import apply_inferred_dtypes, infer_column_meta, parse_file
from backend.state import StateManager

_ROLE_OPTIONS = ["feature", "id", "datetime", "ignore"]
_DTYPE_OPTIONS = ["object", "int64", "float64", "bool", "datetime64[ns]", "category"]


def data_upload():
    st.title("Data Upload & Schema Review")

    if not StateManager.has_dataset():
        uploaded_file = st.file_uploader(
            "Upload a dataset",
            type=["csv", "tsv", "txt", "xlsx", "xls", "json"],
        )
        if uploaded_file is not None:
            try:
                raw_df = parse_file(uploaded_file)
                column_meta = infer_column_meta(raw_df)
                StateManager.load_dataset(raw_df, column_meta)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to parse file: {e}")
                return
        else:
            st.info("Upload a file to get started.")
            return

    # ── Dataset loaded ───────────────────────────────────────────────────
    df: pd.DataFrame = st.session_state.df
    column_meta: dict = st.session_state.column_meta

    st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")

    # Row preview
    max_rows = max(1, df.shape[0])
    n = st.sidebar.slider(
        "Rows to preview", 1, min(max_rows, 100),
        5 if max_rows >= 5 else 1,
    )
    st.dataframe(df.head(n))

    # ── Editable schema table ────────────────────────────────────────────
    st.subheader("Schema Review")
    st.caption("Review the conversion preview, then click **Confirm Schema**. "
               "Units, compound values, and ambiguous dates stay as text. "
               "For dates outside YYYY-MM-DD, specify a format such as %d/%m/%Y.")

    schema_rows = []
    for col_name in df.columns:
        m = column_meta.get(col_name, {})
        schema_rows.append({
            "column": col_name,
            "original_dtype": m.get("original_dtype", str(df[col_name].dtype)),
            "inferred_dtype": m.get("user_dtype") or m.get("inferred_dtype", str(df[col_name].dtype)),
            "confidence": m.get("confidence", 0.0),
            "role": m.get("role", "feature"),
            "allowed_values": ", ".join(m.get("allowed_values") or []),
            "date_format": m.get("date_format") or "",
        })

    schema_df = pd.DataFrame(schema_rows).set_index("column")

    edited = st.data_editor(
        schema_df,
        column_config={
            "original_dtype": st.column_config.TextColumn("Original Type", disabled=True),
            "inferred_dtype": st.column_config.SelectboxColumn(
                "Requested Type", options=_DTYPE_OPTIONS,
            ),
            "confidence": st.column_config.ProgressColumn(
                "Parser Coverage", min_value=0.0, max_value=1.0, format="%.0%%",
            ),
            "role": st.column_config.SelectboxColumn("Role", options=_ROLE_OPTIONS),
            "allowed_values": st.column_config.TextColumn(
                "Allowed Values", help="Optional comma-separated values for mismatch checks."
            ),
            "date_format": st.column_config.TextColumn(
                "Date Format", help="For example, %d/%m/%Y or %m/%d/%Y. Blank means YYYY-MM-DD."
            ),
        },
        width='stretch',
        key="schema_editor",
    )

    candidate_meta = copy.deepcopy(column_meta)
    for col_name, row in edited.iterrows():
        date_format = row.get("date_format", "")
        candidate_meta.setdefault(col_name, {}).update(
            user_dtype=row["inferred_dtype"],
            role=row["role"],
            allowed_values=[
                value.strip()
                for value in str(row.get("allowed_values", "")).split(",")
                if value.strip()
            ] or None,
            date_format=str(date_format).strip() if pd.notna(date_format) and date_format else None,
        )

    candidate_df = None
    try:
        candidate_df = apply_inferred_dtypes(df, candidate_meta)
    except ValueError as exc:
        st.error(str(exc))
    else:
        preview = []
        for name in df.columns:
            if str(df[name].dtype) != str(candidate_df[name].dtype):
                preview.append({
                    "column": name,
                    "before_type": str(df[name].dtype),
                    "after_type": str(candidate_df[name].dtype),
                    "new_missing": int((df[name].notna() & candidate_df[name].isna()).sum()),
                    "before_examples": repr(df[name].head(3).tolist()),
                    "after_examples": repr(candidate_df[name].head(3).tolist()),
                })
        st.subheader("Conversion Preview")
        if preview:
            st.dataframe(pd.DataFrame(preview), width="stretch")
        else:
            st.caption("No type conversions requested.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Confirm Schema", type="primary", disabled=candidate_df is None):
            st.session_state.df = candidate_df
            st.session_state.column_meta = candidate_meta
            st.session_state.schema_confirmed = True
            st.success("Schema confirmed and types applied.")
            st.rerun()

    with col2:
        if st.button("Clear Dataset"):
            StateManager.clear()
            st.rerun()

    # ── Expanders ────────────────────────────────────────────────────────
    with st.expander("Summary Statistics"):
        st.dataframe(df.describe(include="all").T)

    with st.expander("Missing Values"):
        missing = df.isna().sum().to_frame("missing")
        missing["%"] = (100 * missing["missing"] / len(df)).round(2)
        st.dataframe(missing)
