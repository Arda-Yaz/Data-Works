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
    st.caption("Edit inferred types or roles, then click **Confirm Schema**.")

    schema_rows = []
    for col_name in df.columns:
        m = column_meta.get(col_name, {})
        schema_rows.append({
            "column": col_name,
            "original_dtype": m.get("original_dtype", str(df[col_name].dtype)),
            "inferred_dtype": m.get("inferred_dtype", str(df[col_name].dtype)),
            "confidence": m.get("confidence", 0.0),
            "role": m.get("role", "feature"),
        })

    schema_df = pd.DataFrame(schema_rows).set_index("column")

    edited = st.data_editor(
        schema_df,
        column_config={
            "original_dtype": st.column_config.TextColumn("Original Type", disabled=True),
            "inferred_dtype": st.column_config.SelectboxColumn(
                "Inferred Type", options=_DTYPE_OPTIONS,
            ),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.0%%",
            ),
            "role": st.column_config.SelectboxColumn("Role", options=_ROLE_OPTIONS),
        },
        width='stretch',
        key="schema_editor",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Confirm Schema", type="primary"):
            # Apply user edits back to column_meta
            for col_name in edited.index:
                row = edited.loc[col_name]
                StateManager.update_column_meta(
                    col_name,
                    user_dtype=row["inferred_dtype"],
                    role=row["role"],
                )
            # Cast dtypes
            st.session_state.df = apply_inferred_dtypes(df, st.session_state.column_meta)
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
