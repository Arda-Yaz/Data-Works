import streamlit as st
import pandas as pd

from backend.cleaning import (
    find_mismatches_by_allowed_values,
    find_outliers,
    get_duplicate_summary,
    get_missing_summary,
    handle_missing,
    handle_outliers,
    remove_duplicates,
)
from backend.state import StateManager


_MISSING_STRATEGIES = [
    "drop_rows", "mean", "median", "mode", "constant", "interpolate", "drop_column",
]
_OUTLIER_METHODS = ["iqr", "zscore"]
_OUTLIER_ACTIONS = ["remove", "cap", "replace_nan", "ignore"]


def data_cleaning():
    st.title("Data Cleaning")

    if not StateManager.has_dataset():
        st.warning("Please upload a dataset in the Upload page first.")
        return

    df: pd.DataFrame = st.session_state.df

    # Undo at top
    if st.session_state.history:
        if st.button("↩ Undo Last Operation"):
            StateManager.undo()
            st.rerun()

    st.dataframe(df.head(5))

    tab_missing, tab_dupes, tab_outliers, tab_mismatch = st.tabs(
        ["Missing Values", "Duplicates", "Outliers", "Mismatches"]
    )

    # ── Tab 1: Missing Values ────────────────────────────────────────────
    with tab_missing:
        missing_summary = get_missing_summary(df)
        if missing_summary.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(missing_summary)

            strategies: dict[str, str] = {}
            fill_values: dict[str, str] = {}

            for col in missing_summary.index:
                suggested = missing_summary.loc[col, "suggested"]
                c1, c2 = st.columns([2, 1])
                with c1:
                    strat = st.selectbox(
                        f"Strategy for **{col}**",
                        _MISSING_STRATEGIES,
                        index=_MISSING_STRATEGIES.index(str(suggested)),
                        key=f"ms_{col}",
                    )
                    strategies[col] = strat
                with c2:
                    if strat == "constant":
                        fill_values[col] = st.text_input(
                            f"Fill value for {col}", key=f"fv_{col}"
                        )

            if st.button("Apply Missing Value Strategies", key="apply_missing"):
                StateManager.save_snapshot("Handle missing values")
                new_df = df.copy()
                for col, strat in strategies.items():
                    fv = fill_values.get(col)
                    new_df = handle_missing(new_df, col, strat, fill_value=fv)
                st.session_state.df = new_df
                st.success(
                    f"Applied. Rows: {len(df)} → {len(new_df)}, "
                    f"Columns: {df.shape[1]} → {new_df.shape[1]}"
                )
                st.rerun()

    # ── Tab 2: Duplicates ────────────────────────────────────────────────
    with tab_dupes:
        subset_cols = st.multiselect(
            "Check duplicates across specific columns (leave empty for all)",
            df.columns.tolist(),
            key="dup_subset",
        )
        keep = st.radio("Keep", ["first", "last", False], horizontal=True, key="dup_keep")
        
        n_dupes, sample = get_duplicate_summary(df, subset=subset_cols if subset_cols else None)
        st.write(f"**{n_dupes}** duplicate rows found.")
        
        if n_dupes > 0:
            with st.expander("Show sample duplicates"):
                st.dataframe(sample)

        if st.button("Remove Duplicates", key="apply_dupes"):
            StateManager.save_snapshot("Remove duplicates")
            subset = subset_cols if subset_cols else None
            new_df, removed = remove_duplicates(df, subset=subset, keep=keep)
            st.session_state.df = new_df
            st.success(f"Removed {removed} duplicate rows.")
            st.rerun()
    # ── Tab 3: Outliers ──────────────────────────────────────────────────
    with tab_outliers:
        outliers = find_outliers(df)
        has_outliers = any(len(v) > 0 for v in outliers.values())
        if not has_outliers:
            st.success("No outliers detected (IQR method).")
        else:
            summary_data = [
                {"Column": col, "Outlier Count": len(idxs)}
                for col, idxs in outliers.items()
                if len(idxs) > 0
            ]
            st.dataframe(pd.DataFrame(summary_data).set_index("Column"))

            actions: dict[str, tuple[str, str]] = {}
            for col, idxs in outliers.items():
                if not idxs:
                    continue
                c1, c2 = st.columns(2)
                with c1:
                    method = st.selectbox(
                        f"Method for **{col}**",
                        _OUTLIER_METHODS,
                        key=f"om_{col}",
                    )
                with c2:
                    action = st.selectbox(
                        f"Action for **{col}**",
                        _OUTLIER_ACTIONS,
                        key=f"oa_{col}",
                    )
                actions[col] = (method, action)

            if st.button("Apply Outlier Handling", key="apply_outliers"):
                StateManager.save_snapshot("Handle outliers")
                new_df = df.copy()
                for col, (method, action) in actions.items():
                    if action != "ignore":
                        new_df = handle_outliers(new_df, col, method=method, action=action)
                st.session_state.df = new_df
                st.success(f"Applied. Rows: {len(df)} → {len(new_df)}")
                st.rerun()

    # ── Tab 4: Mismatches ────────────────────────────────────────────────
    with tab_mismatch:
        column_meta = st.session_state.get("column_meta", {})
        allowed_dict = {
            col: m["allowed_values"]
            for col, m in column_meta.items()
            if m.get("allowed_values")
        }

        st.caption("Set allowed values per column in the Upload page schema, or add them here.")

        for col in df.columns:
            current = allowed_dict.get(col)
            val = st.text_input(
                f"Allowed values for **{col}** (comma-separated)",
                value=", ".join(current) if current else "",
                key=f"av_{col}",
            )
            if val.strip():
                allowed_dict[col] = [v.strip() for v in val.split(",")]

        if st.button("Find Mismatches", key="find_mismatch"):
            mismatches = find_mismatches_by_allowed_values(df, allowed_dict)
            any_found = False
            for col, idxs in mismatches.items():
                if idxs:
                    any_found = True
                    st.write(f"**{col}**: {len(idxs)} mismatches")
                    st.dataframe(df.loc[idxs, [col]].head(20))
            if not any_found:
                st.success("No mismatches found.")

        if st.button("Remove Mismatch Rows", key="apply_mismatch"):
            mismatches = find_mismatches_by_allowed_values(df, allowed_dict)
            all_bad = set()
            for idxs in mismatches.values():
                all_bad.update(idxs)
            if all_bad:
                StateManager.save_snapshot("Remove mismatch rows")
                new_df = df.drop(index=list(all_bad))
                st.session_state.df = new_df
                st.success(f"Removed {len(all_bad)} rows with mismatches.")
                st.rerun()
            else:
                st.info("Nothing to remove.")
