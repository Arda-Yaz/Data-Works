import pandas as pd
import streamlit as st

from backend.eda import correlation_matrix, group_by_summary, univariate_stats, value_counts_summary
from backend.state import StateManager
from backend.visualization import bar_chart, heatmap, histogram


def data_eda():
    st.title("Exploratory Data Analysis")

    if not StateManager.has_dataset():
        st.warning("Upload a dataset first.")
        return

    df: pd.DataFrame = st.session_state.df

    tab_uni, tab_corr, tab_group = st.tabs(
        ["Univariate Analysis", "Correlation", "Group-By"]
    )

    # ── Univariate ───────────────────────────────────────────────────────
    with tab_uni:
        col = st.selectbox("Select a column", df.columns.tolist(), key="eda_uni_col")
        if col:
            stats = univariate_stats(df, col)
            st.json(stats)

            if stats["type"] == "numeric":
                fig = histogram(df, col)
            else:
                vc = value_counts_summary(df, col)
                fig = bar_chart(df, col)
            st.plotly_chart(fig, width='stretch')

    # ── Correlation ──────────────────────────────────────────────────────
    with tab_corr:
        method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="corr_method")
        corr = correlation_matrix(df, method=method)
        if corr.empty:
            st.info("Need at least 2 numeric columns for a correlation matrix.")
        else:
            fig = heatmap(corr)
            st.plotly_chart(fig, width='stretch')
            with st.expander("Raw correlation table"):
                st.dataframe(corr)

    # ── Group-By ─────────────────────────────────────────────────────────
    with tab_group:
        c1, c2, c3 = st.columns(3)
        with c1:
            group_col = st.selectbox("Group by", df.columns.tolist(), key="gb_col")
        with c2:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            agg_col = st.selectbox(
                "Aggregate column",
                numeric_cols if numeric_cols else df.columns.tolist(),
                key="gb_agg_col",
            )
        with c3:
            agg_func = st.selectbox(
                "Function", ["mean", "sum", "count", "min", "max", "median"],
                key="gb_func",
            )

        if group_col and agg_col:
            result = group_by_summary(df, group_col, agg_col, agg_func)
            st.dataframe(result)
            agg_label = f"{agg_func}({agg_col})"
            fig = bar_chart(result, group_col, y=agg_label, agg="sum")
            st.plotly_chart(fig, width='stretch')
