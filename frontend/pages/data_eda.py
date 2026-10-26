import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.eda import correlation_matrix, group_by_summary, univariate_stats, value_counts_summary
from backend.state import StateManager
from backend.visualization import chart_notes, heatmap, histogram, validate_chart_size


def _valid_selection(key, choices):
    if key in st.session_state and st.session_state[key] not in choices:
        del st.session_state[key]


def _cached(key, calculate):
    revision = st.session_state.dataset_revision
    cache = st.session_state.get("_analysis_cache")
    if not isinstance(cache, dict) or cache.get("revision") != revision:
        cache = {"revision": revision, "values": {}}
        st.session_state["_analysis_cache"] = cache
    values = cache["values"]
    if key not in values:
        result = calculate()
        if len(values) >= 8:
            values.pop(next(iter(values)))
        values[key] = result
    return values[key]


def _univariate(df, column):
    stats = univariate_stats(df, column)
    figure = None
    if stats["type"] == "numeric" and stats["finite_count"]:
        observed = df[column].dropna()
        observed = observed.iloc[np.flatnonzero(np.isfinite(observed.to_numpy(dtype=float)))]
        figure = histogram(observed.to_frame(), column)
    elif stats["type"] != "numeric":
        counts = value_counts_summary(df, column, top_n=20)
        if not counts.empty:
            figure = validate_chart_size(px.bar(x=counts.iloc[:, 0], y=counts.iloc[:, 1], labels={"x": str(column), "y": "Count"}, title=f"Value counts of {column}"))
    return stats, figure


def _grouped(df, group_column, aggregate_column, function):
    summary = group_by_summary(df, group_column, aggregate_column, function)
    figure = None
    if not summary.empty and summary.iloc[:, 1].notna().any():
        shown = summary.head(50)
        figure = validate_chart_size(px.bar(x=shown.iloc[:, 0], y=shown.iloc[:, 1], labels={"x": str(summary.columns[0]), "y": str(summary.columns[1])}, title=f"{function}({aggregate_column}) by {group_column}"))
    return summary.head(1000), len(summary), figure


def data_eda():
    st.title("Exploratory Data Analysis")
    if not StateManager.has_dataset():
        st.warning("Upload a dataset first.")
        return
    df = st.session_state.df
    if df.shape[1] == 0:
        st.info("No columns remain. Undo the last operation or upload another dataset.")
        return
    tab_uni, tab_corr, tab_group = st.tabs(["Univariate Analysis", "Correlation", "Group-By"], key="eda_section", on_change="rerun")
    columns = df.columns.tolist()
    if tab_uni.open:
        with tab_uni:
            _valid_selection("eda_uni_col", columns)
            column = st.selectbox("Select a column", columns, key="eda_uni_col")
            if column is not None:
                try:
                    stats, figure = _cached(("univariate", column), lambda: _univariate(df, column))
                    st.json(stats)
                    if stats["type"] == "numeric":
                        if stats["non_finite"]:
                            st.caption(f"{stats['non_finite']} non-finite values were excluded from numeric summaries and the histogram.")
                        if not stats["finite_count"]:
                            st.info("No finite numeric values are available to plot.")
                    elif figure is None:
                        st.info("No observed values are available to plot.")
                    else:
                        st.caption("Showing up to 20 observed categories; counts use all rows.")
                    if figure is not None:
                        for note in chart_notes(figure):
                            st.caption(note)
                        st.plotly_chart(figure, width="stretch")
                except (TypeError, ValueError) as exc:
                    st.error(f"Could not analyze this column. {exc}")
    if tab_corr.open:
        with tab_corr:
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="corr_method")
            numeric = [name for name in columns if pd.api.types.is_numeric_dtype(df[name].dtype) and not pd.api.types.is_bool_dtype(df[name].dtype) and not pd.api.types.is_complex_dtype(df[name].dtype)]
            chosen = st.multiselect("Correlation columns (maximum 50)", numeric, default=numeric[:20], key="corr_columns")
            if len(chosen) > 50:
                st.warning("Select at most 50 correlation columns.")
            else:
                try:
                    limit = 2000 if method == "kendall" else 10000
                    if len(df) > limit:
                        st.caption(f"Correlation uses a deterministic sample of {limit:,} from {len(df):,} rows; estimates are approximate.")
                    def calculate_correlation():
                        if len(df) > limit:
                            positions = np.random.RandomState(42).choice(len(df), size=limit, replace=False)
                            source = df.iloc[positions, df.columns.get_indexer(chosen)]
                        else:
                            source = df.loc[:, chosen]
                        return correlation_matrix(source, method=method)
                    correlation = _cached(("correlation", method, tuple(chosen)), calculate_correlation)
                    if correlation.empty:
                        st.info("Need at least two varying numeric columns with finite observations for correlation.")
                    else:
                        st.caption("Constant or empty columns are excluded. Correlations use finite pairs of observations.")
                        st.plotly_chart(heatmap(correlation), width="stretch")
                        with st.expander("Raw correlation table"):
                            st.dataframe(correlation.rename(index=str, columns=str))
                except (TypeError, ValueError) as exc:
                    st.error(f"Could not calculate correlations. {exc}")
    if tab_group.open:
        with tab_group:
            first, second, third = st.columns(3)
            with first:
                _valid_selection("gb_col", columns)
                group_column = st.selectbox("Group by", columns, key="gb_col")
            with second:
                numeric = [name for name in columns if pd.api.types.is_numeric_dtype(df[name].dtype) and not pd.api.types.is_bool_dtype(df[name].dtype) and not pd.api.types.is_complex_dtype(df[name].dtype)]
                choices = numeric or columns
                _valid_selection("gb_agg_col", choices)
                aggregate_column = st.selectbox("Aggregate column", choices, key="gb_agg_col")
            with third:
                functions = ["mean", "sum", "count", "min", "max", "median"] if aggregate_column in numeric else ["count"]
                _valid_selection("gb_func", functions)
                function = st.selectbox("Function", functions, key="gb_func")
            if group_column is not None and aggregate_column is not None:
                try:
                    summary, total, figure = _cached(("group", group_column, aggregate_column, function), lambda: _grouped(df, group_column, aggregate_column, function))
                    st.dataframe(summary.rename(columns=str))
                    if total > len(summary):
                        st.caption(f"The table shows the first {len(summary):,} of {total:,} groups; aggregates use all rows.")
                    if summary.empty:
                        st.info("No observed, non-missing group values are available.")
                    elif figure is not None:
                        if total > 50:
                            st.caption("The chart shows the first 50 groups in the summary table; aggregates use all rows.")
                        st.plotly_chart(figure, width="stretch")
                    else:
                        st.info("The selected aggregate has no observed values.")
                except (TypeError, ValueError) as exc:
                    st.error(f"Could not summarize these groups. {exc}")
