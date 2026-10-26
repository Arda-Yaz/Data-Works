import pandas as pd
import streamlit as st

from backend.state import StateManager
from backend import visualization as viz


_CHART_TYPES = ["Histogram", "Scatter", "Bar", "Box", "Line"]


def _cached_chart(key, calculate):
    revision = st.session_state.dataset_revision
    cache = st.session_state.get("_visualization_cache")
    if not isinstance(cache, dict) or cache.get("revision") != revision:
        cache = {"revision": revision, "values": {}}
        st.session_state["_visualization_cache"] = cache
    values = cache["values"]
    if key not in values:
        figure = calculate()
        if len(values) >= 4:
            values.pop(next(iter(values)))
        values[key] = figure
    return values[key]


def data_viz():
    st.title("Visualization")
    if not StateManager.has_dataset():
        st.warning("Upload a dataset first.")
        return
    df = st.session_state.df
    all_cols = df.columns.tolist()
    if not all_cols:
        st.info("No columns remain. Undo the last operation or upload another dataset.")
        return
    num_cols = [column for column in df.select_dtypes(include="number").columns if not pd.api.types.is_complex_dtype(df[column].dtype)]
    cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    st.subheader("Create a Chart")
    chart_type = st.selectbox("Chart type", _CHART_TYPES, key="vz_type")
    calculate = None
    key = None
    if chart_type == "Histogram":
        column = st.selectbox("Column", num_cols or all_cols, key="vz_hist_col")
        bins = st.slider("Bins", 5, 100, 30, key="vz_hist_bins")
        if column is not None:
            key, calculate = (chart_type, column, bins), lambda: viz.histogram(df, column, bins=bins)
    elif chart_type == "Scatter":
        first, second = st.columns(2)
        with first:
            x = st.selectbox("X axis", num_cols or all_cols, key="vz_sc_x")
        with second:
            y = st.selectbox("Y axis", num_cols or all_cols, key="vz_sc_y")
        color = st.selectbox("Color (optional)", [None] + cat_cols + num_cols, key="vz_sc_c")
        size = st.selectbox("Size (optional)", [None] + num_cols, key="vz_sc_s")
        if x is not None and y is not None:
            key, calculate = (chart_type, x, y, color, size), lambda: viz.scatter(df, x, y, color=color, size=size)
    elif chart_type == "Bar":
        x = st.selectbox("X axis (category)", cat_cols or all_cols, key="vz_bar_x")
        y = st.selectbox("Y axis (optional numeric)", [None] + num_cols, key="vz_bar_y")
        aggregation = st.selectbox("Aggregation", ["count", "mean", "sum"], key="vz_bar_agg")
        if x is not None:
            key, calculate = (chart_type, x, y, aggregation), lambda: viz.bar_chart(df, x, y=y, agg=aggregation)
    elif chart_type == "Box":
        column = st.selectbox("Value column", num_cols or all_cols, key="vz_box_col")
        group = st.selectbox("Group by (optional)", [None] + cat_cols, key="vz_box_g")
        if column is not None:
            key, calculate = (chart_type, column, group), lambda: viz.box_plot(df, column, group_by=group)
    elif chart_type == "Line":
        first, second = st.columns(2)
        with first:
            x = st.selectbox("X axis", all_cols, key="vz_ln_x")
        with second:
            y = st.selectbox("Y axis", num_cols or all_cols, key="vz_ln_y")
        if x is not None and y is not None:
            key, calculate = (chart_type, x, y), lambda: viz.line_chart(df, x, y)
    if calculate is not None:
        try:
            figure = _cached_chart(key, calculate)
            for note in viz.chart_notes(figure):
                st.caption(note)
            st.plotly_chart(figure, width="stretch")
        except (TypeError, ValueError) as exc:
            st.error(f"Could not build this chart. {exc}")
        else:
            if st.button("Save Chart", key="vz_save"):
                try:
                    StateManager.save_chart(figure)
                except ValueError as exc:
                    st.error(f"Chart was not saved. {exc}")
                else:
                    st.toast("Chart saved.")
                    st.rerun()
    if st.session_state.viz_charts:
        st.markdown("---")
        st.subheader("Saved Charts")
        if st.button("Clear Saved Charts", key="vz_clear_saved"):
            StateManager.clear_charts()
            st.rerun()
        for position, saved_figure in enumerate(st.session_state.viz_charts):
            for note in viz.chart_notes(saved_figure):
                st.caption(note)
            st.plotly_chart(saved_figure, width="stretch", key=f"saved_{position}")
