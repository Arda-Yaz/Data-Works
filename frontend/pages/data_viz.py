import streamlit as st
import pandas as pd

from backend.state import StateManager
from backend import visualization as viz


_CHART_TYPES = ["Histogram", "Scatter", "Bar", "Box", "Line"]


def data_viz():
    st.title("Visualization")

    if not StateManager.has_dataset():
        st.warning("Upload a dataset first.")
        return

    df: pd.DataFrame = st.session_state.df
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Persist a list of charts in session
    if "viz_charts" not in st.session_state:
        st.session_state.viz_charts = []

    st.subheader("Create a Chart")

    chart_type = st.selectbox("Chart type", _CHART_TYPES, key="vz_type")

    fig = None

    if chart_type == "Histogram":
        col = st.selectbox("Column", num_cols or all_cols, key="vz_hist_col")
        bins = st.slider("Bins", 5, 100, 30, key="vz_hist_bins")
        if col:
            fig = viz.histogram(df, col, bins=bins)

    elif chart_type == "Scatter":
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("X axis", num_cols or all_cols, key="vz_sc_x")
        with c2:
            y = st.selectbox("Y axis", num_cols or all_cols, key="vz_sc_y")
        color = st.selectbox("Color (optional)", [None] + cat_cols + num_cols, key="vz_sc_c")
        size = st.selectbox("Size (optional)", [None] + num_cols, key="vz_sc_s")
        if x and y:
            fig = viz.scatter(df, x, y, color=color, size=size)

    elif chart_type == "Bar":
        x = st.selectbox("X axis (category)", cat_cols or all_cols, key="vz_bar_x")
        y = st.selectbox("Y axis (optional numeric)", [None] + num_cols, key="vz_bar_y")
        agg = st.selectbox("Aggregation", ["count", "mean", "sum"], key="vz_bar_agg")
        if x:
            fig = viz.bar_chart(df, x, y=y, agg=agg)

    elif chart_type == "Box":
        col = st.selectbox("Value column", num_cols or all_cols, key="vz_box_col")
        group = st.selectbox("Group by (optional)", [None] + cat_cols, key="vz_box_g")
        if col:
            fig = viz.box_plot(df, col, group_by=group)

    elif chart_type == "Line":
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("X axis", all_cols, key="vz_ln_x")
        with c2:
            y = st.selectbox("Y axis", num_cols or all_cols, key="vz_ln_y")
        if x and y:
            fig = viz.line_chart(df, x, y)

    if fig is not None:
        st.plotly_chart(fig, width='stretch')

    # ── Saved charts gallery ─────────────────────────────────────────────
    if st.session_state.viz_charts:
        st.markdown("---")
        st.subheader("Saved Charts")
        for i, saved_fig in enumerate(st.session_state.viz_charts):
            st.plotly_chart(saved_fig, width='stretch', key=f"saved_{i}")
