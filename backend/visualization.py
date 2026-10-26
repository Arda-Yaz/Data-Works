"""Bounded chart inputs; category aggregates always use the full dataset."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backend.eda import group_by_summary, value_counts_summary
from backend.resources import MAX_CHART_BYTES, MAX_PLOT_CATEGORIES, MAX_PLOT_ROWS


def validate_chart_size(figure):
    """Reject oversized labels/data before a chart enters session caches or UI."""
    if len(figure.to_json().encode("utf-8")) > MAX_CHART_BYTES:
        raise ValueError("This chart exceeds the chart size budget; use shorter labels or fewer rows/categories.")
    return figure


def _annotate(figure, notes):
    figure.update_layout(meta={"dataworks": {"notes": notes}})
    return validate_chart_size(figure)


def chart_notes(figure):
    metadata = figure.layout.meta
    return metadata.get("dataworks", {}).get("notes", []) if isinstance(metadata, dict) else []


def _categorical(series):
    if pd.api.types.is_datetime64_any_dtype(series.dtype) or pd.api.types.is_timedelta64_dtype(series.dtype):
        return False
    return not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype)


def _bounded(df, columns, *, category=None, ordered=False):
    selected = list(dict.fromkeys(column for column in columns if column is not None))
    data = df.loc[:, selected]
    notes = []
    if category is not None and _categorical(df[category]):
        counts = df[category].value_counts()
        observed = counts[counts > 0]
        if len(observed) > MAX_PLOT_CATEGORIES:
            data = data.loc[data[category].isin(observed.head(MAX_PLOT_CATEGORIES).index)]
            notes.append(f"Showing the {MAX_PLOT_CATEGORIES} most frequent categories of {category}; other categories are omitted.")
    if ordered:
        data = data.sort_values(selected[0], kind="stable")
    if len(data) > MAX_PLOT_ROWS:
        total = len(data)
        if ordered:
            positions = np.linspace(0, total - 1, MAX_PLOT_ROWS, dtype=int)
            data = data.iloc[positions]
            notes.append(f"Line reduced to {MAX_PLOT_ROWS:,} evenly spaced rows from {total:,} sorted rows; intermediate values are omitted.")
        else:
            data = data.sample(n=MAX_PLOT_ROWS, random_state=42)
            notes.append(f"Chart uses a deterministic sample of {MAX_PLOT_ROWS:,} from {total:,} rows; displayed distributions are approximate.")
    return data, notes


def histogram(df: pd.DataFrame, col, bins: int = 30) -> go.Figure:
    if _categorical(df[col]):
        data = value_counts_summary(df, col, top_n=MAX_PLOT_CATEGORIES)
        notes = [f"Showing up to {MAX_PLOT_CATEGORIES} observed categories; counts use all rows."]
        return _annotate(px.bar(x=data.iloc[:, 0], y=data.iloc[:, 1], title=f"Distribution of {col}", labels={"x": str(col), "y": "Count"}), notes)
    data, notes = _bounded(df, [col])
    return _annotate(px.histogram(x=data[col], nbins=bins, title=f"Distribution of {col}", labels={"x": str(col)}), notes)


def scatter(df: pd.DataFrame, x, y, color=None, size=None) -> go.Figure:
    data, notes = _bounded(df, [x, y, color, size], category=color)
    values = {"x": data[x], "y": data[y]}
    if color is not None:
        values["color"] = data[color]
    if size is not None:
        values["size"] = data[size]
    figure = px.scatter(pd.DataFrame(values), x="x", y="y", color="color" if color is not None else None,
                        size="size" if size is not None else None, labels={"x": str(x), "y": str(y), "color": str(color), "size": str(size)}, title=f"{y} vs {x}")
    return _annotate(figure, notes)


def bar_chart(df: pd.DataFrame, x, y=None, agg: str = "count") -> go.Figure:
    if y is None:
        data = value_counts_summary(df, x, top_n=MAX_PLOT_CATEGORIES)
        title = f"Value Counts of {x}"
    else:
        data = group_by_summary(df, x, y, agg)
        data = data.sort_values(data.columns[1], ascending=False, kind="stable", na_position="last").head(MAX_PLOT_CATEGORIES)
        title = f"{agg}({y}) by {x}"
    return _annotate(px.bar(x=data.iloc[:, 0], y=data.iloc[:, 1], title=title,
                           labels={"x": str(data.columns[0]), "y": str(data.columns[1])}),
                     [f"Showing up to {MAX_PLOT_CATEGORIES} categories ranked by aggregate value; each aggregate uses all rows."])


def box_plot(df: pd.DataFrame, col, group_by=None) -> go.Figure:
    data, notes = _bounded(df, [col, group_by], category=group_by)
    values = {"value": data[col]}
    if group_by is not None:
        values["group"] = data[group_by]
    figure = px.box(pd.DataFrame(values), x="group" if group_by is not None else None, y="value",
                    labels={"value": str(col), "group": str(group_by)},
                    title=f"Box Plot of {col}" + (f" by {group_by}" if group_by is not None else ""))
    return _annotate(figure, notes)


def line_chart(df: pd.DataFrame, x, y) -> go.Figure:
    data, notes = _bounded(df, [x, y], category=x, ordered=True)
    return _annotate(px.line(x=data[x], y=data[y], labels={"x": str(x), "y": str(y)}, title=f"{y} over {x}"), notes)


def heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    shown = corr_matrix.iloc[:MAX_PLOT_CATEGORIES, :MAX_PLOT_CATEGORIES]
    notes = [] if shown.shape == corr_matrix.shape else [f"Showing the first {MAX_PLOT_CATEGORIES} correlation columns."]
    figure = px.imshow(shown, x=[str(column) for column in shown.columns], y=[str(index) for index in shown.index],
                        text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap")
    figure.update_xaxes(type="category")
    figure.update_yaxes(type="category")
    figure.update_layout(width=700, height=600)
    return _annotate(figure, notes)
