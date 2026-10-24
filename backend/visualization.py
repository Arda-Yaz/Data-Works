import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


def histogram(df: pd.DataFrame, col: str, bins: int = 30) -> go.Figure:
    return px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
) -> go.Figure:
    return px.scatter(df, x=x, y=y, color=color, size=size,
                      title=f"{y} vs {x}")


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str | None = None,
    agg: str = "count",
) -> go.Figure:
    if y is None or agg == "count":
        data = df[x].value_counts().reset_index()
        data.columns = [x, "count"]
        return px.bar(data, x=x, y="count", title=f"Value Counts of {x}")
    grouped = df.groupby(x)[y].agg(agg).reset_index()
    grouped.columns = [x, f"{agg}({y})"]
    return px.bar(grouped, x=x, y=f"{agg}({y})",
                  title=f"{agg}({y}) by {x}")


def box_plot(
    df: pd.DataFrame,
    col: str,
    group_by: str | None = None,
) -> go.Figure:
    return px.box(df, x=group_by, y=col,
                  title=f"Box Plot of {col}" + (f" by {group_by}" if group_by else ""))


def line_chart(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    return px.line(df.sort_values(x), x=x, y=y, title=f"{y} over {x}")


def heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap",
    )
    fig.update_layout(width=700, height=600)
    return fig
