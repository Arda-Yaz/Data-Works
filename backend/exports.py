"""Portable table exports without changing the working dataset."""

import re

import pandas as pd


_FORMULA = re.compile(r"^[\s\ufeff]*[=+\-@]")


def _spreadsheet_text(value):
    if isinstance(value, str) and _FORMULA.match(value):
        return "'" + value
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def export_csv(df: pd.DataFrame, spreadsheet_safe: bool = True, include_index: bool = False) -> bytes:
    """Return UTF-8 CSV; safe mode prefixes formula-like text with an apostrophe.

    Raw mode preserves cell values. CSV is textual and does not preserve pandas
    dtypes or the distinction between all missing-value representations.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Choose a DataFrame to export.")
    if not isinstance(spreadsheet_safe, bool) or not isinstance(include_index, bool):
        raise ValueError("CSV export options must be booleans.")
    frame = df.copy(deep=True)
    if spreadsheet_safe:
        for position in range(frame.shape[1]):
            series = frame.iloc[:, position]
            if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype):
                frame.isetitem(position, series.astype(object).map(_spreadsheet_text))
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = pd.MultiIndex.from_tuples(
                [tuple(_spreadsheet_text(value) for value in column) for column in frame.columns],
                names=[_spreadsheet_text(name) for name in frame.columns.names],
            )
        else:
            frame.columns = pd.Index([_spreadsheet_text(column) for column in frame.columns], name=frame.columns.name)
        if include_index:
            if isinstance(frame.index, pd.MultiIndex):
                frame.index = pd.MultiIndex.from_tuples(
                    [tuple(_spreadsheet_text(value) for value in row) for row in frame.index],
                    names=[_spreadsheet_text(name) for name in frame.index.names],
                )
            else:
                frame.index = pd.Index([_spreadsheet_text(value) for value in frame.index], name=_spreadsheet_text(frame.index.name))
    return frame.to_csv(index=include_index, lineterminator="\n").encode("utf-8-sig")
