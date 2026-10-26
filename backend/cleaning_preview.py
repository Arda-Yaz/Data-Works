"""Describe a cleaning candidate without modifying data or session state."""

from numbers import Integral

import numpy as np
import pandas as pd


_CHANGE_COLUMNS = [
    "column", "before_dtype", "after_dtype", "changed_cells",
    "filled_missing", "introduced_missing",
]


def _validate_mapping(before, after, sample_size):
    if not isinstance(before, pd.DataFrame) or not isinstance(after, pd.DataFrame):
        raise ValueError("A cleaning preview requires before and after DataFrames.")
    if not before.columns.is_unique or not after.columns.is_unique:
        raise ValueError("Cleaning previews require unique column names.")
    if isinstance(sample_size, bool) or not isinstance(sample_size, Integral) or sample_size < 0:
        raise ValueError("The preview sample size must be a non-negative integer.")
    columns = before.columns.get_indexer(after.columns)
    if (columns < 0).any() or (np.diff(columns) <= 0).any():
        raise ValueError("A cleaning candidate may drop columns but must preserve their original order.")
    positions = list(after.index)
    if any(
        isinstance(position, (bool, np.bool_)) or not isinstance(position, Integral)
        or not 0 <= position < len(before)
        for position in positions
    ):
        raise ValueError("Candidate row indices must be integer positions in the original dataset.")
    positions = np.asarray(positions, dtype=np.intp)
    if (np.diff(positions) <= 0).any():
        raise ValueError("Candidate row positions must be unique and preserve original row order.")
    return positions


def _column_difference(before: pd.Series, after: pd.Series):
    old_missing = before.isna().to_numpy(dtype=bool)
    new_missing = after.isna().to_numpy(dtype=bool)
    changed = old_missing ^ new_missing
    present = ~(old_missing | new_missing)
    if present.any():
        # Object conversion prevents categorical-set comparison failures and avoids
        # coercing exact integers into floats when their dtypes differ.
        old_values = before.loc[present].astype(object).reset_index(drop=True)
        new_values = after.loc[present].astype(object).reset_index(drop=True)
        try:
            changed[present] = ~old_values.eq(new_values).to_numpy(dtype=bool)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {before.name!r} contains values that cannot be compared for preview.") from exc
    return changed, old_missing & ~new_missing, ~old_missing & new_missing


def describe_cleaning_preview(
    before: pd.DataFrame,
    positional_after: pd.DataFrame,
    sample_size: int = 20,
) -> dict:
    """Return exact impact counts and bounded samples for a prepared candidate.

    ``positional_after`` must retain unique integer positions from
    ``before.reset_index(drop=True)``. This distinguishes rows with duplicate
    original index labels. Returned candidate/sample frames restore those labels.
    Value-change counts concern retained cells; row/column removals are separate.
    """
    positions = _validate_mapping(before, positional_after, sample_size)
    candidate = positional_after.copy(deep=True)
    candidate.index = before.index.take(positions)
    old_retained = before.iloc[positions].reset_index(drop=True)
    new_retained = positional_after.reset_index(drop=True)
    changed_rows = np.zeros(len(positions), dtype=bool)
    changes = []
    totals = {"changed_cells": 0, "filled_missing": 0, "introduced_missing": 0}
    for column in positional_after.columns:
        old = old_retained[column]
        new = new_retained[column]
        changed, filled, introduced = _column_difference(old, new)
        changed_rows |= changed
        counts = {
            "changed_cells": int(changed.sum()),
            "filled_missing": int(filled.sum()),
            "introduced_missing": int(introduced.sum()),
        }
        for key, value in counts.items():
            totals[key] += value
        if changed.any() or old.dtype != new.dtype:
            changes.append({
                "column": column,
                "before_dtype": str(old.dtype),
                "after_dtype": str(new.dtype),
                **counts,
            })

    dropped_columns = [column for column in before.columns if column not in positional_after.columns]
    affected = np.ones(len(before), dtype=bool)
    if not dropped_columns:
        affected[positions] = changed_rows
    affected_positions = np.flatnonzero(affected)
    sampled_positions = affected_positions[:sample_size]
    surviving_samples = np.isin(positions, sampled_positions)
    summary = {
        "rows_before": len(before), "rows_after": len(candidate),
        "rows_removed": len(before) - len(candidate),
        "columns_before": before.shape[1], "columns_after": candidate.shape[1],
        "columns_removed": len(dropped_columns),
        "changed_rows": int(changed_rows.sum()),
        "missing_before": int(before.isna().sum().sum()),
        "missing_after": int(candidate.isna().sum().sum()),
        **totals,
    }
    return {
        "candidate": candidate,
        "summary": summary,
        "column_changes": pd.DataFrame(changes, columns=_CHANGE_COLUMNS),
        "dropped_columns": dropped_columns,
        "before_sample": before.iloc[sampled_positions].copy(deep=True),
        "after_sample": candidate.iloc[np.flatnonzero(surviving_samples)].copy(deep=True),
        "affected_positions": affected_positions,
    }
