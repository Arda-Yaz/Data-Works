"""Practical per-session budgets and one process-wide ML execution slot."""

from contextlib import contextmanager
import sys
from threading import BoundedSemaphore

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


MAX_UPLOAD_BYTES = 50 * 1024**2
MAX_DATASET_ROWS = 1_000_000
MAX_DATASET_COLUMNS = 10_000
MAX_DATASET_CELLS = 5_000_000
MAX_DATASET_BYTES = 128 * 1024**2
MAX_HISTORY_BYTES = 128 * 1024**2
MAX_CHARTS = 10
MAX_CHART_BYTES = 5 * 1024**2
MAX_CHART_TOTAL_BYTES = 20 * 1024**2
MAX_EXPERIMENT_BYTES = 64 * 1024**2
MAX_EXPERIMENT_TOTAL_BYTES = 128 * 1024**2
MAX_PLOT_ROWS = 10_000
MAX_PLOT_CATEGORIES = 50
MAX_ML_ROWS = 50_000
MAX_ML_CELLS = 5_000_000
MAX_KERNEL_ROWS = 10_000
MAX_KNN_PAIRS = 20_000_000

_ML_SLOT = BoundedSemaphore(1)


def frame_bytes(frame):
    return int(frame.memory_usage(index=True, deep=True).sum() + frame.columns.memory_usage(deep=True))


def validate_dataset(frame):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("The dataset must be a pandas DataFrame.")
    if len(frame) > MAX_DATASET_ROWS or frame.shape[1] > MAX_DATASET_COLUMNS or frame.size > MAX_DATASET_CELLS:
        raise ValueError("This dataset exceeds the row/cell budget. Use fewer rows or columns.")
    if frame_bytes(frame) > MAX_DATASET_BYTES:
        raise ValueError("This dataset exceeds the 128 MiB memory budget. Use a smaller dataset.")


def estimate_bytes(value):
    """Estimate retained Python/array storage, counting shared references once.

    This is a retention budget, not a guarantee about process RSS or peak memory
    during a pandas/scikit-learn operation. Tree states include their native arrays.
    """
    seen = set()
    tree_states = []  # Keep temporary state dictionaries alive so object IDs cannot be reused.

    def size(item):
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(item, (pd.DataFrame, pd.Series)):
            usage = item.memory_usage(index=True, deep=True)
            return int(usage.sum() if isinstance(usage, pd.Series) else usage)
        if isinstance(item, np.ndarray):
            return item.nbytes + 128 + (sum(size(cell) for cell in item.flat) if item.dtype.hasobject else 0)
        own = sys.getsizeof(item)
        if isinstance(item, dict):
            return own + sum(size(key) + size(cell) for key, cell in item.items())
        if isinstance(item, (list, tuple, set, frozenset)):
            return own + sum(size(cell) for cell in item)
        if type(item).__module__ == "sklearn.tree._tree" and hasattr(item, "__getstate__"):
            state = item.__getstate__()
            tree_states.append(state)
            return own + size(state)
        if hasattr(item, "__dict__"):
            return own + size(vars(item))
        return own

    return size(value)


def validate_ml_shape(rows, columns):
    if rows > MAX_ML_ROWS:
        raise ValueError(f"ML is limited to {MAX_ML_ROWS:,} rows per run. Select a smaller dataset.")
    if rows * columns > MAX_ML_CELLS:
        raise ValueError("Encoded features exceed the ML memory budget. Select fewer features or rows.")


@contextmanager
def ml_job():
    """Reject overlapping training in this process and bound numerical threads."""
    if not _ML_SLOT.acquire(blocking=False):
        raise ValueError("Another ML run is active. Try again after it finishes.")
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        _ML_SLOT.release()
