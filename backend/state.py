import copy
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import streamlit as st

from backend.dataset_preparation import (
    empty_column_meta as _empty_column_meta,
    prepare_dataset as _prepare_dataset,
    _schema_dtype,
)
from backend.operations import validate_operation
from backend import resources


MAX_HISTORY = 10
MAX_EXPERIMENTS = 5
_WIDGET_PREFIXES = ("ms_", "fv_", "om_", "oa_", "av_", "ml_", "vz_", "eda_", "corr_", "gb_", "ingest_", "text_", "inspect_")
_WIDGET_KEYS = {
    "schema_editor", "data_upload_file", "preview_rows", "dup_subset", "dup_keep",
    "apply_missing", "apply_dupes", "apply_outliers", "find_mismatch",
    "apply_mismatch", "apply_allowed_values",
    "preview_missing", "preview_dupes", "preview_outliers",
    "outlier_columns", "outlier_include_protected",
    "project_file", "project_name_input",
}


def _defaults() -> dict:
    from backend.fulfillment_state import new_project

    return {
        "original_df": None, "df": None, "column_meta": {}, "history": [],
        "target_column": None, "schema_confirmed": False, "viz_charts": [],
        "statistical_operations": [],
        "cleaning_previews": {},
        "original_column_meta": {}, "operation_log": [], "experiments": [],
        "dataset_id": None, "data_version": None, "project_name": "DataWorks project",
        "_analysis_cache": {},
        "_visualization_cache": {},
        "_upload_summary_cache": {},
        "fulfillment": new_project(),
    }


def _identifier(value, field):
    try:
        return UUID(value).hex
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Project field {field!r} must be a valid identifier.") from exc


def _statistical_names(value):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Statistical operation names must be a list of strings.")
    return list(value)


def _bounded_history(history):
    """Retain the newest owned snapshots that fit both undo limits."""
    retained = list(history[-MAX_HISTORY:])
    sizes = [resources.estimate_bytes(snapshot) for snapshot in retained]
    total = sum(sizes)
    while retained and total > resources.MAX_HISTORY_BYTES:
        total -= sizes.pop(0)
        retained.pop(0)
    return retained


def _chart_sizes(charts):
    import plotly.io as pio

    if len(charts) > resources.MAX_CHARTS:
        raise ValueError(f"At most {resources.MAX_CHARTS} charts can be saved; clear saved charts before adding more.")
    sizes = []
    for chart in charts:
        try:
            size = len(pio.to_json(chart).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise ValueError("A saved chart is invalid.") from exc
        if size > resources.MAX_CHART_BYTES:
            raise ValueError("This chart exceeds the saved-chart size limit; plot fewer rows or categories.")
        sizes.append(size)
    if sum(sizes) > resources.MAX_CHART_TOTAL_BYTES:
        raise ValueError("Saved charts exceed their total size limit; clear saved charts before adding more.")
    return sizes


def _validate_experiment_sizes(experiments):
    sizes = [resources.estimate_bytes(experiment) for experiment in experiments]
    if any(size > resources.MAX_EXPERIMENT_BYTES for size in sizes):
        raise ValueError("An experiment exceeds the retained experiment size limit; select fewer models or use less data.")
    if sum(sizes) > resources.MAX_EXPERIMENT_TOTAL_BYTES:
        raise ValueError("Saved experiments exceed their total memory limit; remove older experiments.")
    return sizes


class StateManager:
    """Own dataset mutations, synchronized schema, undo, and dependent state."""

    @staticmethod
    def init_session() -> None:
        # A widget cannot be reset after it has been instantiated in the same run.
        # Operations schedule only existing widget keys for the next rerun.
        pending = st.session_state.get("_dataset_widgets_pending_reset", [])
        for key in pending:
            st.session_state.pop(key, None)
        st.session_state["_dataset_widgets_pending_reset"] = []
        if "dataset_revision" not in st.session_state:
            st.session_state["dataset_revision"] = 0
        if "project_revision" not in st.session_state:
            st.session_state["project_revision"] = 0
        if "statistical_operations" not in st.session_state:
            # Older sessions did not record cleaning parameters; be conservative.
            st.session_state["statistical_operations"] = list(dict.fromkeys(
                item["name"] for item in st.session_state.get("history", [])
                if item.get("name") in ("Handle missing values", "Handle outliers")
            ))
        if "operation_log" not in st.session_state and StateManager.has_dataset():
            st.session_state.operation_log = [{
                "id": uuid4().hex, "name": "Earlier session changes", "kind": "legacy",
                "parameters": {}, "replayable": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]
        for key, value in _defaults().items():
            if key not in st.session_state:
                st.session_state[key] = value
        if StateManager.has_dataset():
            if st.session_state.dataset_id is None:
                st.session_state.dataset_id = uuid4().hex
            if st.session_state.data_version is None:
                st.session_state.data_version = uuid4().hex
            if not st.session_state.original_column_meta and st.session_state.original_df is not None:
                _, st.session_state.original_column_meta = _prepare_dataset(st.session_state.original_df, {})
            for snapshot in st.session_state.history:
                if isinstance(snapshot, dict) and "df" in snapshot and "column_meta" in snapshot:
                    snapshot.setdefault("schema_confirmed", False)
                    snapshot.setdefault("target_column", None)
                    snapshot.setdefault("statistical_operations", list(st.session_state.statistical_operations))
                    snapshot.setdefault("operation_count", len(st.session_state.operation_log))
                    snapshot.setdefault("data_version", uuid4().hex)

    @staticmethod
    def _invalidate_dependent_state() -> None:
        pending = set(st.session_state.get("_dataset_widgets_pending_reset", []))
        pending.update(
            key for key in st.session_state
            if isinstance(key, str) and (key in _WIDGET_KEYS or key.startswith(_WIDGET_PREFIXES))
        )
        st.session_state["_dataset_widgets_pending_reset"] = sorted(pending)
        st.session_state.viz_charts = []
        st.session_state.cleaning_previews = {}
        st.session_state["_analysis_cache"] = {}
        st.session_state["_visualization_cache"] = {}
        st.session_state["_upload_summary_cache"] = {}
        st.session_state.dataset_revision = st.session_state.get("dataset_revision", 0) + 1
        StateManager._touch_project()

    @staticmethod
    def _touch_project() -> None:
        st.session_state.project_revision = st.session_state.get("project_revision", 0) + 1
        st.session_state.pop("prepared_project_download", None)

    @staticmethod
    def load_dataset(df: pd.DataFrame, column_meta: dict) -> None:
        candidate, metadata = _prepare_dataset(df, column_meta, explicit_schema=True)
        original = candidate.copy(deep=True)
        StateManager._invalidate_dependent_state()
        st.session_state.original_df = original
        st.session_state.original_column_meta = copy.deepcopy(metadata)
        st.session_state.df = candidate
        st.session_state.column_meta = metadata
        st.session_state.history = []
        st.session_state.schema_confirmed = False
        st.session_state.target_column = None
        st.session_state.statistical_operations = []
        st.session_state.operation_log = []
        st.session_state.experiments = []
        st.session_state.dataset_id = uuid4().hex
        st.session_state.data_version = uuid4().hex
        st.session_state.project_name = "DataWorks project"

    @staticmethod
    def _snapshot(operation_name: str) -> dict:
        if not StateManager.has_dataset():
            raise ValueError("Load a dataset before recording an operation.")
        return {
            "name": operation_name,
            "df": st.session_state.df.copy(deep=True),
            "column_meta": copy.deepcopy(st.session_state.column_meta),
            "schema_confirmed": st.session_state.schema_confirmed,
            "target_column": st.session_state.target_column,
            "statistical_operations": list(st.session_state.statistical_operations),
            "operation_count": len(st.session_state.operation_log),
            "data_version": st.session_state.data_version,
        }

    @staticmethod
    def apply_operation(
        operation_name: str,
        df: pd.DataFrame,
        *,
        column_meta: dict | None = None,
        schema_confirmed: bool | None = None,
        learns_from_data: bool = False,
        operation: dict | None = None,
    ) -> bool:
        """Atomically apply a prepared operation; return False for a true no-op.

        The caller calculates transformations first. Validation and copying finish
        before history, the current dataset, or dependent outputs are changed.
        """
        if not StateManager.has_dataset():
            raise ValueError("Load a dataset before applying an operation.")
        if schema_confirmed is not None and not isinstance(schema_confirmed, bool):
            raise ValueError("Schema confirmation must be a boolean.")
        if not isinstance(learns_from_data, bool):
            raise ValueError("The statistical-cleaning flag must be a boolean.")
        descriptor = validate_operation(operation if operation is not None else {
            "kind": "custom", "parameters": {}, "replayable": False,
        })
        metadata = st.session_state.column_meta if column_meta is None else column_meta
        candidate, metadata = _prepare_dataset(
            df, metadata, previous_df=st.session_state.df,
            explicit_schema=column_meta is not None,
        )
        confirmed = st.session_state.schema_confirmed if schema_confirmed is None else schema_confirmed
        target = st.session_state.target_column
        if target not in candidate.columns:
            target = None
        if (
            candidate.equals(st.session_state.df)
            and metadata == st.session_state.column_meta
            and confirmed == st.session_state.schema_confirmed
            and target == st.session_state.target_column
        ):
            return False
        snapshot = StateManager._snapshot(operation_name)
        history = _bounded_history([*st.session_state.history, snapshot])
        statistical = list(st.session_state.statistical_operations)
        if learns_from_data and operation_name not in statistical:
            statistical.append(operation_name)
        version = uuid4().hex
        record = {
            **descriptor, "id": uuid4().hex, "name": operation_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_version": st.session_state.data_version, "result_version": version,
            "learns_from_data": learns_from_data,
            "effects": {
                "rows_before": len(st.session_state.df), "rows_after": len(candidate),
                "columns_before": len(st.session_state.df.columns), "columns_after": len(candidate.columns),
            },
        }
        operations = [*st.session_state.operation_log, record]
        StateManager._invalidate_dependent_state()
        st.session_state.df = candidate
        st.session_state.column_meta = metadata
        st.session_state.schema_confirmed = confirmed
        st.session_state.target_column = target
        st.session_state.history = history
        st.session_state.statistical_operations = statistical
        st.session_state.operation_log = operations
        st.session_state.data_version = version
        return True

    @staticmethod
    def save_snapshot(operation_name: str) -> None:
        """Retain explicit checkpoints; normal mutations use apply_operation."""
        snapshot = StateManager._snapshot(operation_name)
        st.session_state.history = _bounded_history([*st.session_state.history, snapshot])
        StateManager._touch_project()

    @staticmethod
    def undo() -> bool:
        history = st.session_state.history
        if not history:
            return False
        snapshot = history[-1]
        candidate, metadata = _prepare_dataset(snapshot["df"], snapshot["column_meta"], explicit_schema=True)
        target = snapshot.get("target_column")
        if target not in candidate.columns:
            target = None
        confirmed = snapshot.get("schema_confirmed", False)
        StateManager._invalidate_dependent_state()
        st.session_state.df = candidate
        st.session_state.column_meta = metadata
        st.session_state.schema_confirmed = confirmed
        st.session_state.target_column = target
        st.session_state.history = history[:-1]
        st.session_state.statistical_operations = list(snapshot.get(
            "statistical_operations", st.session_state.statistical_operations,
        ))
        if "operation_count" in snapshot:
            st.session_state.operation_log = st.session_state.operation_log[:snapshot["operation_count"]]
        else:
            st.session_state.operation_log = [{
                "id": uuid4().hex, "name": "Restored earlier session state", "kind": "legacy",
                "parameters": {}, "replayable": False,
            }]
        st.session_state.data_version = snapshot.get("data_version") or uuid4().hex
        return True

    @staticmethod
    def get_column_meta(col: str) -> dict:
        return copy.deepcopy(st.session_state.column_meta.get(col, _empty_column_meta()))

    @staticmethod
    def update_column_meta(col: str, **kwargs: Any) -> None:
        if not StateManager.has_dataset() or col not in st.session_state.df.columns:
            raise ValueError(f"Column {col!r} is not present in the dataset.")
        metadata = copy.deepcopy(st.session_state.column_meta)
        metadata[col].update(kwargs)
        StateManager.apply_operation(
            f"Update schema: {col}", st.session_state.df, column_meta=metadata,
            operation={"kind": "schema", "parameters": {
                "column_meta": metadata, "schema_confirmed": st.session_state.schema_confirmed,
                "apply_types": False,
            }},
        )

    @staticmethod
    def get_columns_by_role(role: str) -> list[str]:
        if not StateManager.has_dataset():
            return []
        return [
            c for c, meta in st.session_state.column_meta.items()
            if c in st.session_state.df.columns and meta.get("role") == role
        ]

    @staticmethod
    def set_target(col) -> None:
        if col is not None and (
            not StateManager.has_dataset() or col not in st.session_state.df.columns
            or st.session_state.column_meta[col].get("role") in ("id", "datetime", "ignore")
        ):
            raise ValueError("Choose a target column with the feature role.")
        if st.session_state.target_column != col:
            st.session_state.target_column = col
            StateManager._touch_project()

    @staticmethod
    def get_numeric_columns() -> list[str]:
        df = st.session_state.get("df")
        return [] if df is None else df.select_dtypes(include="number").columns.tolist()

    @staticmethod
    def get_categorical_columns() -> list[str]:
        df = st.session_state.get("df")
        return [] if df is None else df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    @staticmethod
    def has_dataset() -> bool:
        return st.session_state.get("df") is not None

    @staticmethod
    def clear() -> None:
        StateManager._invalidate_dependent_state()
        for key, value in _defaults().items():
            st.session_state[key] = value

    @staticmethod
    def clear_dataset() -> None:
        """Clear the generic dataset without discarding fulfillment project work."""
        StateManager._invalidate_dependent_state()
        for key, value in _defaults().items():
            if key not in ("fulfillment", "project_name"):
                st.session_state[key] = value

    @staticmethod
    def _set_fulfillment(candidate: dict) -> bool:
        if candidate == st.session_state.fulfillment:
            return False
        st.session_state.fulfillment = candidate
        StateManager._touch_project()
        return True

    @staticmethod
    def set_fulfillment_source(source: dict) -> bool:
        from backend.fulfillment_state import set_source

        candidate = set_source(st.session_state.fulfillment, source)
        return StateManager._set_fulfillment(candidate)

    @staticmethod
    def set_fulfillment_plan(plan: dict) -> bool:
        from backend.fulfillment_state import set_plan

        candidate = set_plan(st.session_state.fulfillment, plan)
        return StateManager._set_fulfillment(candidate)

    @staticmethod
    def run_fulfillment() -> dict:
        from backend.fulfillment_state import run_project

        candidate = run_project(st.session_state.fulfillment)
        attempt = copy.deepcopy(candidate["last_attempt"])
        StateManager._set_fulfillment(candidate)
        return attempt

    @staticmethod
    def undo_fulfillment() -> bool:
        from backend.fulfillment_state import undo_project

        candidate = undo_project(st.session_state.fulfillment)
        return StateManager._set_fulfillment(candidate)

    @staticmethod
    def fulfillment_result_is_current() -> bool:
        from backend.fulfillment_state import current_result

        return current_result(st.session_state.fulfillment)

    @staticmethod
    def add_experiment(experiment: dict) -> str:
        if not StateManager.has_dataset() or not isinstance(experiment, dict):
            raise ValueError("Load a dataset before saving a completed experiment.")
        if experiment.get("dataset_id") != st.session_state.dataset_id or experiment.get("data_version") != st.session_state.data_version:
            raise ValueError("The dataset changed while training; this experiment cannot be saved as a current run.")
        if not isinstance(experiment.get("config"), dict) or not isinstance(experiment.get("results"), dict) or not experiment["results"]:
            raise ValueError("A completed experiment needs settings and at least one successful model result.")
        new_size = resources.estimate_bytes(experiment)
        if new_size > min(resources.MAX_EXPERIMENT_BYTES, resources.MAX_EXPERIMENT_TOTAL_BYTES):
            raise ValueError("This experiment exceeds the retained experiment size limit; select fewer models or use less data.")
        identifier = experiment.get("id")
        if identifier is not None and any(item["id"] == identifier for item in st.session_state.experiments):
            raise ValueError("An experiment with that identifier already exists.")
        owned = copy.deepcopy(experiment)
        owned.setdefault("id", uuid4().hex)
        owned.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        if any(item["id"] == owned["id"] for item in st.session_state.experiments):
            raise ValueError("An experiment with that identifier already exists.")
        retained = [*st.session_state.experiments, owned][-MAX_EXPERIMENTS:]
        sizes = [resources.estimate_bytes(item) for item in retained]
        total = sum(sizes)
        while len(retained) > 1 and total > resources.MAX_EXPERIMENT_TOTAL_BYTES:
            total -= sizes.pop(0)
            retained.pop(0)
        st.session_state.experiments = retained
        StateManager._touch_project()
        return owned["id"]

    @staticmethod
    def remove_experiment(identifier: str) -> bool:
        previous = st.session_state.experiments
        remaining = [item for item in previous if item["id"] != identifier]
        st.session_state.experiments = remaining
        if len(remaining) != len(previous):
            StateManager._touch_project()
        return len(remaining) != len(previous)

    @staticmethod
    def set_project_name(name: str) -> None:
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("The project name must contain 1–200 characters.")
        if name != st.session_state.project_name:
            st.session_state.project_name = name
            StateManager._touch_project()

    @staticmethod
    def save_chart(figure) -> None:
        _chart_sizes([*st.session_state.viz_charts, figure])
        st.session_state.viz_charts = [*st.session_state.viz_charts, copy.deepcopy(figure)]
        StateManager._touch_project()

    @staticmethod
    def clear_charts() -> None:
        if st.session_state.viz_charts:
            st.session_state.viz_charts = []
            StateManager._touch_project()

    @staticmethod
    def resource_summary() -> dict:
        """Approximate retained state sizes; temporary transform allocations are separate."""
        return {
            "working_data_bytes": resources.frame_bytes(st.session_state.df) if StateManager.has_dataset() else 0,
            "original_data_bytes": resources.frame_bytes(st.session_state.original_df) if st.session_state.original_df is not None else 0,
            "undo_snapshots": len(st.session_state.history),
            "undo_bytes": sum(resources.estimate_bytes(snapshot) for snapshot in st.session_state.history),
            "saved_charts": len(st.session_state.viz_charts),
            "saved_chart_json_bytes": sum(_chart_sizes(st.session_state.viz_charts)),
            "experiments": len(st.session_state.experiments),
            "experiment_bytes": sum(resources.estimate_bytes(experiment) for experiment in st.session_state.experiments),
            "fulfillment_bytes": resources.estimate_bytes(st.session_state.fulfillment),
            "fulfillment_undo_snapshots": len(st.session_state.fulfillment["history"]),
        }

    @staticmethod
    def export_project(include_history: bool = True) -> dict:
        from backend.fulfillment_state import export_project as export_fulfillment

        fulfillment = export_fulfillment(st.session_state.fulfillment, include_history=include_history)
        if not StateManager.has_dataset():
            if fulfillment["source"] is None and fulfillment["committed"] is None:
                raise ValueError("Load a dataset or fulfillment source before saving a project.")
            return {"mode": "fulfillment-only", "project_name": st.session_state.project_name,
                    "fulfillment": fulfillment}
        from backend.experiments import project_experiments
        import plotly.io as pio
        import json

        result = {
            key: copy.deepcopy(st.session_state[key])
            for key in (
                "df", "original_df", "column_meta", "original_column_meta", "schema_confirmed",
                "target_column", "statistical_operations", "operation_log", "dataset_id", "data_version", "project_name",
            )
        }
        result["history"] = copy.deepcopy(st.session_state.history) if include_history else []
        result["experiments"] = project_experiments(st.session_state.experiments)
        result["viz_charts"] = [json.loads(pio.to_json(figure)) for figure in st.session_state.viz_charts]
        result["fulfillment"] = fulfillment
        return result

    @staticmethod
    def restore_project(project: dict) -> None:
        """Validate every restored component before replacing the active project."""
        from backend.experiments import project_experiments
        from backend.fulfillment_state import new_project, validate_project as validate_fulfillment
        import plotly.graph_objects as go

        if not isinstance(project, dict):
            raise ValueError("A project must contain a structured state object.")
        if project.get("mode") == "fulfillment-only":
            if set(project) != {"mode", "project_name", "fulfillment"}:
                raise ValueError("The fulfillment-only project has unsupported fields.")
            name = project["project_name"]
            if not isinstance(name, str) or not name.strip() or len(name) > 200:
                raise ValueError("The project name must contain 1–200 characters.")
            fulfillment = validate_fulfillment(project["fulfillment"])
            if fulfillment["source"] is None and fulfillment["committed"] is None:
                raise ValueError("The fulfillment-only project contains no source or result.")
            candidate = _defaults()
            candidate.update(fulfillment=fulfillment, project_name=name)
            StateManager._invalidate_dependent_state()
            st.session_state.update(candidate)
            return
        if "mode" in project:
            raise ValueError("The project payload mode is unsupported.")
        fulfillment = (validate_fulfillment(project["fulfillment"])
                       if "fulfillment" in project else new_project())
        required = {
            "df", "original_df", "column_meta", "original_column_meta", "history", "operation_log",
            "schema_confirmed", "target_column", "statistical_operations", "dataset_id", "data_version",
        }
        if not required.issubset(project):
            raise ValueError("The project is missing required dataset or history fields.")
        for frame_key, meta_key in (("df", "column_meta"), ("original_df", "original_column_meta")):
            frame, metadata = project[frame_key], project[meta_key]
            if not isinstance(frame, pd.DataFrame) or not isinstance(metadata, dict) or set(metadata) != set(frame.columns):
                raise ValueError("Project column metadata must match its dataset columns.")
        current, metadata = _prepare_dataset(project["df"], project["column_meta"], explicit_schema=True)
        original, original_meta = _prepare_dataset(project["original_df"], project["original_column_meta"], explicit_schema=True)
        if not isinstance(project["schema_confirmed"], bool):
            raise ValueError("Project schema confirmation must be a boolean.")
        target = project["target_column"]
        if target is not None and target not in current.columns:
            raise ValueError("The saved target is not in the working dataset.")
        dataset_id = _identifier(project["dataset_id"], "dataset_id")
        version = _identifier(project["data_version"], "data_version")
        statistical = _statistical_names(project["statistical_operations"])
        records = project["operation_log"]
        if not isinstance(records, list):
            raise ValueError("Project operations must be a list.")
        operations = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("name"), str):
                raise ValueError("Each project operation needs a name and structured parameters.")
            effects = record.get("effects", {})
            if not isinstance(effects, dict) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in effects.values()
            ):
                raise ValueError("Operation effects must contain non-negative row and column counts.")
            normalized = validate_operation(record)
            operations.append({**copy.deepcopy(record), **normalized})
        saved_history = project["history"]
        if not isinstance(saved_history, list) or len(saved_history) > MAX_HISTORY:
            raise ValueError(f"A project can contain at most {MAX_HISTORY} undo snapshots.")
        if sum(resources.estimate_bytes(snapshot) for snapshot in saved_history) > resources.MAX_HISTORY_BYTES:
            raise ValueError("Project undo snapshots exceed the memory limit; save a copy without undo snapshots.")
        history = []
        for snapshot in saved_history:
            if not isinstance(snapshot, dict) or not isinstance(snapshot.get("name"), str):
                raise ValueError("Each undo snapshot needs an operation name.")
            frame, meta = _prepare_dataset(snapshot.get("df"), snapshot.get("column_meta"), explicit_schema=True)
            confirmed, saved_target = snapshot.get("schema_confirmed"), snapshot.get("target_column")
            count = snapshot.get("operation_count")
            if not isinstance(confirmed, bool) or (saved_target is not None and saved_target not in frame.columns):
                raise ValueError("An undo snapshot has invalid schema or target state.")
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(operations):
                raise ValueError("An undo snapshot refers to invalid operation history.")
            history.append({
                "name": snapshot["name"], "df": frame, "column_meta": meta,
                "schema_confirmed": confirmed, "target_column": saved_target,
                "statistical_operations": _statistical_names(snapshot.get("statistical_operations", [])),
                "operation_count": count, "data_version": _identifier(snapshot.get("data_version"), "snapshot data_version"),
            })
        experiments = project.get("experiments", [])
        if not isinstance(experiments, list) or len(experiments) > MAX_EXPERIMENTS:
            raise ValueError(f"A project can contain at most {MAX_EXPERIMENTS} experiment summaries.")
        for experiment in experiments:
            if not isinstance(experiment, dict) or experiment.get("dataset_id") != dataset_id:
                raise ValueError("An experiment does not belong to this project dataset.")
            _identifier(experiment.get("data_version"), "experiment data_version")
            if not isinstance(experiment.get("config"), dict) or not isinstance(experiment.get("results"), dict) or not experiment["results"]:
                raise ValueError("A saved experiment needs settings and model results.")
        summaries = project_experiments(experiments)
        if len({experiment["id"] for experiment in summaries}) != len(summaries):
            raise ValueError("Saved experiment identifiers must be unique.")
        _validate_experiment_sizes(summaries)
        charts = project.get("viz_charts", [])
        if not isinstance(charts, list):
            raise ValueError("Saved charts must be a list.")
        _chart_sizes(charts)
        try:
            figures = [go.Figure(chart) for chart in charts]
        except (TypeError, ValueError) as exc:
            raise ValueError("A saved chart is invalid.") from exc
        name = project.get("project_name", "DataWorks project")
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("The project name must contain 1–200 characters.")
        StateManager._invalidate_dependent_state()
        st.session_state.update({
            "df": current, "original_df": original, "column_meta": metadata,
            "original_column_meta": original_meta, "schema_confirmed": project["schema_confirmed"],
            "target_column": target, "statistical_operations": statistical, "operation_log": operations,
            "history": history, "dataset_id": dataset_id, "data_version": version,
            "experiments": summaries, "viz_charts": figures, "project_name": name,
            "fulfillment": fulfillment,
        })

