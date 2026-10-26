"""Completed ML reports and explicit exports; model binaries are never imported."""

import copy
from datetime import datetime, timezone
from io import BytesIO
import json
import platform
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn

from backend.exports import export_csv
from backend.ingestion import apply_inferred_dtypes
from backend.ml import plot_feature_importance


MAX_EXPERIMENTS = 5
_REPORT_FIELDS = (
    "id", "created_at", "dataset_id", "data_version", "config", "results", "errors",
    "y_test", "test_index", "feature_names", "feature_columns", "preprocessing_info",
    "schema", "importance_figures", "plot_errors", "package_versions",
)


def _report_value(value):
    """Copy report values without executable Python objects or NumPy arrays."""
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _report_value(value.item())
    if isinstance(value, np.ndarray):
        return _report_value(value.tolist())
    if isinstance(value, pd.Series):
        return [_report_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {copy.deepcopy(key): _report_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_report_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_report_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, pd.Timestamp, pd.Timedelta)):
        return copy.deepcopy(value)
    raise ValueError(f"Unsupported value in an experiment report: {type(value).__name__}.")


def package_versions() -> dict:
    return {
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
    }


def create_experiment(
    *, dataset_id, data_version, config, results, errors, y_test,
    feature_names, feature_columns, info, schema,
) -> dict:
    """Own one completed run without retaining full training/test feature tables."""
    if not results:
        raise ValueError("An experiment needs at least one successfully evaluated model.")
    models, reports, figures, plot_errors = {}, {}, {}, {}
    for name, metrics in results.items():
        model = metrics.get("_model")
        if model is None:
            raise ValueError(f"Model {name!r} has no fitted estimator.")
        models[name] = model
        reports[name] = _report_value({key: value for key, value in metrics.items() if key != "_model"})
        try:
            figure = plot_feature_importance(model, feature_names)
            if figure is not None:
                figures[name] = json.loads(figure.to_json())
        except Exception as exc:
            plot_errors[name] = str(exc)
    preprocessor = info.get("preprocessor")
    if preprocessor is None:
        raise ValueError("The fitted preprocessor is required to retain a trained experiment.")
    return {
        "id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id, "data_version": data_version,
        "config": copy.deepcopy(config), "results": reports, "errors": dict(errors),
        "y_test": _report_value(y_test), "test_index": _report_value(list(y_test.index)),
        "feature_names": list(feature_names), "feature_columns": list(feature_columns),
        "preprocessing_info": _report_value({key: value for key, value in info.items() if key != "preprocessor"}),
        "schema": copy.deepcopy(schema), "preprocessor": preprocessor, "models": models,
        "importance_figures": figures, "plot_errors": plot_errors,
        "package_versions": package_versions(),
    }


def project_experiments(experiments: list) -> list[dict]:
    """Produce report-only project records; never serialize live model objects."""
    reports = []
    for experiment in experiments[-MAX_EXPERIMENTS:]:
        report = {
            field: _report_value(experiment[field])
            for field in _REPORT_FIELDS if field in experiment
        }
        validate_experiment_report(report)
        reports.append(report)
    return reports


def _finite_metric(value, name, *, nullable=False, score=False):
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"Experiment metric {name!r} must be a finite number.")
    if score and not 0 <= value <= 1:
        raise ValueError(f"Experiment metric {name!r} must be between zero and one.")


def validate_experiment_report(report: dict) -> None:
    """Reject incomplete project reports before they can replace session state."""
    if not isinstance(report, dict):
        raise ValueError("An experiment report must be a dictionary.")
    for key in ("id", "dataset_id", "data_version", "created_at"):
        if not isinstance(report.get(key), str) or not report[key].strip() or len(report[key]) > 200:
            raise ValueError(f"Experiment {key!r} must be a non-empty string of at most 200 characters.")
    try:
        timestamp = datetime.fromisoformat(report["created_at"])
        if timestamp.tzinfo is None:
            raise ValueError("Timezone missing.")
    except ValueError as exc:
        raise ValueError("An experiment creation time must be an ISO datetime with a timezone.") from exc
    config = report.get("config")
    if not isinstance(config, dict) or config.get("problem_type") not in ("classification", "regression"):
        raise ValueError("An experiment must identify classification or regression settings.")
    if config.get("source") not in ("working", "original") or config.get("split_strategy") not in (
        "random", "stratified", "group", "chronological",
    ):
        raise ValueError("An experiment has an unsupported data source or split strategy.")
    fraction = config.get("test_size")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("An experiment test fraction must be between zero and one.")
    if "target" not in config or not isinstance(config.get("features"), list) or not config["features"]:
        raise ValueError("An experiment must retain its target and selected feature list.")
    for field in ("feature_names", "feature_columns"):
        if not isinstance(report.get(field), list) or not report[field]:
            raise ValueError(f"An experiment must retain a non-empty {field} list.")
    if any(not isinstance(name, str) for name in report["feature_names"]):
        raise ValueError("Encoded experiment feature names must be strings.")
    if not isinstance(report.get("schema"), dict) or not isinstance(report.get("preprocessing_info"), dict):
        raise ValueError("An experiment must retain its schema and preprocessing details.")
    actual = report.get("y_test")
    if not isinstance(actual, list) or not actual or any(not pd.api.types.is_scalar(value) for value in actual):
        raise ValueError("An experiment must retain a non-empty scalar test-target list.")
    if any(pd.isna(value) for value in actual):
        raise ValueError("Stored experiment targets cannot contain missing values.")
    if not isinstance(report.get("test_index"), list) or len(report["test_index"]) != len(actual):
        raise ValueError("An experiment must retain a row label for every test target.")
    results = report.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("An experiment must retain at least one model result.")
    for name, metrics in results.items():
        if not isinstance(name, str) or not name or len(name) > 200 or not isinstance(metrics, dict):
            raise ValueError("Experiment model results must map model names to metrics.")
        predictions = metrics.get("y_pred")
        if not isinstance(predictions, list) or len(predictions) != len(actual) or any(
            not pd.api.types.is_scalar(value) or pd.isna(value) for value in predictions
        ):
            raise ValueError("Stored predictions must match the non-missing test-target rows.")
        if config["problem_type"] == "classification":
            for key in ("accuracy", "macro_f1", "precision", "recall", "f1"):
                _finite_metric(metrics.get(key), key, score=True)
            labels = metrics.get("class_labels")
            if not isinstance(labels, list) or not labels or any(not pd.api.types.is_scalar(value) for value in labels):
                raise ValueError("Classification results must retain actual class labels.")
            try:
                matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
                valid = matrix.shape == (len(labels), len(labels)) and np.isfinite(matrix).all()
                valid = valid and bool((matrix >= 0).all()) and bool((matrix == np.floor(matrix)).all())
                valid = valid and int(matrix.sum()) == len(actual)
            except (KeyError, TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError("The stored confusion matrix must match class labels and test-row counts.")
            classes = metrics.get("per_class")
            if not isinstance(classes, list) or len(classes) != len(labels):
                raise ValueError("Classification results must retain per-class metrics.")
            for item in classes:
                if not isinstance(item, dict) or "class" not in item:
                    raise ValueError("Stored class reports must identify their class.")
                for key in ("precision", "recall", "f1"):
                    _finite_metric(item.get(key), key, score=True)
                support = item.get("support")
                if isinstance(support, bool) or not isinstance(support, int) or support < 0:
                    raise ValueError("Stored class support must be a non-negative integer.")
        else:
            for key in ("MAE", "MSE", "RMSE"):
                _finite_metric(metrics.get(key), key)
                if metrics[key] < 0:
                    raise ValueError("Stored regression errors cannot be negative.")
            _finite_metric(metrics.get("R²"), "R²", nullable=True)
            try:
                if not np.isfinite(np.asarray(actual + predictions, dtype=float)).all():
                    raise ValueError("Non-finite values.")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Stored regression targets and predictions must be finite numbers.") from exc
    for field in ("errors", "plot_errors"):
        values = report.get(field, {})
        if not isinstance(values, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in values.items()):
            raise ValueError("Stored experiment errors must be text keyed by model name.")
    figures = report.get("importance_figures", {})
    if not isinstance(figures, dict):
        raise ValueError("Stored importance figures must be a dictionary.")
    try:
        for name, figure in figures.items():
            if name not in results or not isinstance(figure, dict):
                raise ValueError("Invalid importance figure.")
            go.Figure(figure)
    except (TypeError, ValueError) as exc:
        raise ValueError("A stored feature-importance figure is invalid.") from exc


def experiment_summary(experiment: dict) -> dict:
    """Return the portable report without runtime estimators or preprocessors."""
    return project_experiments([experiment])[0]


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else {"type": "float", "value": str(value)}
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return {"type": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_json_value(item) for item in value]}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _json_value(item) for key, item in value.items()}
        return {"type": "mapping", "items": [[_json_value(key), _json_value(item)] for key, item in value.items()]}
    raise ValueError(f"Unsupported report value: {type(value).__name__}.")


def export_experiment_json(experiment: dict) -> bytes:
    report = {"format": "dataworks-experiment-report", "version": 1, "experiment": experiment_summary(experiment)}
    return json.dumps(_json_value(report), ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def experiment_comparison(experiment: dict) -> pd.DataFrame:
    problem_type = experiment["config"]["problem_type"]
    columns = (
        ["accuracy", "macro_f1", "precision", "recall", "f1"]
        if problem_type == "classification" else ["MAE", "MSE", "RMSE", "R²"]
    )
    comparison = pd.DataFrame([
        {"Model": name, **{column: metrics.get(column) for column in columns}}
        for name, metrics in experiment["results"].items()
    ]).set_index("Model")
    return comparison.rename(columns={
        "precision": "weighted_precision", "recall": "weighted_recall", "f1": "weighted_f1",
    })


def export_experiment_csv(experiment: dict) -> bytes:
    table = experiment_comparison(experiment).reset_index()
    table.insert(0, "experiment_id", experiment["id"])
    table.insert(1, "created_at_utc", experiment["created_at"])
    return export_csv(table, spreadsheet_safe=True, include_index=False)


def export_model_artifact(experiment: dict, model_name: str) -> bytes:
    """Export a trusted Python/joblib bundle; this module deliberately has no loader."""
    model = experiment.get("models", {}).get(model_name)
    preprocessor = experiment.get("preprocessor")
    if model is None or preprocessor is None:
        raise ValueError("This report has no live fitted model. Retrain it to export a Python model artifact.")
    bundle = {
        "format": "dataworks-model", "version": 1,
        "experiment_id": experiment["id"], "model_name": model_name,
        "dataset_id": experiment["dataset_id"], "data_version": experiment["data_version"],
        "config": copy.deepcopy(experiment["config"]), "schema": copy.deepcopy(experiment["schema"]),
        "feature_names": list(experiment["feature_names"]),
        "feature_columns": list(experiment["feature_columns"]),
        "package_versions": dict(experiment["package_versions"]),
        "preprocessor": preprocessor, "model": model,
    }
    stream = BytesIO()
    joblib.dump(bundle, stream, compress=3)
    return stream.getvalue()


def predict_artifact(artifact: dict, frame: pd.DataFrame):
    """Predict with an already loaded, trusted artifact and the stored schema.

    Loading joblib/pickle files can execute code. Callers must load only their own
    trusted exports in a compatible DataWorks installation; the app never loads them.
    """
    if artifact.get("format") != "dataworks-model" or artifact.get("version") != 1:
        raise ValueError("Unsupported DataWorks model artifact.")
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Prediction input must be a DataFrame with unique columns.")
    missing = [column for column in artifact["feature_columns"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required features: {missing}.")
    values = apply_inferred_dtypes(frame.loc[:, artifact["feature_columns"]], artifact["schema"])
    transformed = artifact["preprocessor"].transform(values)
    prepared = pd.DataFrame(transformed, index=frame.index, columns=artifact["feature_names"])
    return artifact["model"].predict(prepared)


def importance_figure(experiment: dict, model_name: str):
    figure = experiment.get("importance_figures", {}).get(model_name)
    return None if figure is None else go.Figure(figure)
