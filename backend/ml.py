import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numbers import Real
from scipy import sparse
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_fscore_support,
    r2_score,
    recall_score,
)
from sklearn.utils.multiclass import type_of_target


# ── Problem type detection ───────────────────────────────────────────────────

def detect_problem_type(y: pd.Series) -> str:
    """Suggest a task; categorical types stay discrete and fractional numbers stay continuous."""
    if pd.api.types.is_bool_dtype(y.dtype) or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    observed = y.dropna()
    numeric = pd.api.types.is_numeric_dtype(y.dtype) or (
        not observed.empty and all(isinstance(value, Real) for value in observed)
    )
    if numeric:
        values = observed.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if np.any(finite != np.floor(finite)):
            return "regression"
        return "classification" if y.nunique() <= 20 else "regression"
    return "classification"


# ── Model registry ───────────────────────────────────────────────────────────

_CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
}

_REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
}


def get_available_models(problem_type: str) -> dict:
    """Return fresh estimators; registry templates are never fitted."""
    _validate_problem_type(problem_type)
    registry = _CLASSIFIERS if problem_type == "classification" else _REGRESSORS
    return {name: clone(model) for name, model in registry.items()}


# ── Training & evaluation ────────────────────────────────────────────────────

def _validate_problem_type(problem_type: str) -> None:
    if problem_type not in ("classification", "regression"):
        raise ValueError("Choose classification or regression as the problem type.")


def _target_series(values, name: str) -> pd.Series:
    if np.ndim(values) != 1:
        raise ValueError(f"{name} must be a one-dimensional target.")
    series = pd.Series(values)
    if series.empty:
        raise ValueError(f"{name} must contain at least one row.")
    if series.isna().any():
        raise ValueError(f"{name} contains missing values; exclude rows with missing targets.")
    for value in series:
        if isinstance(value, complex):
            raise ValueError(f"{name} must not contain complex numbers.")
        if isinstance(value, Real) and not np.isfinite(value):
            raise ValueError(f"{name} must contain only finite numeric values.")
    return series


def _validate_feature_rows(features, target, name: str) -> None:
    shape = getattr(features, "shape", None)
    if shape is None:
        shape = np.shape(features)
    if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
        raise ValueError(f"{name} features must be a non-empty two-dimensional table.")
    if shape[0] != len(target):
        raise ValueError(f"{name} features and targets must contain the same number of rows.")


def _regression_target(values: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(values.dtype) or pd.api.types.is_timedelta64_dtype(values.dtype):
        raise ValueError(f"{name} must contain numeric values for regression, not dates or durations.")
    try:
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values for regression.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite numeric values.")
    return numeric


def train_model(model, X_train, y_train):
    target = _target_series(y_train, "Training target")
    _validate_feature_rows(X_train, target, "Training")
    if is_classifier(model):
        if type_of_target(target) not in ("binary", "multiclass"):
            raise ValueError("Classification requires discrete class labels; choose regression for continuous values.")
        if target.nunique() < 2:
            raise ValueError("Training requires at least two classes in the training split.")
    elif is_regressor(model):
        _regression_target(target, "Training target")
    if isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)) and len(target) < model.n_neighbors:
        raise ValueError(f"KNN needs at least {model.n_neighbors} training rows; increase the training split.")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, problem_type: str) -> dict:
    _validate_problem_type(problem_type)
    actual = _target_series(y_test, "Test target")
    _validate_feature_rows(X_test, actual, "Test")
    if problem_type == "classification" and type_of_target(actual) not in ("binary", "multiclass"):
        raise ValueError("Classification evaluation requires discrete class labels.")
    actual_values = _regression_target(actual, "Test target") if problem_type == "regression" else None
    y_pred = model.predict(X_test)
    predicted = _target_series(y_pred, "Predictions")
    if len(predicted) != len(actual):
        raise ValueError("Predictions and test targets must contain the same number of rows.")
    if problem_type == "classification":
        if type_of_target(predicted) not in ("binary", "multiclass"):
            raise ValueError("Classification predictions must contain discrete class labels.")
        labels = list(getattr(model, "classes_", []))
        for value in [*pd.unique(actual), *pd.unique(predicted)]:
            if value not in labels:
                labels.append(value)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, zero_division=0,
        )
        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "class_labels": labels,
            "per_class": [
                {"class": label, "precision": round(float(p), 4), "recall": round(float(r), 4),
                 "f1": round(float(f), 4), "support": int(n)}
                for label, p, r, f, n in zip(labels, precision, recall, f1, support)
            ],
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
            "y_pred": y_pred,
        }
    # Regression
    predicted_values = _regression_target(predicted, "Predictions")
    r2_reason = None
    if len(actual_values) < 2:
        r2_reason = "R² needs at least two test rows."
    elif np.all(actual_values == actual_values[0]):
        r2_reason = "R² is undefined because the test target is constant."
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r2 = None if r2_reason else float(r2_score(actual_values, predicted_values, force_finite=False))
        mae = float(mean_absolute_error(actual_values, predicted_values))
        mse = float(mean_squared_error(actual_values, predicted_values))
    if not np.isfinite(mae) or not np.isfinite(mse):
        raise ValueError("Regression error metrics overflowed; rescale the target values before evaluating.")
    if r2 is not None and not np.isfinite(r2):
        r2 = None
        r2_reason = "R² could not be computed as a finite value for this test split."
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R²": None if r2 is None else round(r2, 4),
        "r2_unavailable_reason": r2_reason,
        "y_pred": y_pred,
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, labels=None) -> go.Figure:
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] == 0 or cm.shape[0] != cm.shape[1]:
        raise ValueError("The confusion matrix must be a non-empty square matrix.")
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    if len(labels) != cm.shape[0]:
        raise ValueError("Confusion matrix labels must match the matrix dimensions.")
    labels = [str(label) for label in labels]
    fig = px.imshow(
        cm,
        x=labels, y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels={"x": "Predicted", "y": "Actual"},
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def plot_residuals(y_test, y_pred) -> go.Figure:
    residuals = np.array(y_test) - np.array(y_pred)
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={"x": "Predicted", "y": "Residual"},
        title="Residual Plot",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig


def plot_feature_importance(model, feature_names: list[str]) -> go.Figure | None:
    importance = None
    coefficients = False
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefficients = True
        coef = model.coef_
        if sparse.issparse(coef):
            if coef.ndim != 2 or coef.shape[0] == 0:
                raise ValueError("Model coefficients must contain at least one coefficient row.")
            importance = np.asarray(abs(coef).mean(axis=0)).ravel()
        else:
            coef = np.asarray(coef, dtype=float)
            if coef.ndim not in (1, 2) or (coef.ndim == 2 and coef.shape[0] == 0):
                raise ValueError("Model coefficients must be a feature vector or a class-by-feature matrix.")
            importance = np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)
    if importance is None:
        return None

    if len(feature_names) == 0:
        raise ValueError("Feature importance needs at least one feature name.")
    importance = np.asarray(importance, dtype=float)
    if importance.ndim != 1 or len(importance) != len(feature_names):
        raise ValueError("The number of model importance values does not match the feature names.")
    if not np.isfinite(importance).all():
        raise ValueError("Model importance values must be finite.")

    idx = np.argsort(-importance, kind="stable")[:20]
    names = [feature_names[i] for i in idx]
    vals = importance[idx]
    title = "Mean absolute coefficient (top 20)" if coefficients else "Feature importance (top 20)"
    fig = px.bar(x=vals, y=names, orientation="h", title=title)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig
