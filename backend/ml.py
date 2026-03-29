import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
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
    r2_score,
    recall_score,
)


# ── Problem type detection ───────────────────────────────────────────────────

def detect_problem_type(y: pd.Series) -> str:
    """Return 'classification' or 'regression'."""
    if y.dtype == "object" or y.dtype.name in ("category", "bool"):
        return "classification"
    if y.nunique() <= 20:
        return "classification"
    return "regression"


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
    if problem_type == "classification":
        return _CLASSIFIERS.copy()
    return _REGRESSORS.copy()


# ── Training & evaluation ────────────────────────────────────────────────────

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, problem_type: str) -> dict:
    y_pred = model.predict(X_test)
    if problem_type == "classification":
        avg = "weighted" if y_test.nunique() > 2 else "binary"
        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "y_pred": y_pred,
        }
    # Regression
    return {
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "MSE": round(mean_squared_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R²": round(r2_score(y_test, y_pred), 4),
        "y_pred": y_pred,
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, labels=None) -> go.Figure:
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    fig = px.imshow(
        cm,
        x=labels, y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels={"x": "Predicted", "y": "Actual"},
    )
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
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importance = np.abs(coef).flatten() if coef.ndim > 1 else np.abs(coef)
    if importance is None:
        return None

    idx = np.argsort(importance)[::-1][:20]
    names = [feature_names[i] for i in idx]
    vals = importance[idx]
    fig = px.bar(x=vals, y=names, orientation="h", title="Feature Importance (top 20)")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig
