from collections import Counter
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler,
)
from sklearn.utils.multiclass import type_of_target


def encode_column(
    df: pd.DataFrame, col: str, method: str = "onehot",
) -> pd.DataFrame:
    """Encode a single categorical column. Returns a modified copy."""
    df = df.copy()
    if method == "label":
        le = LabelEncoder()
        mask = df[col].notna()
        encoded = le.fit_transform(df.loc[mask, col].astype(str))
        df.loc[mask, col] = pd.array(encoded)  # type: ignore[call-overload]
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif method == "onehot":
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def scale_column(
    df: pd.DataFrame, col: str, method: str = "standard",
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler]:
    """Scale a numeric column. Returns (df, fitted_scaler)."""
    df = df.copy()
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    values = df[[col]].values
    df[col] = scaler.fit_transform(values)
    return df, scaler


def _categorical_strings(frame):
    # Unify None/pd.NA/NaN for sklearn, including nullable strings and booleans.
    return frame.astype("string").to_numpy(dtype=object, na_value=np.nan)


def _numeric_values(frame):
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Numeric features contain infinity; clean those values first.")
    return values


def _category_feature_name(feature, category):
    return f"{feature!r}={category!r}"


class _DataFrameColumnTransformer(ColumnTransformer):
    """Retain source labels while giving sklearn unambiguous string names."""

    def fit_transform(self, X, y=None, **params):
        self.source_columns_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        if self.source_columns_ is not None:
            counts = Counter(map(str, self.source_columns_))
            names = []
            used = set()
            for position, column in enumerate(self.source_columns_):
                name = str(column) if counts[str(column)] == 1 else f"{type(column).__name__}:{column!r}"
                candidate = name
                while candidate in used:
                    candidate = f"{candidate} [{position}]"
                names.append(candidate)
                used.add(candidate)
            self._column_labels_ = names
            X = X.copy(deep=False)
            X.columns = names
        return super().fit_transform(X, y, **params)

    def transform(self, X, **params):
        if isinstance(X, pd.DataFrame) and self.source_columns_ is not None:
            if not X.columns.is_unique:
                raise ValueError("Prediction feature column names must be unique.")
            missing = [name for name in self.source_columns_ if name not in X.columns]
            if missing:
                raise ValueError(f"Prediction features are missing required columns: {missing}.")
            X = X.loc[:, self.source_columns_].copy(deep=False)
            X.columns = self._column_labels_
        return super().transform(X, **params)


def _validate_target(target: pd.Series, problem_type: str | None) -> None:
    if problem_type not in (None, "classification", "regression"):
        raise ValueError("Choose classification or regression.")
    if target.empty:
        raise ValueError("No rows have a non-missing target value.")
    if pd.api.types.is_complex_dtype(target.dtype):
        raise ValueError("Complex target values are not supported.")
    if pd.api.types.is_numeric_dtype(target.dtype):
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError("The target contains non-finite values; clean them before training.")
    else:
        # Object columns can contain real numeric scalars alongside text labels.
        for value in target:
            if isinstance(value, Real):
                try:
                    finite = np.isfinite(float(value))
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError("The target contains non-finite values; clean them before training.")
    if problem_type == "regression":
        if not pd.api.types.is_numeric_dtype(target.dtype) or pd.api.types.is_bool_dtype(target.dtype):
            raise ValueError("Regression requires a numeric target; confirm its numeric schema first.")
    elif problem_type == "classification":
        try:
            kind = type_of_target(target.to_numpy())
        except (TypeError, ValueError) as exc:
            raise ValueError("Classification requires consistently typed, discrete class labels.") from exc
        if kind not in ("binary", "multiclass"):
            raise ValueError("Classification requires discrete class labels; use regression for fractional targets.")
        if target.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")


def _chronological_values(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"Time column {series.name!r} contains missing values.")
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
        if pd.api.types.is_complex_dtype(series.dtype) or not np.isfinite(series.to_numpy(dtype=float)).all():
            raise ValueError("Chronological split times must be real, finite values.")
        return series
    if pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError("Choose a numeric or datetime time column, not a boolean column.")
    try:
        parsed = pd.to_datetime(series, format="ISO8601", errors="raise", utc=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Time column {series.name!r} must contain valid ISO dates/datetimes or numeric times.") from exc
    if parsed.isna().any():
        raise ValueError(f"Time column {series.name!r} contains values that parse as missing timestamps.")
    return parsed


def _split_positions(
    usable: pd.DataFrame, target: pd.Series, test_size: float, random_state: int,
    problem_type: str | None, strategy: str, group_col, time_col,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if isinstance(test_size, bool) or not isinstance(test_size, Real) or not 0 < test_size < 1:
        raise ValueError("Test size must be a fraction strictly between 0 and 1.")
    if len(usable) < 2:
        raise ValueError("At least two rows with non-missing targets are required for a train/test split.")
    if strategy not in ("random", "stratified", "group", "chronological"):
        raise ValueError(f"Unknown split strategy: {strategy!r}.")
    positions = np.arange(len(usable))
    messages = []
    if strategy == "stratified":
        if problem_type != "classification":
            raise ValueError("Stratified splitting is available only for classification.")
        counts = target.value_counts()
        counts = counts[counts > 0]  # Categorical targets may retain unused levels.
        classes = len(counts)
        test_rows = int(np.ceil(len(usable) * test_size))
        if counts.min() < 2:
            raise ValueError("Stratified splitting requires at least two rows in every class.")
        if min(test_rows, len(usable) - test_rows) < classes:
            raise ValueError("The chosen split must leave at least one row per class in both train and test sets.")
        train, test = train_test_split(positions, test_size=test_size, random_state=random_state, stratify=target)
    elif strategy == "group":
        if group_col is None or group_col not in usable.columns or group_col == target.name:
            raise ValueError("Select a grouping column different from the target.")
        groups = usable[group_col]
        if groups.isna().any():
            raise ValueError(f"Grouping column {group_col!r} contains missing values.")
        try:
            codes, identities = pd.factorize(groups, sort=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Grouping requires scalar, consistently identifiable group values.") from exc
        if len(identities) < 2:
            raise ValueError("A group split requires at least two distinct groups.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train, test = next(splitter.split(positions, groups=codes))
    elif strategy == "chronological":
        if time_col is None or time_col not in usable.columns or time_col == target.name:
            raise ValueError("Select a time column different from the target.")
        times = _chronological_values(usable[time_col])
        if times.nunique() < 2:
            raise ValueError("A chronological split requires at least two distinct times.")
        ordered = times.sort_values(kind="stable")
        # Use positional ordering even if the caller has duplicate index labels.
        order = np.argsort(times.to_numpy(), kind="stable")
        proposed = len(usable) - int(np.ceil(len(usable) * test_size))
        if proposed <= 0:
            raise ValueError("The chosen test size leaves no chronological training rows.")
        boundary = ordered.iloc[proposed]
        train_rows = int((ordered < boundary).sum())
        if train_rows == 0:
            raise ValueError("Keeping equal timestamps together leaves no training rows; reduce the test size.")
        train, test = order[:train_rows], order[train_rows:]
        if train_rows != proposed:
            messages.append("The split size was adjusted to keep equal timestamps in the same partition.")
    else:
        train, test = train_test_split(positions, test_size=test_size, random_state=random_state)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("The chosen split must leave both training and test rows.")
    if problem_type == "classification":
        observed = set(target.tolist())
        training = set(target.iloc[train].tolist())
        if training != observed:
            raise ValueError("The training partition is missing target classes; choose another split or provide more class coverage.")
        if set(target.iloc[test].tolist()) != observed:
            if strategy == "stratified":
                raise ValueError(
                    "The stratified test partition is missing target classes. "
                    "Increase the test fraction or provide more minority-class samples."
                )
            messages.append("The test partition does not contain every class; its metrics do not assess the missing classes.")
    return train, test, messages


def auto_preprocess(
    df: pd.DataFrame,
    target_col: str,
    column_meta: dict,
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    problem_type: str | None = None,
    split_strategy: str = "random",
    group_col: str | None = None,
    time_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Split first, then fit one reusable preprocessor using training data only.

    Rows with missing features are retained. Explicit task calls remove constant
    or entirely missing training features; legacy calls retain their imputation
    behavior. Categorical fills use the training mode. High-cardinality categories
    group infrequent levels, limiting each to 64 output features. Chronological
    dates are normalized to UTC and equal timestamps remain in one partition.
    """
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError("Provide a DataFrame with unique column names.")
    if target_col not in df.columns:
        raise ValueError("Select a target present in the dataset.")
    usable = df.dropna(subset=[target_col])
    y = usable[target_col]
    _validate_target(y, problem_type)
    train_positions, test_positions, messages = _split_positions(
        usable, y, test_size, random_state, problem_type, split_strategy, group_col, time_col,
    )
    features = usable.drop(columns=[target_col])
    dropped = [
        c for c in features.columns
        if column_meta.get(c, {}).get("role") in ("id", "ignore", "datetime")
    ]
    split_column = group_col if split_strategy == "group" else time_col if split_strategy == "chronological" else None
    if split_column in features.columns and split_column not in dropped:
        dropped.append(split_column)
    features = features.drop(columns=dropped)
    if features.shape[1] == 0:
        raise ValueError("Select at least one usable feature.")

    X_train, X_test = features.iloc[train_positions], features.iloc[test_positions]
    y_train, y_test = y.iloc[train_positions], y.iloc[test_positions]
    constants = [name for name in X_train.columns if X_train[name].nunique(dropna=True) <= 1]
    if constants:
        messages.append(f"Constant or entirely missing training features: {constants}.")
    dropped_constants = constants if problem_type is not None else []
    if dropped_constants:
        X_train, X_test = X_train.drop(columns=dropped_constants), X_test.drop(columns=dropped_constants)
        dropped.extend(dropped_constants)
    if X_train.shape[1] == 0:
        raise ValueError("No informative training features remain after removing constant or entirely missing columns.")
    for name in X_train.columns:
        if pd.api.types.is_complex_dtype(X_train[name].dtype):
            raise ValueError(f"Feature {name!r} contains complex values; real numeric or categorical features are required.")
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("values", FunctionTransformer(_numeric_values, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), [name in numeric for name in X_train.columns]))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("strings", FunctionTransformer(_categorical_strings, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, max_categories=64,
                feature_name_combiner=_category_feature_name,
            )),
        ]), [name in categorical for name in X_train.columns]))

    preprocessor = _DataFrameColumnTransformer(transformers, remainder="drop")
    train_values = preprocessor.fit_transform(X_train)
    test_values = preprocessor.transform(X_test)
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise ValueError("Preprocessing produced non-finite values; rescale extreme numeric features before training.")
    names = preprocessor.get_feature_names_out()
    prepared_train = pd.DataFrame(train_values, index=X_train.index, columns=names)
    prepared_test = pd.DataFrame(test_values, index=X_test.index, columns=names)
    fill_values = {}
    if numeric:
        fill_values = dict(zip(
            numeric, preprocessor.named_transformers_["numeric"].named_steps["impute"].statistics_,
        ))
    info = {
        "dropped": dropped,
        "dropped_constants": dropped_constants,
        "dropped_missing_targets": len(df) - len(usable),
        "encoded": {c: "onehot" for c in categorical},
        "scaled": numeric,
        "fill_values": fill_values,
        "preprocessor": preprocessor,
        "feature_columns": list(X_train.columns),
        "split_strategy": split_strategy,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "warnings": messages,
    }
    return prepared_train, prepared_test, y_train, y_test, info

