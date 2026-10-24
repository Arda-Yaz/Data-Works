import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler,
)


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


def auto_preprocess(
    df: pd.DataFrame,
    target_col: str,
    column_meta: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Split first, then fit one reusable preprocessor using training data only.

    Rows with missing features are retained. Entirely missing numeric training
    features use zero; categorical features use the training mode. High-cardinality
    categoricals group infrequent levels, limiting each to 64 output features.
    """
    if target_col not in df.columns:
        raise ValueError("Select a target present in the dataset.")
    usable = df.dropna(subset=[target_col])
    y = usable[target_col]
    features = usable.drop(columns=[target_col])
    dropped = [
        c for c in features.columns
        if column_meta.get(c, {}).get("role") in ("id", "ignore", "datetime")
    ]
    features = features.drop(columns=dropped)
    if features.shape[1] == 0:
        raise ValueError("Select at least one usable feature.")

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=random_state,
    )
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("values", FunctionTransformer(_numeric_values, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("strings", FunctionTransformer(_categorical_strings, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, max_categories=64,
                feature_name_combiner=_category_feature_name,
            )),
        ]), categorical))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    train_values = preprocessor.fit_transform(X_train)
    test_values = preprocessor.transform(X_test)
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
        "dropped_missing_targets": len(df) - len(usable),
        "encoded": {c: "onehot" for c in categorical},
        "scaled": numeric,
        "fill_values": fill_values,
        "preprocessor": preprocessor,
    }
    return prepared_train, prepared_test, y_train, y_test, info

