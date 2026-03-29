import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler


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


def auto_preprocess(
    df: pd.DataFrame,
    target_col: str,
    column_meta: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Full preprocessing pipeline.

    Returns (X_train, X_test, y_train, y_test, info_dict).
    """
    df = df.copy()

    # Separate target
    y = df[target_col]
    df = df.drop(columns=[target_col])

    # Drop columns with role id / ignore / datetime
    drop_cols = [
        c for c, m in column_meta.items()
        if m.get("role") in ("id", "ignore", "datetime") and c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    # Track what we did
    info: dict = {"dropped": drop_cols, "encoded": {}, "scaled": []}

    # Encode categoricals
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        cardinality = int(df[col].nunique())
        method = "onehot" if cardinality < 10 else "label"
        df = encode_column(df, col, method=method)
        info["encoded"][col] = method

    # Drop any remaining non-numeric columns (safety)
    df = df.select_dtypes(include="number")

    # Fill remaining NaNs with column median
    df = df.fillna(df.median())

    # Scale numeric features
    scalers = {}
    for col in df.columns:
        df, scaler = scale_column(df, col, method="standard")
        scalers[col] = scaler
    info["scaled"] = list(df.columns)

    # Encode target if categorical
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        encoded_y = le.fit_transform(y.astype(str))
        y = pd.Series(encoded_y, index=y.index, name=target_col)  # type: ignore[arg-type]
        info["target_encoder"] = le

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, info
