import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


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

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state,
    )

    # Track what we did
    info: dict = {
        "dropped": drop_cols,
        "encoded": {},
        "scaled": [],
        "fill_values": {},
        "encoders": {},
        "scalers": {},
    }

    # Encode categoricals
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        cardinality = int(X_train[col].nunique())
        method = "onehot" if cardinality < 10 else "label"
        info["encoded"][col] = method
        if method == "onehot":
            train_dummies = pd.get_dummies(X_train[col], prefix=col, drop_first=True, dtype=int)
            test_dummies = pd.get_dummies(X_test[col], prefix=col, drop_first=True, dtype=int)
            test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=False)
            X_train = pd.concat([X_train.drop(columns=[col]), train_dummies], axis=1)
            X_test = pd.concat([X_test.drop(columns=[col]), test_dummies], axis=1)
            info["encoders"][col] = train_dummies.columns.tolist()
        else:
            categories = {
                value: code
                for code, value in enumerate(X_train[col].dropna().astype(str).unique())
            }
            X_train[col] = X_train[col].astype(str).map(categories)
            X_test[col] = X_test[col].astype(str).map(categories).fillna(-1)
            info["encoders"][col] = categories

    # Drop any remaining non-numeric columns (safety)
    X_train = X_train.select_dtypes(include="number")
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Fill remaining NaNs with train medians only
    fill_values = X_train.median()
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)
    info["fill_values"] = fill_values.to_dict()

    # Scale numeric features
    for col in X_train.columns:
        scaler = StandardScaler()
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col] = scaler.transform(X_test[[col]])
        info["scalers"][col] = scaler
    info["scaled"] = list(X_train.columns)

    # Encode target if categorical
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        le.fit(y.astype(str))
        y_train = pd.Series(
            le.transform(y_train.astype(str)), index=y_train.index, name=target_col
        )
        y_test = pd.Series(
            le.transform(y_test.astype(str)), index=y_test.index, name=target_col
        )
        info["target_encoder"] = le

    return X_train, X_test, y_train, y_test, info
