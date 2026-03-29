import re

import pandas as pd


# ── Low-level helpers ────────────────────────────────────────────────────────
def _strip_non_numeric(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", pd.NA)
    )


_BOOL_MAP = {
    "true": True, "false": False,
    "yes": True, "no": False,
    "1": True, "0": False,
    "t": True, "f": False,
}

_ID_PATTERN = re.compile(
    r"^(id|index|key)$|_id$|_key$|^unnamed:\s*\d+$", re.IGNORECASE
)


# ── File parsing ─────────────────────────────────────────────────────────────
def parse_file(uploaded_file) -> pd.DataFrame:
    name: str = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep="\t")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


# ── Single-column type inference ─────────────────────────────────────────────
def infer_dtype(col: pd.Series) -> tuple[str, float]:
    """Return (dtype_string, confidence) for a single column."""
    non_null = col.dropna()
    if non_null.empty:
        return str(col.dtype), 0.0

    # Already numeric / datetime — high confidence
    if pd.api.types.is_numeric_dtype(col):
        return str(col.dtype), 1.0
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime64[ns]", 1.0

    # Boolean detection (check before numeric since '1'/'0' overlap)
    if col.dtype == "object":
        stripped = non_null.astype(str).str.strip().str.lower()
        unique_vals = set(stripped.unique())
        if unique_vals and unique_vals.issubset(_BOOL_MAP.keys()):
            return "bool", 1.0

    # Numeric attempt
    if col.dtype == "object":
        s = non_null.astype(str).str.strip()
        num = pd.to_numeric(_strip_non_numeric(s), errors="coerce")
        rate = num.notna().mean()
        if rate > 0.8:
            # decide int vs float
            is_int = (num.dropna() == num.dropna().astype(int)).all()
            dtype_str = "int64" if is_int and rate == 1.0 else "float64"
            return dtype_str, round(float(rate), 3)

    # Datetime attempt
    if col.dtype == "object":
        s = non_null.astype(str).str.strip()
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        rate = dt.notna().mean()
        if rate > 0.8:
            return "datetime64[ns]", round(float(rate), 3)

    # Fallback: keep as object
    return "object", 1.0


# ── Column role classification ───────────────────────────────────────────────
def classify_role(col: pd.Series, col_name: str) -> str:
    # ID-like names with all-unique or sequential values
    if _ID_PATTERN.search(col_name):
        if col.nunique() == len(col):
            return "id"

    # Datetime columns
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"

    return "feature"


# ── Full-dataset metadata inference ──────────────────────────────────────────
def infer_column_meta(df: pd.DataFrame) -> dict:
    meta: dict[str, dict] = {}
    n_rows = len(df)

    for col_name in df.columns:
        col = df[col_name]
        inferred_dtype, confidence = infer_dtype(col)
        cardinality = int(col.nunique())
        is_categorical = (
            inferred_dtype in ("object", "bool")
            or (cardinality / max(n_rows, 1) < 0.05)
            or (cardinality < 30 and inferred_dtype not in ("datetime64[ns]",))
        )
        role = classify_role(col, col_name)
        if inferred_dtype == "datetime64[ns]":
            role = "datetime"

        meta[col_name] = {
            "original_dtype": str(col.dtype),
            "inferred_dtype": inferred_dtype,
            "confidence": confidence,
            "user_dtype": None,
            "role": role,
            "cardinality": cardinality,
            "is_categorical": is_categorical,
            "allowed_values": None,
        }

    return meta


def apply_inferred_dtypes(df: pd.DataFrame, column_meta: dict) -> pd.DataFrame:
    """Cast columns according to inferred (or user-overridden) dtypes."""
    df = df.copy()
    for col_name, meta in column_meta.items():
        if col_name not in df.columns:
            continue
        target = meta.get("user_dtype") or meta["inferred_dtype"]
        if target == str(df[col_name].dtype):
            continue
        try:
            if target == "bool":
                stripped = df[col_name].astype(str).str.strip().str.lower()
                df[col_name] = stripped.map(_BOOL_MAP)
            elif target.startswith("datetime"):
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce", dayfirst=True)
            elif target in ("int64", "float64"):
                df[col_name] = pd.to_numeric(
                    _strip_non_numeric(df[col_name].astype(str).str.strip()),
                    errors="coerce",
                )
                if target == "int64":
                    df[col_name] = df[col_name].astype("Int64")
            else:
                df[col_name] = df[col_name].astype(target, errors="ignore")
        except Exception:
            pass  # keep original if cast fails
    return df
