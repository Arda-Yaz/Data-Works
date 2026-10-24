"""Lossless ingestion and conservative, atomic schema conversion."""

import re
from decimal import Decimal

import numpy as np
import pandas as pd


_BOOL_MAP = {
    "true": True, "false": False, "yes": True, "no": False,
    "1": True, "0": False, "t": True, "f": False,
}
_BOOL_WORDS = set(_BOOL_MAP) - {"1", "0"}
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_LEADING_ZERO = re.compile(r"[+-]?0\d+")
_ID_PATTERN = re.compile(
    r"^(id|index|key)$|_id$|_key$|^unnamed:\s*\d+$", re.IGNORECASE,
)
_INT_LIMITS = np.iinfo(np.int64)


def parse_file(uploaded_file) -> pd.DataFrame:
    """Preserve textual values until the user reviews the schema.

    Only empty fields are treated as missing in CSV/TSV/Excel. In particular,
    'NA' and 'NULL' remain literal strings. Native Excel numeric/date cells and
    native JSON scalars retain their existing types.
    """
    name = uploaded_file.name.lower()
    options = {"dtype": object, "keep_default_na": False, "na_values": [""]}
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, **options)
    if name.endswith((".tsv", ".txt")):
        return pd.read_csv(uploaded_file, sep="\t", **options)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, **options)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file, dtype=False, convert_dates=False)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def _numeric_column(col: pd.Series, target: str) -> pd.Series:
    text = col.astype("string").str.strip()
    present = text.dropna()
    if not present.str.fullmatch(_NUMBER).all():
        raise ValueError("Expected plain numbers; units and compound values need explicit parsing.")

    decimals = [None if pd.isna(value) else Decimal(value) for value in text]
    if target == "int64":
        if any(
            value is not None and (
                value != value.to_integral_value()
                or not _INT_LIMITS.min <= value <= _INT_LIMITS.max
            )
            for value in decimals
        ):
            raise ValueError("Values must be whole numbers within the signed 64-bit range.")
        values = [pd.NA if value is None else int(value) for value in decimals]
        result = pd.Series(pd.array(values, dtype="Int64"), index=col.index, name=col.name)
        return result if result.isna().any() else result.astype("int64")

    if any(
        value is not None and value == value.to_integral_value() and abs(value) > 2**53
        for value in decimals
    ):
        raise ValueError("Large whole numbers would lose precision as floats; use integers or text.")
    result = pd.to_numeric(text, errors="raise").astype("Float64")
    if not np.isfinite(result.dropna().to_numpy(dtype=float)).all():
        raise ValueError("Numeric values must be finite.")
    if any(
        original is not None and original != 0 and converted == 0
        for original, converted in zip(decimals, result)
    ):
        raise ValueError("Values are too small to represent as floats without becoming zero.")
    return result if result.isna().any() else result.astype("float64")


def infer_dtype(col: pd.Series) -> tuple[str, float]:
    """Infer only when every present value has an unambiguous interpretation.

    Confidence measures parser coverage, not semantic certainty. Ambiguous dates,
    leading-zero codes, units, and mixed numeric/text columns remain text.
    """
    present = col.dropna()
    if present.empty:
        return "object", 0.0
    if pd.api.types.is_bool_dtype(col):
        return "bool", 1.0
    if pd.api.types.is_numeric_dtype(col):
        return str(col.dtype), 1.0
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime64[ns]", 1.0

    text = present.astype("string").str.strip()
    if set(text.str.lower().unique()).issubset(_BOOL_WORDS):
        return "bool", 1.0
    if text.str.fullmatch(_NUMBER).all() and not text.str.fullmatch(_LEADING_ZERO).any():
        for target in ("int64", "float64"):
            try:
                _numeric_column(col, target)
                return target, 1.0
            except (ValueError, TypeError, OverflowError):
                continue

    # Slash-separated dates require a user-specified format, never day/month guessing.
    if text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        try:
            pd.to_datetime(text, format="%Y-%m-%d", errors="raise")
            return "datetime64[ns]", 1.0
        except (ValueError, TypeError, OverflowError):
            pass
    return "object", 1.0


def classify_role(col: pd.Series, col_name: str) -> str:
    if _ID_PATTERN.search(str(col_name)):
        return "id"
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"
    return "feature"


def infer_column_meta(df: pd.DataFrame) -> dict:
    meta = {}
    for name in df.columns:
        col = df[name]
        role = classify_role(col, name)
        dtype, confidence = infer_dtype(col)
        if role == "id" and not pd.api.types.is_numeric_dtype(col):
            dtype = "object"
        if dtype == "datetime64[ns]" and role != "id":
            role = "datetime"
        cardinality = int(col.nunique())
        meta[name] = {
            "original_dtype": str(col.dtype),
            "inferred_dtype": dtype,
            "confidence": confidence,
            "user_dtype": None,
            "role": role,
            "cardinality": cardinality,
            "is_categorical": (
                dtype in ("object", "bool", "category")
                or cardinality / max(len(df), 1) < 0.05
                or (cardinality < 30 and dtype != "datetime64[ns]")
            ),
            "allowed_values": None,
            "date_format": None,
        }
    return meta


def apply_inferred_dtypes(df: pd.DataFrame, column_meta: dict) -> pd.DataFrame:
    """Return a converted copy, or raise without changing the input.

    No non-missing value may silently turn into a missing value. Callers must
    commit both the returned frame and its schema only after this call succeeds.
    """
    result = df.copy()
    for name, meta in column_meta.items():
        if name not in result.columns:
            continue
        target = meta.get("user_dtype") or meta["inferred_dtype"]
        col = result[name]
        if target == str(col.dtype):
            continue
        try:
            if target in ("int64", "float64"):
                converted = _numeric_column(col, target)
            elif target == "bool":
                text = col.astype("string").str.strip().str.lower()
                if not text.dropna().isin(_BOOL_MAP).all():
                    raise ValueError("Expected true/false, yes/no, t/f, or 1/0.")
                converted = text.map(_BOOL_MAP).astype("boolean")
            elif target.startswith("datetime"):
                converted = pd.to_datetime(
                    col, format=meta.get("date_format") or "%Y-%m-%d", errors="raise",
                )
            elif target in ("object", "string", "category"):
                converted = col.astype(target)
            else:
                raise ValueError(f"Unsupported requested type: {target}")
            if (col.notna() & converted.isna()).any():
                raise ValueError("Conversion would introduce missing values.")
            result[name] = converted
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Column {name!r} cannot be converted to {target}: {exc}") from exc
    return result

