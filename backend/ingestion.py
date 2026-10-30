"""Lossless ingestion and conservative, atomic schema conversion."""

import csv
import io
import json
import re
from decimal import Decimal

import numpy as np
import pandas as pd

from backend import resources


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


def _read_source(uploaded_file):
    """Read independently of an upload's cursor without taking ownership of it."""
    if hasattr(uploaded_file, "getvalue"):
        source = uploaded_file.getvalue()
        size = len(source.encode("utf-8")) if isinstance(source, str) else len(source)
        if size > resources.MAX_UPLOAD_BYTES:
            raise ValueError("The upload exceeds the 50 MiB file limit.")
        return source
    position = uploaded_file.tell()
    try:
        uploaded_file.seek(0)
        source = uploaded_file.read(resources.MAX_UPLOAD_BYTES + 1)
        size = len(source.encode("utf-8")) if isinstance(source, str) else len(source)
        if size > resources.MAX_UPLOAD_BYTES:
            raise ValueError("The upload exceeds the 50 MiB file limit.")
        return source
    finally:
        uploaded_file.seek(position)


def _text_source(uploaded_file, encoding):
    if not isinstance(encoding, str) or not encoding:
        raise ValueError("Choose a text encoding.")
    source = _read_source(uploaded_file)
    if isinstance(source, str):
        return source.removeprefix("\ufeff")
    try:
        return source.decode(encoding, errors="strict")
    except LookupError as exc:
        raise ValueError(f"Unknown text encoding: {encoding}.") from exc
    except UnicodeError as exc:
        raise ValueError(f"The file cannot be decoded with {encoding}; choose its actual encoding.") from exc


def _excel_engine(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx"):
        return "openpyxl"
    if name.endswith(".xls"):
        return "xlrd"
    raise ValueError("Choose an .xlsx or .xls workbook.")


def list_excel_sheets(uploaded_file) -> list[str]:
    """List workbook sheets without consuming or closing the uploaded stream."""
    with pd.ExcelFile(io.BytesIO(_read_source(uploaded_file)), engine=_excel_engine(uploaded_file)) as workbook:
        return list(workbook.sheet_names)


def _unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _json_document(text):
    def finite_float(token):
        value = float(token)
        if not np.isfinite(value):
            raise ValueError("A JSON number exceeds the supported finite floating-point range.")
        if value == 0 and Decimal(token) != 0:
            raise ValueError("A JSON number is too small to represent without becoming zero; quote it to preserve it as text.")
        return value

    return json.loads(
        text, object_pairs_hook=_unique_json_keys,
        parse_float=finite_float,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-finite JSON value: {value}.")),
    )


def _json_frame(value):
    def scalar(item):
        return item is None or isinstance(item, (str, bool, int, float))

    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        if any(not scalar(item) for row in value for item in row.values()):
            raise ValueError("Nested JSON cells are unsupported; choose a path to a flat records array.")
        return pd.DataFrame(value, dtype=object)
    if isinstance(value, dict):
        if all(scalar(item) for item in value.values()):
            return pd.DataFrame([value], dtype=object)
        if all(isinstance(column, list) for column in value.values()):
            columns = list(value.values())
            if len({len(column) for column in columns}) > 1:
                raise ValueError("JSON columns must contain the same number of values.")
            if any(not scalar(item) for column in columns for item in column):
                raise ValueError("Nested JSON cells are unsupported; choose a path to a flat table.")
            return pd.DataFrame(value, dtype=object)
        if value and all(isinstance(column, dict) for column in value.values()):
            if all(scalar(item) for column in value.values() for item in column.values()):
                return pd.DataFrame(value, dtype=object)
    raise ValueError("Select a JSON records array or a flat column-oriented object.")


def parse_file(
    uploaded_file, *, delimiter=None, encoding="utf-8-sig", sheet_name=0,
    json_path=None, json_lines=False,
) -> pd.DataFrame:
    """Preserve textual values until the user reviews the schema.

    Only empty fields are treated as missing in CSV/TSV/Excel. In particular,
    'NA' and 'NULL' remain literal strings. Native Excel numeric/date cells and
    native JSON scalars retain their existing types. Numeric separator choices
    belong to schema conversion, so reading a file never normalizes its text.
    """
    name = uploaded_file.name.lower()
    options = {"dtype": object, "keep_default_na": False, "na_values": [""]}
    if name.endswith((".csv", ".tsv", ".txt")):
        delimiter = ("," if name.endswith(".csv") else "\t") if delimiter is None else delimiter
        if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in ('"', "\r", "\n", "\0"):
            raise ValueError("The delimiter must be one literal character (not a regular expression).")
        text = _text_source(uploaded_file, encoding)
        try:
            rows = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
            header = next((row for row in rows if row), [])
            if not header:
                raise ValueError("The delimited file has no header row.")
            if len(header) > resources.MAX_DATASET_COLUMNS:
                raise ValueError("The file exceeds the column budget. Select fewer columns before uploading.")
            if len(set(header)) != len(header):
                raise ValueError("Column names must be unique; the file contains duplicate headers.")
            for number, row in enumerate(rows, start=2):
                if len(row) > len(header):
                    raise ValueError(f"Row {number} has more fields than the header; check the delimiter.")
        except csv.Error as exc:
            raise ValueError(f"The delimited file has invalid quoting: {exc}") from exc
        result = pd.read_csv(io.StringIO(text), sep=delimiter, nrows=resources.MAX_DATASET_ROWS + 1, **options)
        resources.validate_dataset(result)
        return result
    if name.endswith((".xlsx", ".xls")):
        if isinstance(sheet_name, bool) or not isinstance(sheet_name, (str, int)) or isinstance(sheet_name, int) and sheet_name < 0:
            raise ValueError("Choose one Excel sheet by name or non-negative position.")
        raw = pd.read_excel(io.BytesIO(_read_source(uploaded_file)), sheet_name=sheet_name,
                            header=None, nrows=resources.MAX_DATASET_ROWS + 2,
                            engine=_excel_engine(uploaded_file), **options)
        if raw.empty:
            raise ValueError("The selected Excel sheet has no header row.")
        header = [f"Unnamed: {position}" if pd.isna(value) else value for position, value in enumerate(raw.iloc[0])]
        if len(set(header)) != len(header):
            raise ValueError("Column names must be unique; the sheet contains duplicate headers.")
        result = raw.iloc[1:].reset_index(drop=True).copy()
        result.columns = header
        resources.validate_dataset(result)
        return result
    if name.endswith((".json", ".jsonl", ".ndjson")):
        if not isinstance(json_lines, bool):
            raise ValueError("JSON Lines must be a boolean option.")
        if json_path is not None and (not isinstance(json_path, list) or any(not isinstance(key, str) for key in json_path)):
            raise ValueError("The JSON path must be a list of object keys.")
        text = _text_source(uploaded_file, encoding)
        if json_lines or name.endswith((".jsonl", ".ndjson")):
            if json_path:
                raise ValueError("JSON paths are unavailable for JSON Lines files.")
            value = [_json_document(line) for line in text.splitlines() if line.strip()]
        else:
            value = _json_document(text)
            for key in json_path or []:
                if not isinstance(value, dict) or key not in value:
                    raise ValueError(f"JSON path key {key!r} was not found in an object.")
                value = value[key]
        result = _json_frame(value)
        resources.validate_dataset(result)
        return result
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def validate_numeric_format(decimal=".", thousands=None) -> None:
    if decimal not in (".", ","):
        raise ValueError("The decimal separator must be a dot or comma.")
    if thousands not in (None, ".", ",", " ") or thousands == decimal:
        raise ValueError("Choose no thousands separator, a space, or a separator different from the decimal mark.")


def _numeric_text(col, decimal=".", thousands=None):
    validate_numeric_format(decimal, thousands)
    integer = r"\d+"
    if thousands is not None:
        integer = rf"(?:\d+|\d{{1,3}}(?:{re.escape(thousands)}\d{{3}})+)"
    mark = re.escape(decimal)
    pattern = re.compile(rf"[+-]?(?:{integer}(?:{mark}\d*)?|{mark}\d+)(?:[eE][+-]?\d+)?")
    values = []
    for value in col:
        if pd.isna(value):
            values.append(pd.NA)
            continue
        text = str(value).strip()
        # Workbook/JSON numbers already have native numerical meaning.
        native = not isinstance(value, str)
        if not (bool(_NUMBER.fullmatch(text)) if native else bool(pattern.fullmatch(text))):
            raise ValueError("Expected plain numbers with the selected separators and groups of three; units and compound values need explicit parsing.")
        if not native:
            if thousands is not None:
                text = text.replace(thousands, "")
            if decimal != ".":
                text = text.replace(decimal, ".")
        values.append(text)
    return pd.Series(values, index=col.index, name=col.name, dtype="string")


def _numeric_column(col: pd.Series, target: str, decimal=".", thousands=None) -> pd.Series:
    text = _numeric_text(col, decimal, thousands)

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


def infer_dtype(col: pd.Series, *, decimal=".", thousands=None) -> tuple[str, float]:
    """Infer only when every present value has an unambiguous interpretation.

    Confidence measures parser coverage, not semantic certainty. Ambiguous dates,
    leading-zero codes, units, and mixed numeric/text columns remain text.
    """
    validate_numeric_format(decimal, thousands)
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
    try:
        numeric_text = _numeric_text(col, decimal, thousands).dropna()
    except (ValueError, TypeError, OverflowError):
        numeric_text = None
    if numeric_text is not None and not numeric_text.str.fullmatch(_LEADING_ZERO).any():
        for target in ("int64", "float64"):
            try:
                _numeric_column(col, target, decimal, thousands)
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


def infer_column_meta(df: pd.DataFrame, *, decimal=".", thousands=None) -> dict:
    validate_numeric_format(decimal, thousands)
    meta = {}
    for name in df.columns:
        col = df[name]
        role = classify_role(col, name)
        dtype, confidence = infer_dtype(col, decimal=decimal, thousands=thousands)
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
            "numeric_decimal": decimal,
            "numeric_thousands": thousands,
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
        if (meta.get("user_dtype") is None and target == "object"
                and isinstance(col.dtype, pd.StringDtype)):
            # "object" is the inferred logical text choice, not a request to
            # replace existing string storage or its missing-value sentinel.
            # An explicit user_dtype="object" still performs that conversion.
            continue
        try:
            if target in ("int64", "float64"):
                converted = _numeric_column(col, target, meta.get("numeric_decimal", "."), meta.get("numeric_thousands"))
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

