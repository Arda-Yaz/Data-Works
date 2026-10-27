"""Versioned project files containing data, never executable Python objects."""

import base64
import binascii
import io
import json
import math
import zipfile
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from numbers import Integral, Real
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


FORMAT = "dataworks-project"
FORMAT_VERSION = 1
FULFILLMENT_FORMAT_VERSION = 2
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_JSON_BYTES = 100 * 1024 * 1024
MAX_DATA_ITEMS = 5_000_000
MAX_NODES = 10_000_000
MAX_ROWS = 1_000_000
MAX_COLUMNS = 10_000
MAX_DEPTH = 100
_TAG = "$dataworks"
_EXTENSIONS = {"boolean", "Float32", "Float64", *(
    f"{prefix}Int{bits}" for prefix in ("", "U") for bits in (8, 16, 32, 64)
)}
_TIME_UNITS = {"Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"}


class _Budget:
    def __init__(self):
        self.nodes = 0
        self.data_items = 0
        self.allocated_bytes = 0

    def node(self, depth):
        self.nodes += 1
        if depth > MAX_DEPTH or self.nodes > MAX_NODES:
            raise ValueError("The project has too many nested values.")

    def data(self, size):
        self.data_items += size
        if size < 0 or self.data_items > MAX_DATA_ITEMS:
            raise ValueError("The project contains too many data values; omit undo snapshots or use a smaller dataset.")

    def allocation(self, size):
        self.allocated_bytes += size
        if size < 0 or self.allocated_bytes > MAX_JSON_BYTES:
            raise ValueError("Declared arrays exceed the project's memory allocation limit.")


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Expected an integer in the project data.")
    return int(value)


def _numpy_dtype(name):
    if not isinstance(name, str) or len(name) > 100:
        raise ValueError("Invalid array dtype.")
    try:
        dtype = np.dtype(name)
    except (TypeError, ValueError) as exc:
        raise ValueError("Unsupported array dtype.") from exc
    if dtype.fields is not None or dtype.subdtype is not None or dtype.kind not in "biufcOMmUS":
        raise ValueError(f"Unsupported array dtype: {name}.")
    if (dtype.kind == "f" and dtype.itemsize > 8) or (dtype.kind == "c" and dtype.itemsize > 16):
        raise ValueError("Extended-precision arrays are not supported by the project format.")
    if dtype.kind in "Mm" and np.datetime_data(dtype)[0] not in _TIME_UNITS:
        raise ValueError("Unsupported datetime array unit.")
    if dtype.itemsize > MAX_JSON_BYTES:
        raise ValueError("The declared array dtype is too large.")
    return dtype


def _dtype_to_json(dtype, budget, depth):
    if isinstance(dtype, pd.CategoricalDtype):
        return {"kind": "category", "categories": _encode_index(dtype.categories, budget, depth + 1), "ordered": dtype.ordered}
    if isinstance(dtype, pd.DatetimeTZDtype):
        return {"kind": "datetime_tz", "unit": dtype.unit, "tz": str(dtype.tz)}
    if isinstance(dtype, pd.StringDtype):
        return {"kind": "string", "storage": dtype.storage}
    if str(dtype) in _EXTENSIONS:
        return {"kind": "extension", "name": str(dtype)}
    if isinstance(dtype, pd.PeriodDtype):
        return {"kind": "period", "freq": str(dtype)[7:-1]}
    if isinstance(dtype, pd.IntervalDtype):
        return {"kind": "interval", "subtype": _dtype_to_json(dtype.subtype, budget, depth + 1), "closed": dtype.closed}
    try:
        dtype = _numpy_dtype(np.dtype(dtype).str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported pandas dtype {dtype!s}; convert it before saving.") from exc
    return {"kind": "numpy", "name": dtype.str}


def _dtype_from_json(spec, budget, depth):
    if not isinstance(spec, dict):
        raise ValueError("Invalid dtype descriptor.")
    kind = spec.get("kind")
    if kind == "numpy" and set(spec) == {"kind", "name"}:
        return _numpy_dtype(spec["name"])
    if kind == "extension" and set(spec) == {"kind", "name"} and spec["name"] in _EXTENSIONS:
        return pd.api.types.pandas_dtype(spec["name"])
    if kind == "string" and set(spec) == {"kind", "storage"} and spec["storage"] in ("python", "pyarrow", "pyarrow_numpy"):
        return pd.StringDtype(storage=spec["storage"])
    if kind == "datetime_tz" and set(spec) == {"kind", "unit", "tz"}:
        if spec["unit"] not in ("s", "ms", "us", "ns") or not isinstance(spec["tz"], str) or len(spec["tz"]) > 200:
            raise ValueError("Invalid timezone dtype.")
        return pd.DatetimeTZDtype(unit=spec["unit"], tz=spec["tz"])
    if kind == "category" and set(spec) == {"kind", "categories", "ordered"}:
        if not isinstance(spec["ordered"], bool):
            raise ValueError("Invalid categorical ordering.")
        return pd.CategoricalDtype(_decode_index(spec["categories"], budget, depth + 1), ordered=spec["ordered"])
    if kind == "period" and set(spec) == {"kind", "freq"}:
        return pd.PeriodDtype(_frequency(spec["freq"]))
    if kind == "interval" and set(spec) == {"kind", "subtype", "closed"} and spec["closed"] in ("left", "right", "both", "neither"):
        subtype = _dtype_from_json(spec["subtype"], budget, depth + 1)
        return pd.IntervalDtype(subtype, closed=spec["closed"])
    raise ValueError("Unsupported or malformed dtype descriptor.")


def _frequency(value):
    if not isinstance(value, str) or not value or len(value) > 100:
        raise ValueError("Invalid time frequency.")
    return value


def _encode_data(series, budget, depth):
    dtype = _dtype_to_json(series.dtype, budget, depth + 1)
    budget.data(len(series))
    storage = getattr(series.dtype, "numpy_dtype", None)
    if storage is None:
        storage = series.dtype if isinstance(series.dtype, np.dtype) else np.dtype(object)
    budget.allocation(len(series) * np.dtype(storage).itemsize)
    if isinstance(series.dtype, pd.CategoricalDtype):
        return {"dtype": dtype, "codes": series.cat.codes.tolist()}
    return {"dtype": dtype, "values": [_encode(value, budget, depth + 1) for value in series.tolist()]}


def _decode_data(data, length, budget, depth):
    if not isinstance(data, dict) or set(data) not in ({"dtype", "values"}, {"dtype", "codes"}):
        raise ValueError("Invalid column data.")
    budget.data(length)
    dtype = _dtype_from_json(data["dtype"], budget, depth + 1)
    storage = getattr(dtype, "numpy_dtype", None)
    if storage is None:
        storage = dtype if isinstance(dtype, np.dtype) else np.dtype(object)
    budget.allocation(length * np.dtype(storage).itemsize)
    if isinstance(dtype, pd.CategoricalDtype):
        codes = data.get("codes")
        if not isinstance(codes, list) or len(codes) != length:
            raise ValueError("Categorical column length does not match its index.")
        if any(_integer(code) < -1 or code >= len(dtype.categories) for code in codes):
            raise ValueError("Invalid categorical codes.")
        return pd.Series(pd.Categorical.from_codes(codes, dtype=dtype))
    values = data.get("values")
    if not isinstance(values, list) or len(values) != length:
        raise ValueError("Column length does not match its index.")
    decoded = [_decode(value, budget, depth + 1) for value in values]
    if pd.api.types.is_bool_dtype(dtype):
        if any(not _missing(value) and not isinstance(value, (bool, np.bool_)) for value in decoded):
            raise ValueError("Boolean columns must contain boolean values.")
    elif pd.api.types.is_integer_dtype(dtype):
        if any(not _missing(value) and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)) for value in decoded):
            raise ValueError("Integer columns must contain exact integer values.")
    elif isinstance(dtype, pd.StringDtype):
        if any(not _missing(value) and not isinstance(value, str) for value in decoded):
            raise ValueError("String columns must contain strings.")
    elif pd.api.types.is_float_dtype(dtype):
        for value in decoded:
            if _missing(value):
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError("Floating-point columns must contain numeric values.")
            with np.errstate(over="ignore", under="ignore"):
                stored = np.asarray(value, dtype=storage).item()
            original = value.item() if isinstance(value, np.generic) else value
            if stored != original:
                raise ValueError("A declared floating-point dtype would round the stored value.")
    return pd.Series(decoded, dtype=dtype)


def _missing(value):
    return value is None or value is pd.NA or value is pd.NaT or isinstance(value, (float, np.floating)) and math.isnan(value)


def _encode_index(index, budget, depth):
    budget.node(depth)
    if len(index) > MAX_ROWS:
        raise ValueError("The project index has too many entries.")
    if isinstance(index, pd.RangeIndex):
        return {"kind": "range", "start": index.start, "stop": index.stop, "step": index.step, "name": _encode(index.name, budget, depth + 1)}
    if isinstance(index, pd.MultiIndex):
        return {"kind": "multi", "levels": [_encode_index(level, budget, depth + 1) for level in index.levels],
                "codes": [code.tolist() for code in index.codes], "names": [_encode(name, budget, depth + 1) for name in index.names]}
    return {"kind": "index", "name": _encode(index.name, budget, depth + 1),
            "data": _encode_data(pd.Series(index.array), budget, depth + 1),
            "length": len(index), "freq": getattr(index, "freqstr", None)}


def _decode_index(data, budget, depth):
    budget.node(depth)
    if not isinstance(data, dict):
        raise ValueError("Invalid index descriptor.")
    kind = data.get("kind")
    if kind == "range" and set(data) == {"kind", "start", "stop", "step", "name"}:
        start, stop, step = (_integer(data[key]) for key in ("start", "stop", "step"))
        length = len(range(start, stop, step))
        if length > MAX_ROWS:
            raise ValueError("The declared index has too many rows.")
        return pd.RangeIndex(start, stop, step, name=_decode(data["name"], budget, depth + 1))
    if kind == "multi" and set(data) == {"kind", "levels", "codes", "names"}:
        levels, codes, names = data["levels"], data["codes"], data["names"]
        if not all(isinstance(value, list) for value in (levels, codes, names)) or not 1 <= len(levels) <= 32 or len(levels) != len(codes) or len(levels) != len(names):
            raise ValueError("Invalid multi-index dimensions.")
        decoded_levels = [_decode_index(level, budget, depth + 1) for level in levels]
        length = len(codes[0]) if isinstance(codes[0], list) else -1
        if not 0 <= length <= MAX_ROWS or any(not isinstance(code, list) or len(code) != length for code in codes):
            raise ValueError("Invalid multi-index row counts.")
        budget.data(length * len(levels))
        for level, code in zip(decoded_levels, codes):
            if any(_integer(value) < -1 or value >= len(level) for value in code):
                raise ValueError("Invalid multi-index code.")
        return pd.MultiIndex(levels=decoded_levels, codes=codes, names=[_decode(name, budget, depth + 1) for name in names])
    if kind == "index" and set(data) == {"kind", "name", "data", "length", "freq"}:
        length = _integer(data["length"])
        if not 0 <= length <= MAX_ROWS:
            raise ValueError("Invalid index length.")
        series = _decode_data(data["data"], length, budget, depth + 1)
        name = _decode(data["name"], budget, depth + 1)
        index = pd.Index(series.array, name=name)
        if data["freq"] is not None:
            frequency = _frequency(data["freq"])
            if isinstance(index, pd.DatetimeIndex):
                index = pd.DatetimeIndex(index, name=name, freq=frequency)
            elif isinstance(index, pd.TimedeltaIndex):
                index = pd.TimedeltaIndex(index, name=name, freq=frequency)
        return index
    raise ValueError("Unsupported or malformed index descriptor.")


def _encode(value, budget, depth):
    budget.node(depth)
    if value is pd.NA:
        return {_TAG: "missing", "kind": "NA"}
    if value is pd.NaT:
        return {_TAG: "missing", "kind": "NaT"}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {_TAG: "float", "value": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    if isinstance(value, pd.DataFrame):
        rows, columns = value.shape
        if rows > MAX_ROWS or columns > MAX_COLUMNS or rows * columns > MAX_DATA_ITEMS:
            raise ValueError("The dataset is too large for a project file.")
        return {_TAG: "frame", "index": _encode_index(value.index, budget, depth + 1),
                "columns": _encode_index(value.columns, budget, depth + 1),
                "data": [_encode_data(value.iloc[:, position], budget, depth + 1) for position in range(columns)]}
    if isinstance(value, pd.Series):
        return {_TAG: "series", "name": _encode(value.name, budget, depth + 1),
                "index": _encode_index(value.index, budget, depth + 1), "data": _encode_data(value, budget, depth + 1)}
    if isinstance(value, pd.Timestamp):
        raw = value.asm8
        return {_TAG: "timestamp", "ticks": int(raw.view("i8")), "unit": np.datetime_data(raw.dtype)[0], "tz": str(value.tz) if value.tz else None}
    if isinstance(value, pd.Timedelta):
        return {_TAG: "timedelta", "ticks": int(value.asm8.view("i8")), "unit": np.datetime_data(value.asm8.dtype)[0]}
    if isinstance(value, pd.Period):
        return {_TAG: "period", "ordinal": value.ordinal, "freq": value.freqstr}
    if isinstance(value, pd.Interval):
        return {_TAG: "interval", "left": _encode(value.left, budget, depth + 1), "right": _encode(value.right, budget, depth + 1), "closed": value.closed}
    if isinstance(value, np.ndarray):
        dtype = _numpy_dtype(value.dtype.str)
        if value.size * dtype.itemsize > MAX_JSON_BYTES:
            raise ValueError("The array is too large for a project file.")
        budget.data(value.size)
        budget.allocation(value.size * dtype.itemsize)
        return {_TAG: "array", "dtype": dtype.str, "shape": list(value.shape),
                "values": [_encode(item, budget, depth + 1) for item in value.flat]}
    if isinstance(value, np.generic):
        dtype = _numpy_dtype(value.dtype.str)
        if dtype.kind in "Mm":
            return {_TAG: "numpy_time", "dtype": dtype.str, "ticks": int(value.view("i8"))}
        budget.allocation(dtype.itemsize)
        return {_TAG: "numpy_scalar", "dtype": dtype.str, "value": _encode(value.item(), budget, depth + 1)}
    if isinstance(value, complex):
        return {_TAG: "complex", "real": _encode(value.real, budget, depth + 1), "imag": _encode(value.imag, budget, depth + 1)}
    if isinstance(value, bytes):
        return {_TAG: "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Decimal):
        return {_TAG: "decimal", "value": str(value)}
    if isinstance(value, Fraction):
        return {_TAG: "fraction", "numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, datetime):
        return {_TAG: "datetime", "value": value.isoformat(), "zone": getattr(value.tzinfo, "key", None), "fold": value.fold}
    if isinstance(value, date):
        return {_TAG: "date", "value": value.isoformat()}
    if isinstance(value, time):
        return {_TAG: "time", "value": value.isoformat(), "fold": value.fold}
    if isinstance(value, timedelta):
        return {_TAG: "python_timedelta", "days": value.days, "seconds": value.seconds, "microseconds": value.microseconds}
    if isinstance(value, (list, tuple)):
        items = [_encode(item, budget, depth + 1) for item in value]
        return items if isinstance(value, list) else {_TAG: "tuple", "items": items}
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value) and _TAG not in value:
            return {key: _encode(item, budget, depth + 1) for key, item in value.items()}
        return {_TAG: "mapping", "items": [[_encode(key, budget, depth + 1), _encode(item, budget, depth + 1)] for key, item in value.items()]}
    raise ValueError(f"Project files do not support executable or custom objects ({type(value).__name__}).")


def _decode(value, budget, depth):
    budget.node(depth)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite JSON numbers must use an explicit project tag.")
        return value
    if isinstance(value, list):
        return [_decode(item, budget, depth + 1) for item in value]
    if not isinstance(value, dict):
        raise ValueError("Unsupported JSON value.")
    if _TAG not in value:
        return {key: _decode(item, budget, depth + 1) for key, item in value.items()}
    kind = value[_TAG]
    keys = set(value) - {_TAG}
    if kind == "missing" and keys == {"kind"} and value["kind"] in ("NA", "NaT"):
        return pd.NA if value["kind"] == "NA" else pd.NaT
    if kind == "float" and keys == {"value"} and value["value"] in ("nan", "inf", "-inf"):
        return float(value["value"])
    if kind == "frame" and keys == {"index", "columns", "data"}:
        index = _decode_index(value["index"], budget, depth + 1)
        columns = _decode_index(value["columns"], budget, depth + 1)
        data = value["data"]
        if len(columns) > MAX_COLUMNS or len(index) * len(columns) > MAX_DATA_ITEMS or not isinstance(data, list) or len(data) != len(columns):
            raise ValueError("Invalid project table dimensions.")
        series = [_decode_data(item, len(index), budget, depth + 1) for item in data]
        frame = pd.concat(series, axis=1) if series else pd.DataFrame(index=range(len(index)))
        frame.index, frame.columns = index, columns
        return frame
    if kind == "series" and keys == {"name", "index", "data"}:
        index = _decode_index(value["index"], budget, depth + 1)
        series = _decode_data(value["data"], len(index), budget, depth + 1)
        series.index = index
        return series.rename(_decode(value["name"], budget, depth + 1))
    if kind in ("timestamp", "timedelta") and keys == ({"ticks", "unit", "tz"} if kind == "timestamp" else {"ticks", "unit"}):
        if value["unit"] not in ("s", "ms", "us", "ns"):
            raise ValueError("Unsupported timestamp unit.")
        ticks = _integer(value["ticks"])
        if kind == "timedelta":
            return pd.Timedelta(ticks, unit=value["unit"])
        zone = value["tz"]
        if zone is not None and (not isinstance(zone, str) or len(zone) > 200):
            raise ValueError("Invalid timestamp timezone.")
        stamp = pd.Timestamp(ticks, unit=value["unit"], tz="UTC" if zone else None)
        return stamp.tz_convert(zone) if zone else stamp
    if kind == "period" and keys == {"ordinal", "freq"}:
        return pd.Period(ordinal=_integer(value["ordinal"]), freq=_frequency(value["freq"]))
    if kind == "interval" and keys == {"left", "right", "closed"} and value["closed"] in ("left", "right", "both", "neither"):
        return pd.Interval(_decode(value["left"], budget, depth + 1), _decode(value["right"], budget, depth + 1), closed=value["closed"])
    if kind == "numpy_time" and keys == {"dtype", "ticks"}:
        dtype = _numpy_dtype(value["dtype"])
        if dtype.kind not in "Mm":
            raise ValueError("Invalid numpy time dtype.")
        return np.array(_integer(value["ticks"]), dtype="i8").view(dtype)[()]
    if kind in ("array", "numpy_scalar") and keys == ({"dtype", "shape", "values"} if kind == "array" else {"dtype", "value"}):
        dtype = _numpy_dtype(value["dtype"])
        if kind == "numpy_scalar":
            budget.allocation(dtype.itemsize)
            return np.array(_decode(value["value"], budget, depth + 1), dtype=dtype)[()]
        shape, values = value["shape"], value["values"]
        if not isinstance(shape, list) or len(shape) > 32 or any(_integer(size) < 0 for size in shape):
            raise ValueError("Invalid array dimensions.")
        count = math.prod(shape)
        if not isinstance(values, list) or count != len(values) or count * dtype.itemsize > MAX_JSON_BYTES:
            raise ValueError("Invalid or oversized array allocation.")
        budget.data(count)
        budget.allocation(count * dtype.itemsize)
        decoded = [_decode(item, budget, depth + 1) for item in values]
        array = np.empty(count, dtype=dtype)
        if dtype.kind == "O":
            for position, item in enumerate(decoded):
                array[position] = item
        else:
            array[:] = decoded
        return array.reshape(shape)
    if kind == "complex" and keys == {"real", "imag"}:
        real, imaginary = _decode(value["real"], budget, depth + 1), _decode(value["imag"], budget, depth + 1)
        if not isinstance(real, Real) or not isinstance(imaginary, Real):
            raise ValueError("Invalid complex value.")
        return complex(real, imaginary)
    if kind == "bytes" and keys == {"value"}:
        return base64.b64decode(value["value"], validate=True)
    if kind == "decimal" and keys == {"value"} and isinstance(value["value"], str):
        return Decimal(value["value"])
    if kind == "fraction" and keys == {"numerator", "denominator"}:
        return Fraction(_integer(value["numerator"]), _integer(value["denominator"]))
    if kind == "datetime" and keys == {"value", "zone", "fold"}:
        stamp = datetime.fromisoformat(value["value"])
        if value["zone"] is not None:
            if stamp.tzinfo is None or not isinstance(value["zone"], str) or len(value["zone"]) > 200:
                raise ValueError("Invalid Python datetime timezone.")
            stamp = stamp.astimezone(ZoneInfo(value["zone"]))
        return stamp.replace(fold=_integer(value["fold"]))
    if kind == "date" and keys == {"value"}:
        return date.fromisoformat(value["value"])
    if kind == "time" and keys == {"value", "fold"}:
        return time.fromisoformat(value["value"]).replace(fold=_integer(value["fold"]))
    if kind == "python_timedelta" and keys == {"days", "seconds", "microseconds"}:
        return timedelta(**{key: _integer(value[key]) for key in keys})
    if kind in ("tuple", "mapping") and keys == {"items"} and isinstance(value["items"], list):
        if kind == "tuple":
            return tuple(_decode(item, budget, depth + 1) for item in value["items"])
        result = {}
        for pair in value["items"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Invalid typed mapping entry.")
            key = _decode(pair[0], budget, depth + 1)
            if key in result:
                raise ValueError("A project mapping contains duplicate keys.")
            result[key] = _decode(pair[1], budget, depth + 1)
        return result
    raise ValueError(f"Unknown or malformed project value tag: {kind!r}.")


def encode_value(value):
    """Convert supported values to JSON-safe structures without losing dtype data."""
    try:
        return _encode(value, _Budget(), 0)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("The project contains unsupported or excessively nested values.") from exc


def decode_value(value):
    """Decode only explicitly allowlisted data types; never import classes or code."""
    try:
        return _decode(value, _Budget(), 0)
    except (TypeError, OverflowError, RecursionError, KeyError, ZeroDivisionError, InvalidOperation, binascii.Error) as exc:
        raise ValueError("The project contains malformed or unsupported values.") from exc


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Project JSON contains duplicate keys.")
        result[key] = value
    return result


def dumps_project(payload: dict) -> bytes:
    """Create bounded data-only archives with an explicit compatibility barrier.

    Version 1 remains readable for generic projects. Fulfillment data requires
    version 2 so older readers cannot silently ignore and discard its ledger.
    """
    if not isinstance(payload, dict):
        raise ValueError("A project payload must be a dictionary.")
    version = FULFILLMENT_FORMAT_VERSION if "fulfillment" in payload else FORMAT_VERSION
    wire_payload = payload
    fulfillment = payload.get("fulfillment")
    if (type(fulfillment) is dict and type(fulfillment.get("version")) is int
            and fulfillment.get("version") == 1 and "format" not in fulfillment):
        from backend.fulfillment_wire import pack_fulfillment

        wire_payload = {**payload, "fulfillment": pack_fulfillment(fulfillment)}
    encoded = {"format": FORMAT, "version": version, "payload": encode_value(wire_payload)}
    data = json.dumps(encoded, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("The project is too large; omit undo snapshots or use a smaller dataset.")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", data)
    result = output.getvalue()
    if len(result) > MAX_ARCHIVE_BYTES:
        raise ValueError("The compressed project exceeds the supported size limit.")
    return result


def loads_project(data: bytes) -> dict:
    """Read one bounded JSON member in memory, without extracting any file."""
    if not isinstance(data, (bytes, bytearray)) or not 0 < len(data) <= MAX_ARCHIVE_BYTES:
        raise ValueError("The project file is empty or exceeds the 50 MiB archive limit.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) != 1 or entries[0].filename != "project.json":
                raise ValueError("A DataWorks project must contain only project.json.")
            info = entries[0]
            if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("Encrypted or unsupported project archives are not accepted.")
            if info.file_size > MAX_JSON_BYTES:
                raise ValueError("The project exceeds the 100 MiB uncompressed limit.")
            with archive.open(info) as member:
                raw = member.read(MAX_JSON_BYTES + 1)
            if len(raw) > MAX_JSON_BYTES or len(raw) != info.file_size:
                raise ValueError("The project has an invalid or oversized JSON member.")
        decoded = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys,
                             parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {token}.")))
        del raw
        if not isinstance(decoded, dict) or set(decoded) != {"format", "version", "payload"}:
            raise ValueError("The file is not a supported DataWorks project.")
        if (decoded["format"] != FORMAT or type(decoded["version"]) is not int
                or decoded["version"] not in (FORMAT_VERSION, FULFILLMENT_FORMAT_VERSION)):
            raise ValueError("The DataWorks project format/version is not supported.")
        envelope_version = decoded["version"]
        payload = decode_value(decoded["payload"])
        del decoded
        if not isinstance(payload, dict):
            raise ValueError("The project payload must be a dictionary.")
        if ("fulfillment" in payload) != (envelope_version == FULFILLMENT_FORMAT_VERSION):
            raise ValueError("The project envelope version does not match its fulfillment data.")
        fulfillment = payload.get("fulfillment")
        if type(fulfillment) is dict and (
                "format" in fulfillment or "sources" in fulfillment
                or "attempts" in fulfillment or fulfillment.get("version") == 2):
            from backend.fulfillment_wire import unpack_fulfillment

            payload["fulfillment"] = unpack_fulfillment(fulfillment)
        return payload
    except (zipfile.BadZipFile, UnicodeError, json.JSONDecodeError, RuntimeError, RecursionError, NotImplementedError, EOFError) as exc:
        raise ValueError("The project file is malformed or unsupported.") from exc
