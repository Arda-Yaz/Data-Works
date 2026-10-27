"""Versioned, UI-independent execution for monthly fulfillment CSVs.

Sources retain original bytes. Plans contain only supported declarative steps.
Execution returns a complete candidate attempt; session commitment is separate.
"""

from collections import Counter
import copy
import csv
from datetime import date
import hashlib
import io
import json
import re
import sys
from uuid import uuid4

from backend import resources


COLUMNS = ("order_id", "line_id", "order_date", "sku", "quantity", "fulfillment_status")
STATUSES = ("fulfilled", "pending", "cancelled")
ENGINE_VERSION = 1
MAX_PLAN_BYTES = 1024 * 1024
_UUID = re.compile(r"[0-9a-f]{32}\Z")
_DATE = re.compile(r"[0-9]{4}([-\/])[0-9]{2}\1[0-9]{2}\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_MAPPING = {
    "shipped": "fulfilled", "dispatched": "fulfilled",
    "complete": "fulfilled", "fulfilled": "fulfilled",
    "open": "pending", "awaiting": "pending", "pending": "pending",
    "canceled": "cancelled", "cancelled": "cancelled",
}


def _json_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("Expected bounded, finite JSON data.") from exc


def _keys(value, fields, description):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"Unsupported {description} fields.")


def default_plan():
    return {
        "format": "dataworks-fulfillment-plan", "version": 1,
        "plan_id": uuid4().hex, "revision": 1, "statistical_policy": "none",
        "steps": [
            {"kind": "schema", "version": 1, "columns": list(COLUMNS), "extra_columns": "block"},
            {"kind": "normalize", "version": 1,
             "date_formats": ["%Y-%m-%d", "%Y/%m/%d"], "sku_case": "upper",
             "status_mapping": dict(_MAPPING)},
            {"kind": "validate_rows", "version": 1, "quantity": "positive_int64",
             "identifiers": "preserve_or_reject_whitespace", "unknown_status": "decision_required"},
            {"kind": "deduplicate", "version": 1, "key": ["order_id", "line_id"],
             "exact": "keep_first", "conflict": "decision_required"},
            {"kind": "reconcile", "version": 1, "order": "source"},
        ],
    }


def validate_plan(plan):
    _keys(plan, ("format", "version", "plan_id", "revision", "statistical_policy", "steps"), "plan")
    encoded = _json_bytes(plan)
    if len(encoded) > MAX_PLAN_BYTES:
        raise ValueError("The fulfillment plan exceeds the 1 MiB limit.")
    if plan["format"] != "dataworks-fulfillment-plan" or type(plan["version"]) is not int or plan["version"] != 1:
        raise ValueError("Unsupported fulfillment plan format/version.")
    if not isinstance(plan["plan_id"], str) or not _UUID.fullmatch(plan["plan_id"]):
        raise ValueError("A plan needs a canonical UUID identifier.")
    if type(plan["revision"]) is not int or not 1 <= plan["revision"] <= 2**63 - 1:
        raise ValueError("The plan revision must be a positive signed-64 integer.")
    if plan["statistical_policy"] != "none":
        raise ValueError("Statistical operations are unsupported in fulfillment plans.")
    steps = plan["steps"]
    if type(steps) is not list or len(steps) != 5:
        raise ValueError("The plan must contain the five supported ordered steps.")
    templates = default_plan()["steps"]
    for position, (step, template) in enumerate(zip(steps, templates)):
        _keys(step, template, "step")
        if type(step["version"]) is not int or step["version"] != 1:
            raise ValueError("Unsupported operation version.")
        if position != 1:
            if step != template:
                raise ValueError("Unsupported operation parameters or order.")
        else:
            if any(step[key] != template[key] for key in template if key != "status_mapping"):
                raise ValueError("Unsupported normalization parameters.")
            mapping = step["status_mapping"]
            if type(mapping) is not dict or not mapping:
                raise ValueError("A status mapping must be a nonempty dictionary.")
            if any(not isinstance(key, str) or not key or key != key.strip().lower()
                   or not isinstance(value, str) or value not in STATUSES for key, value in mapping.items()):
                raise ValueError("Map normalized status text to a declared canonical status.")
            if any(mapping.get(status) != status for status in STATUSES):
                raise ValueError("Canonical statuses must map to themselves.")
    return copy.deepcopy(plan)


def dumps_plan(plan):
    return _json_bytes(validate_plan(plan))


def plan_digest(plan):
    return hashlib.sha256(dumps_plan(plan)).hexdigest()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not allowed in a plan.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}.")


def loads_plan(data):
    if type(data) is not bytes or len(data) > MAX_PLAN_BYTES:
        raise ValueError("Choose plan bytes within the 1 MiB limit.")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys,
                           parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("The plan is not supported finite JSON.") from exc
    return validate_plan(value)


def revise_plan(plan, *, status_mapping):
    candidate = validate_plan(plan)
    candidate["steps"][1]["status_mapping"] = copy.deepcopy(status_mapping)
    candidate["revision"] += 1
    return validate_plan(candidate)


def ingest_csv(data, *, encoding="utf-8-sig", delimiter=","):
    """Capture raw records and evidence before business transformations.

    Structural parse errors raise before session state can be replaced. Business
    column compatibility belongs to execution, so a drifted header is retained.
    """
    if type(data) is not bytes or len(data) > resources.MAX_UPLOAD_BYTES:
        raise ValueError("Choose CSV bytes within the upload limit.")
    if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in ('"', "\r", "\n", "\0"):
        raise ValueError("Choose one literal CSV delimiter.")
    if not isinstance(encoding, str) or not encoding:
        raise ValueError("Choose an explicit text encoding.")
    options = {"encoding": encoding, "delimiter": delimiter}
    try:
        text = data.decode(encoding)
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
        columns = next((row for row in reader if row), None)
        if not columns or any(not name.strip() for name in columns) or len(set(columns)) != len(columns):
            raise ValueError("CSV headers must be nonempty and unique.")
        if len(columns) > resources.MAX_DATASET_COLUMNS:
            raise ValueError("The CSV exceeds the column budget.")
        rows = []
        storage = len(data) + sum(sys.getsizeof(name) for name in columns)
        if storage > resources.MAX_DATASET_BYTES:
            raise ValueError("The source exceeds the estimated storage budget.")
        for row in reader:
            if not row:
                continue
            if len(row) != len(columns):
                raise ValueError("Every CSV record must have the same field count as the header.")
            count = len(rows) + 1
            if count > resources.MAX_DATASET_ROWS or count * len(columns) > resources.MAX_DATASET_CELLS:
                raise ValueError("The CSV exceeds the row/cell budget.")
            storage += sys.getsizeof(row) + sum(sys.getsizeof(cell) for cell in row) + 128
            if storage > resources.MAX_DATASET_BYTES:
                raise ValueError("The source exceeds the estimated storage budget.")
            rows.append(row)
    except (UnicodeError, LookupError, csv.Error) as exc:
        raise ValueError("The CSV could not be decoded or parsed with these reading options.") from exc
    raw_sha256 = hashlib.sha256(data).hexdigest()
    identity = hashlib.sha256(_json_bytes({"raw_sha256": raw_sha256, "read_options": options})).hexdigest()
    return {"format": "dataworks-csv-source", "version": 1, "source_id": identity,
            "raw_sha256": raw_sha256, "raw_bytes": data, "read_options": options,
            "columns": columns, "rows": rows,
            "row_ids": [f"{identity}:{number}" for number in range(1, len(rows) + 1)]}


def validate_source(source):
    _keys(source, ("format", "version", "source_id", "raw_sha256", "raw_bytes", "read_options", "columns", "rows", "row_ids"), "source")
    if source["format"] != "dataworks-csv-source" or type(source["version"]) is not int or source["version"] != 1:
        raise ValueError("Unsupported source format/version.")
    _keys(source["read_options"], ("encoding", "delimiter"), "reading options")
    restored = ingest_csv(source["raw_bytes"], **source["read_options"])
    if restored != source:
        raise ValueError("Source bytes, parsed records, or identities do not agree.")
    return restored


def _issue(code, stage, columns, **fields):
    return {"code": code, "stage": stage, "columns": list(columns), **fields}


def _canonicalize_row(raw, mapping):
    """Return canonical values, all ordinary row issues, and unknown status."""
    issues = []
    for column in ("order_id", "line_id"):
        value = raw[column]
        if not value.strip():
            issues.append(_issue("missing_" + column, "validate_rows", [column]))
        elif value != value.strip():
            issues.append(_issue("identifier_whitespace", "validate_rows", [column]))
    original_date = raw["order_date"]
    canonical_date = None
    if not original_date.strip():
        issues.append(_issue("missing_order_date", "validate_rows", ["order_date"]))
    else:
        try:
            if not _DATE.fullmatch(original_date):
                raise ValueError("Date lexeme")
            canonical_date = date.fromisoformat(original_date.replace("/", "-")).isoformat()
        except ValueError:
            issues.append(_issue("invalid_order_date", "validate_rows", ["order_date"]))
    sku = raw["sku"].strip().upper()
    if not sku:
        issues.append(_issue("missing_sku", "validate_rows", ["sku"]))
    text = raw["quantity"].strip()
    quantity = None
    if not text:
        issues.append(_issue("missing_quantity", "validate_rows", ["quantity"]))
    elif not _INTEGER.fullmatch(text):
        issues.append(_issue("invalid_quantity", "validate_rows", ["quantity"]))
    else:
        digits = text.lstrip("+-").lstrip("0") or "0"
        if len(digits) > 19:
            issues.append(_issue("quantity_out_of_range", "validate_rows", ["quantity"]))
        else:
            quantity = int(("-" if text.startswith("-") else "") + digits)
            if not -(2**63) <= quantity <= 2**63 - 1:
                issues.append(_issue("quantity_out_of_range", "validate_rows", ["quantity"]))
            elif quantity <= 0:
                issues.append(_issue("nonpositive_quantity", "validate_rows", ["quantity"]))
    normalized_status = raw["fulfillment_status"].strip().lower()
    status = mapping.get(normalized_status)
    unknown = None
    if not normalized_status:
        issues.append(_issue("missing_fulfillment_status", "validate_rows", ["fulfillment_status"]))
    elif status is None:
        unknown = normalized_status
    # Two invalid identifiers can share a reason; keep one reason with both fields.
    merged = {}
    for issue in issues:
        current = merged.setdefault(issue["code"], {**issue, "columns": []})
        current["columns"].extend(issue["columns"])
    return [raw["order_id"], raw["line_id"], canonical_date, sku, quantity, status], list(merged.values()), unknown


def _report(accepted, rejected, count):
    lines = Counter(record["values"][5] for record in accepted)
    quantities = Counter()
    for record in accepted:
        quantities[record["values"][5]] += record["values"][4]
    return {"source_rows": count, "accepted_rows": len(accepted), "rejected_rows": len(rejected),
            "rejection_reason_assignments": sum(len(record["reasons"]) for record in rejected),
            "accepted_distinct_orders": len({record["values"][0] for record in accepted}),
            "accepted_quantity": sum(quantities.values()),
            "accepted_line_counts_by_status": {status: lines[status] for status in STATUSES},
            "accepted_quantity_by_status": {status: quantities[status] for status in STATUSES}}


def execute(source, plan):
    """Compute a complete attempt; never commit or mutate source/plan/session."""
    source, plan = validate_source(source), validate_plan(plan)
    attempt = {"run_id": uuid4().hex, "engine_version": ENGINE_VERSION,
               "source_id": source["source_id"], "plan_id": plan["plan_id"],
               "plan_revision": plan["revision"], "plan_digest": plan_digest(plan),
               "status": "blocked", "accepted": None, "rejected": None, "report": None, "issues": []}
    stage = "schema"
    try:
        missing = [column for column in COLUMNS if column not in source["columns"]]
        extra = [column for column in source["columns"] if column not in COLUMNS]
        if missing:
            attempt["issues"].append(_issue("missing_required_column", stage, missing))
        if extra:
            attempt["issues"].append(_issue("unexpected_column", stage, extra))
        if missing or extra:
            return attempt
        stage = "validate_rows"
        valid, rejected, unknowns = [], [], {}
        mapping = plan["steps"][1]["status_mapping"]
        for number, row in enumerate(source["rows"], 1):
            raw = dict(zip(source["columns"], row))
            values, issues, unknown = _canonicalize_row(raw, mapping)
            if unknown is not None:
                unknowns.setdefault(unknown, []).append(number)
            if issues:
                rejected.append({"source_record": number, "row_id": source["row_ids"][number - 1],
                                 "raw": raw, "reasons": [issue["code"] for issue in issues], "issues": issues})
            elif unknown is None:
                valid.append({"source_record": number, "row_id": source["row_ids"][number - 1], "values": values})
        if unknowns:
            attempt["status"] = "decision_required"
            attempt["issues"] = [_issue("unseen_fulfillment_status", stage, ["fulfillment_status"],
                                        normalized_value=value, source_records=numbers)
                                 for value, numbers in unknowns.items()]
            return attempt
        stage = "deduplicate"
        groups = {}
        for record in valid:
            groups.setdefault(tuple(record["values"][:2]), []).append(record)
        accepted, conflicts = [], []
        for key, records in groups.items():
            first = records[0]
            differing = [COLUMNS[position] for position in range(len(COLUMNS))
                         if any(record["values"][position] != first["values"][position] for record in records[1:])]
            if differing:
                conflicts.append(_issue("conflicting_duplicate", stage, differing, key=list(key),
                                        source_records=[record["source_record"] for record in records],
                                        differing_columns=differing))
                continue
            accepted.append(first)
            for record in records[1:]:
                number = record["source_record"]
                rejected.append({"source_record": number, "row_id": record["row_id"],
                                 "raw": dict(zip(source["columns"], source["rows"][number - 1])),
                                 "reasons": ["exact_duplicate"],
                                 "issues": [_issue("exact_duplicate", stage, ["order_id", "line_id"])],
                                 "duplicate_of_source_record": first["source_record"],
                                 "duplicate_of_row_id": first["row_id"]})
        if conflicts:
            attempt["status"] = "decision_required"
            attempt["issues"] = conflicts
            return attempt
        stage = "reconcile"
        accepted.sort(key=lambda record: record["source_record"])
        rejected.sort(key=lambda record: record["source_record"])
        identities = [record["row_id"] for record in accepted + rejected]
        if len(identities) != len(source["rows"]) or len(set(identities)) != len(identities) or set(identities) != set(source["row_ids"]):
            raise ValueError("Source row reconciliation failed.")
        report = _report(accepted, rejected, len(source["rows"]))
        attempt.update(status="completed", accepted=accepted, rejected=rejected, report=report)
        return attempt
    except Exception as exc:
        attempt.update(status="failed", accepted=None, rejected=None, report=None,
                       issues=[_issue("execution_failed", stage, [], error_type=type(exc).__name__)])
        return attempt
