"""Audit hand-authored Phase 1 expectations; never execute application code.

This checks the synthetic acceptance oracle, not DataWorks implementation.
It reads fixtures and prints JSON, without changing any input or expectation.
"""

from collections import Counter
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from statistics import median, quantiles


ROOT = Path(__file__).resolve().parents[4]
FIXTURES = ROOT / "tests" / "fixtures" / "fulfillment_v1"
STATUS = {
    "shipped": "fulfilled", "dispatched": "fulfilled",
    "complete": "fulfilled", "fulfilled": "fulfilled",
    "open": "pending", "awaiting": "pending", "pending": "pending",
    "canceled": "cancelled", "cancelled": "cancelled",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def records(name):
    with (FIXTURES / name).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def iso_date(value):
    require(bool(re.fullmatch(r"\d{4}([-\/])\d{2}\1\d{2}", value)), "Date format")
    return date.fromisoformat(value.replace("/", "-")).isoformat()


def canonical(row):
    return [row["order_id"], row["line_id"], iso_date(row["order_date"]),
            row["sku"].strip().upper(), int(row["quantity"]),
            STATUS[row["fulfillment_status"].strip().lower()]]


def invalid_reasons(row):
    result = set()
    for column in ("order_id", "line_id", "sku", "fulfillment_status"):
        if not row[column].strip():
            result.add("missing_" + column)
    for column in ("order_id", "line_id"):
        if row[column].strip() and row[column] != row[column].strip():
            result.add("identifier_whitespace")
    if not row["order_date"]:
        result.add("missing_order_date")
    else:
        try:
            iso_date(row["order_date"])
        except (ValueError, AssertionError):
            result.add("invalid_order_date")
    quantity = row["quantity"].strip()
    if not quantity:
        result.add("missing_quantity")
    elif not re.fullmatch(r"[+-]?[0-9]+", quantity):
        result.add("invalid_quantity")
    elif not -(2**63) <= int(quantity) <= 2**63 - 1:
        result.add("quantity_out_of_range")
    elif int(quantity) <= 0:
        result.add("nonpositive_quantity")
    return result


def main():
    expected = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
    checked = []
    for name, outcome in expected["completed_runs"].items():
        columns, source = records(name)
        require(columns == expected["output_columns"], name + ": columns")
        accepted, rejected = outcome["accepted"], outcome["rejected"]
        all_numbers = [item["source_record"] for item in accepted + rejected]
        require(sorted(all_numbers) == list(range(1, len(source) + 1)), name + ": partition")
        require([item["source_record"] for item in accepted] == sorted(item["source_record"] for item in accepted), name + ": accepted order")
        accepted_numbers = {item["source_record"] for item in accepted}
        for item in accepted:
            row = source[item["source_record"] - 1]
            require(not invalid_reasons(row), name + ": accepted invalid row")
            require(item["values"] == canonical(row), name + ": accepted values")
        for item in rejected:
            row = source[item["source_record"] - 1]
            reasons = invalid_reasons(row)
            if "exact_duplicate" in item["reasons"]:
                keeper = item["duplicate_of_source_record"]
                require(keeper in accepted_numbers and keeper < item["source_record"], name + ": duplicate reference")
                require(canonical(row) == canonical(source[keeper - 1]), name + ": canonical duplicate")
                reasons.add("exact_duplicate")
            require(set(item["reasons"]) == reasons, name + ": rejection reasons")
        values = [item["values"] for item in accepted]
        line_counts = Counter(row[5] for row in values)
        quantity_counts = Counter()
        for row in values:
            quantity_counts[row[5]] += row[4]
        actual = {
            "source_rows": len(source), "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "rejection_reason_assignments": sum(len(item["reasons"]) for item in rejected),
            "accepted_distinct_orders": len({row[0] for row in values}),
            "accepted_quantity": sum(row[4] for row in values),
            "accepted_line_counts_by_status": {status: line_counts[status] for status in ("fulfilled", "pending", "cancelled")},
            "accepted_quantity_by_status": {status: quantity_counts[status] for status in ("fulfilled", "pending", "cancelled")},
        }
        require(actual == outcome["report"], name + ": report")
        checked.append({"input": name, "report": actual})

    blockers = expected["noncommitting_attempts"]
    for name, outcome in blockers.items():
        columns, source = records(name)
        require(len(source) == outcome["source_rows"], name + ": source count")
        if outcome["issue"] == "missing_required_column":
            require(sorted(set(expected["output_columns"]) - set(columns)) == outcome["columns"], name)
        elif outcome["issue"] == "unexpected_column":
            require(sorted(set(columns) - set(expected["output_columns"])) == outcome["columns"], name)
        elif outcome["issue"] == "conflicting_duplicate":
            left, right = [canonical(source[number - 1]) for number in outcome["source_records"]]
            require(left[:2] == right[:2] == outcome["key"], name + ": key")
            require([column for column, a, b in zip(columns, left, right) if a != b] == outcome["differing_columns"], name + ": conflict")
        elif outcome["issue"] == "unseen_fulfillment_status":
            value = source[outcome["source_records"][0] - 1]["fulfillment_status"].strip().lower()
            require(value == outcome["normalized_value"] and value not in STATUS, name)

    stats = json.loads((FIXTURES / "statistical_characterization.json").read_text(encoding="utf-8"))
    example = stats["median"]
    for label in ("a", "b"):
        require(median(value for value in example["month_" + label] if value is not None) == example["fitted_" + label], "Median " + label)
    for policy, fitted in (("refit_b", example["fitted_b"]), ("frozen_a_applied_to_b", example["fitted_a"])):
        require([fitted if value is None else value for value in example["month_b"]] == example[policy], policy)
    example = stats["iqr_cap"]
    for label in ("a", "b"):
        q1, _, q3 = quantiles(example["month_" + label], n=4, method="inclusive")
        require([q1, q3] == example["quartiles_" + label], "Quartiles " + label)
        require([q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)] == example["bounds_" + label], "Bounds " + label)
    for policy, label in (("refit_b", "b"), ("frozen_a_applied_to_b", "a")):
        lower, upper = example["bounds_" + label]
        require([min(upper, max(lower, value)) for value in example["month_b"]] == example[policy], policy)

    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        data = (FIXTURES / item["path"]).read_bytes()
        require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], "Manifest " + item["path"])
    print(json.dumps({"status": "passed", "scope": "synthetic oracle audit only; application acceptance not executed", "completed_fixture_audits": checked, "noncommitting_fixture_audits": len(blockers), "statistical_examples": "passed", "manifest_files_verified": len(manifest["files"]), "production_modules_imported": False}, indent=2))


if __name__ == "__main__":
    main()
