"""Owned fulfillment project transitions, independent of Streamlit session state.

Only a completed, validated attempt replaces committed outputs. Original source
and plan travel with the committed result so later drafts cannot relabel it.
"""

import copy
from collections import Counter
import hashlib
import json
import re
from uuid import uuid4

from backend import resources
from backend.fulfillment import COLUMNS, STATUSES, default_plan, execute, plan_digest, validate_plan, validate_source


MAX_HISTORY = 10
_STATE_FIELDS = {"source", "draft_plan", "last_attempt", "committed"}
_PROJECT_FIELDS = {"version", "history", *_STATE_FIELDS}
_ATTEMPT_FIELDS = {
    "run_id", "source_id", "plan_id", "plan_revision", "plan_digest", "status",
    "engine_version", "accepted", "rejected", "report", "issues",
}


def new_project():
    """Create independent empty fulfillment state; this does not load a source."""
    return {"version": 1, "source": None, "draft_plan": default_plan(),
            "last_attempt": None, "committed": None, "history": []}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _plain_json(value, field):
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError(f"Fulfillment {field} must contain finite JSON data.") from exc


def _semantic(attempt):
    # Python equality treats True == 1 == 1.0; persisted evidence must retain
    # JSON types as well as values, especially identifiers, counts and quantities.
    encoded = json.dumps({key: value for key, value in attempt.items() if key != "run_id"},
                         ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _owned_source(source, context):
    """Parse each distinct source once; still compare every saved occurrence."""
    _require(type(source) is dict, "A fulfillment source must be structured data.")
    identity = source.get("source_id")
    _require(type(identity) is str, "A fulfillment source needs an identifier.")
    cache = context["sources"]
    if identity not in cache:
        canonical = validate_source(source)
        cache[identity] = canonical
        return canonical
    canonical = cache[identity]
    _require(type(source.get("version")) is int
             and type(source.get("raw_bytes")) is bytes and source == canonical,
             "Repeated source identities do not preserve the same original evidence.")
    return copy.deepcopy(canonical)


def _owned_plan(plan, context):
    candidate = validate_plan(plan)
    key = (candidate["plan_id"], candidate["revision"])
    digest = plan_digest(candidate)
    known = context["plans"].setdefault(key, digest)
    _require(known == digest, "A plan identity/revision cannot describe different parameters.")
    return candidate


def _completed_outputs(attempt, source):
    """Check typed final invariants without re-running previous transformations."""
    accepted, rejected, report = attempt["accepted"], attempt["rejected"], attempt["report"]
    _require(type(accepted) is list and type(rejected) is list and type(report) is dict,
             "A completed fulfillment attempt needs its outputs and report.")
    identities = []
    for record in accepted + rejected:
        _require(type(record) is dict and type(record.get("source_record")) is int
                 and 1 <= record["source_record"] <= len(source["rows"]),
                 "A fulfillment output has an invalid source record.")
        number = record["source_record"]
        _require(record.get("row_id") == source["row_ids"][number - 1],
                 "A fulfillment output has an invalid row identity.")
        identities.append(record["row_id"])
    _require(len(set(identities)) == len(identities)
             and set(identities) == set(source["row_ids"]),
             "Fulfillment outputs do not partition source rows.")
    _require([row["source_record"] for row in accepted] == sorted(row["source_record"] for row in accepted)
             and [row["source_record"] for row in rejected] == sorted(row["source_record"] for row in rejected),
             "Fulfillment output order does not match the source.")
    for record in accepted:
        values = record.get("values")
        _require(set(record) == {"source_record", "row_id", "values"}
                 and type(values) is list and len(values) == len(COLUMNS)
                 and all(type(values[index]) is str for index in (0, 1, 2, 3, 5))
                 and type(values[4]) is int and 1 <= values[4] <= 2**63 - 1
                 and values[5] in STATUSES,
                 "An accepted fulfillment record has invalid values or types.")
    accepted_numbers = {row["source_record"] for row in accepted}
    for record in rejected:
        base = {"source_record", "row_id", "raw", "reasons", "issues"}
        duplicate = {"duplicate_of_source_record", "duplicate_of_row_id"}
        _require(set(record) in (base, base | duplicate), "A quarantined record has unsupported fields.")
        number = record["source_record"]
        _require(type(record["raw"]) is dict
                 and record["raw"] == dict(zip(source["columns"], source["rows"][number - 1])),
                 "Quarantine evidence does not preserve its raw source record.")
        reasons, issues = record["reasons"], record["issues"]
        _require(type(reasons) is list and bool(reasons)
                 and all(type(reason) is str for reason in reasons)
                 and len(set(reasons)) == len(reasons)
                 and type(issues) is list and len(issues) == len(reasons),
                 "A quarantined record needs distinct attributed reasons.")
        for issue in issues:
            _require(type(issue) is dict and set(issue) == {"code", "stage", "columns"}
                     and issue.get("code") in reasons
                     and issue.get("stage") in ("validate_rows", "deduplicate")
                     and type(issue.get("columns")) is list and bool(issue["columns"])
                     and all(type(column) is str and column in COLUMNS for column in issue["columns"]),
                     "A quarantine issue has unsupported fields or attribution.")
        _require({issue["code"] for issue in issues} == set(reasons), "Quarantine reasons and issues disagree.")
        if duplicate.issubset(record):
            keeper = record["duplicate_of_source_record"]
            _require(type(keeper) is int and keeper in accepted_numbers and keeper < number
                     and record["duplicate_of_row_id"] == source["row_ids"][keeper - 1]
                     and reasons == ["exact_duplicate"],
                     "An exact duplicate does not identify its retained source row.")
    lines = Counter(row["values"][5] for row in accepted)
    quantities = Counter()
    for row in accepted:
        quantities[row["values"][5]] += row["values"][4]
    expected = {
        "source_rows": len(source["rows"]), "accepted_rows": len(accepted),
        "rejected_rows": len(rejected), "rejection_reason_assignments": sum(len(row["reasons"]) for row in rejected),
        "accepted_distinct_orders": len({row["values"][0] for row in accepted}),
        "accepted_quantity": sum(quantities.values()),
        "accepted_line_counts_by_status": {status: lines[status] for status in STATUSES},
        "accepted_quantity_by_status": {status: quantities[status] for status in STATUSES},
    }
    _require(json.dumps(report, sort_keys=True, allow_nan=False)
             == json.dumps(expected, sort_keys=True, allow_nan=False),
             "Fulfillment report counts, quantities, or types do not reconcile.")


def _validate_attempt(attempt, source, plan, context):
    _require(isinstance(attempt, dict) and set(attempt) == _ATTEMPT_FIELDS,
             "A fulfillment attempt has invalid fields.")
    _require(isinstance(attempt["run_id"], str) and bool(re.fullmatch(r"[0-9a-f]{32}", attempt["run_id"])),
             "A fulfillment attempt needs a valid run identifier.")
    _require(type(attempt["engine_version"]) is int and attempt["engine_version"] == 1,
             "The fulfillment engine version is unsupported.")
    _require(attempt["source_id"] == source["source_id"]
             and attempt["plan_id"] == plan["plan_id"]
             and type(attempt["plan_revision"]) is int
             and attempt["plan_revision"] == plan["revision"]
             and attempt["plan_digest"] == plan_digest(plan),
             "Fulfillment attempt source or plan associations do not match.")
    _require(attempt["status"] in ("completed", "blocked", "decision_required", "failed"),
             "The fulfillment attempt status is unsupported.")
    _require(isinstance(attempt["issues"], list), "Fulfillment attempt issues must be a list.")
    for issue in attempt["issues"]:
        _require(isinstance(issue, dict)
                 and isinstance(issue.get("code"), str) and bool(issue["code"])
                 and isinstance(issue.get("stage"), str) and bool(issue["stage"]),
                 "Fulfillment issues need a code and stage.")
    if attempt["status"] == "completed":
        _require(attempt["issues"] == [], "A completed attempt cannot have unresolved execution issues.")
        _completed_outputs(attempt, source)
    else:
        _plain_json(attempt, "attempt")
        _require(all(attempt[field] is None for field in ("accepted", "rejected", "report")),
                 "An uncommitted attempt cannot contain completed outputs.")
        _require(bool(attempt["issues"]), "An unsuccessful attempt needs an issue.")
    fingerprint = _semantic(attempt)
    known = context["runs"].setdefault(attempt["run_id"], fingerprint)
    _require(known == fingerprint, "A run identifier cannot describe different attempts.")
    replay_cache = context["replays"]
    if attempt["status"] == "failed":
        _require(len(attempt["issues"]) == 1
                 and all(set(issue) == {"code", "stage", "columns", "error_type"}
                     and issue["code"] == "execution_failed"
                     and issue["stage"] in ("schema", "normalize", "validate_rows", "deduplicate", "reconcile", "execute")
                     and issue["columns"] == []
                     and isinstance(issue.get("error_type"), str)
                     and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", issue["error_type"]))
                     for issue in attempt["issues"]),
                 "A failed attempt must record its exception class.")
    elif replay_cache is not None:
        # All deterministic business evidence must agree with source+plan.
        # Reuse replay work for repeated snapshots in the same validation call.
        key = (source["source_id"], plan_digest(plan))
        if key not in replay_cache:
            replay_cache[key] = _semantic(execute(source, plan))
        _require(fingerprint == replay_cache[key],
                 "Fulfillment outputs or validation evidence do not match their source and plan.")
    return copy.deepcopy(attempt)


def _validate_state(value, context):
    _require(isinstance(value, dict) and set(value) == _STATE_FIELDS,
             "A fulfillment checkpoint has invalid fields.")
    plan = _owned_plan(value["draft_plan"], context)
    source = _owned_source(value["source"], context) if value["source"] is not None else None
    attempt = value["last_attempt"]
    if attempt is not None:
        _require(source is not None, "A fulfillment attempt has no current source.")
        attempt = _validate_attempt(attempt, source, plan, context)
    committed = value["committed"]
    if committed is not None:
        _require(isinstance(committed, dict) and set(committed) == {"source", "plan", "attempt"},
                 "A committed fulfillment result has invalid fields.")
        committed_source = _owned_source(committed["source"], context)
        committed_plan = _owned_plan(committed["plan"], context)
        committed_attempt = _validate_attempt(committed["attempt"], committed_source, committed_plan, context)
        _require(committed_attempt["status"] == "completed",
                 "Only a completed fulfillment result can be committed.")
        committed = {"source": committed_source, "plan": committed_plan, "attempt": committed_attempt}
    if attempt is not None and attempt["status"] == "completed":
        _require(committed is not None
                 and attempt["run_id"] == committed["attempt"]["run_id"]
                 and _semantic(attempt) == _semantic(committed["attempt"]),
                 "The latest completed attempt does not match the committed result.")
    return {"source": source, "draft_plan": plan, "last_attempt": attempt, "committed": committed}


def _validated_project(project, *, replay, context=None):
    _require(isinstance(project, dict) and set(project) == _PROJECT_FIELDS,
             "The fulfillment project has invalid fields.")
    _require(type(project["version"]) is int and project["version"] == 1,
             "The fulfillment project version is unsupported.")
    history = project["history"]
    _require(isinstance(history, list) and len(history) <= MAX_HISTORY,
             f"A fulfillment project may have at most {MAX_HISTORY} undo checkpoints.")
    _require(sum(resources.estimate_bytes(snapshot) for snapshot in history) <= resources.MAX_HISTORY_BYTES,
             "Fulfillment undo checkpoints exceed the history memory limit.")
    if context is None:
        context = {}
    context.update(sources={}, plans={}, runs={}, replays={} if replay else None)
    current = _validate_state({key: project[key] for key in _STATE_FIELDS}, context)
    snapshots = [_validate_state(snapshot, context) for snapshot in history]
    return {"version": 1, **current, "history": snapshots}


def validate_project(project):
    """Reproduce persisted evidence before returning an owned candidate project."""
    return _validated_project(project, replay=True)


def _snapshot(project):
    return copy.deepcopy({key: project[key] for key in _STATE_FIELDS})


def _checkpoint(candidate):
    # Avoid allocating a full checkpoint that the byte budget immediately drops.
    view = {key: candidate[key] for key in _STATE_FIELDS}
    size = resources.estimate_bytes(view)
    if size > resources.MAX_HISTORY_BYTES:
        candidate["history"] = []
        return
    retained = list(candidate["history"][-(MAX_HISTORY - 1):])
    sizes = [resources.estimate_bytes(snapshot) for snapshot in retained]
    while retained and sum(sizes) + size > resources.MAX_HISTORY_BYTES:
        retained.pop(0)
        sizes.pop(0)
    candidate["history"] = [*retained, _snapshot(candidate)]


def set_source(project, source):
    candidate = _validated_project(project, replay=False)
    owned_source = validate_source(source)
    if candidate["source"] != owned_source:
        _checkpoint(candidate)
        candidate["source"] = owned_source
        candidate["last_attempt"] = None
    return candidate


def set_plan(project, plan):
    context = {}
    candidate = _validated_project(project, replay=False, context=context)
    owned_plan = _owned_plan(plan, context)
    if candidate["draft_plan"] != owned_plan:
        _checkpoint(candidate)
        candidate["draft_plan"] = owned_plan
        candidate["last_attempt"] = None
    return candidate


def run_project(project):
    context = {}
    candidate = _validated_project(project, replay=False, context=context)
    source, plan = candidate["source"], candidate["draft_plan"]
    _require(source is not None, "Load a fulfillment source before running the plan.")
    try:
        attempt = execute(source, plan)
    except Exception as exc:
        # Do not expose exception messages, which can contain source values.
        attempt = {
            "run_id": uuid4().hex, "source_id": source["source_id"],
            "plan_id": plan["plan_id"], "plan_revision": plan["revision"],
            "plan_digest": plan_digest(plan), "engine_version": 1,
            "status": "failed", "accepted": None, "rejected": None, "report": None,
            "issues": [{"code": "execution_failed", "stage": "execute", "columns": [], "error_type": type(exc).__name__}],
        }
    # An attempt returned by the engine is owned and associated with this exact
    # source/plan; deterministic checks are performed when persisted data loads.
    attempt = _validate_attempt(attempt, source, plan, context)
    if attempt["status"] == "completed":
        _checkpoint(candidate)
        candidate["committed"] = {"source": copy.deepcopy(source), "plan": copy.deepcopy(plan),
                                  "attempt": copy.deepcopy(attempt)}
    candidate["last_attempt"] = copy.deepcopy(attempt)
    return candidate


def undo_project(project):
    candidate = _validated_project(project, replay=False)
    if candidate["history"]:
        restored = candidate["history"].pop()
        candidate.update(restored)
    return candidate


def export_project(project, include_history=True):
    _require(type(include_history) is bool, "The history export option must be boolean.")
    _require(type(project) is dict and set(project) == _PROJECT_FIELDS,
             "The fulfillment project has invalid fields.")
    # Explicitly omitted history is not part of the exported representation.
    selected = project if include_history else {**project, "history": []}
    return validate_project(selected)


def current_result(project):
    """Whether the committed success belongs to the current source and plan."""
    committed = project.get("committed")
    source = project.get("source")
    return bool(committed is not None and source is not None
                and committed["source"]["source_id"] == source["source_id"]
                and plan_digest(committed["plan"]) == plan_digest(project["draft_plan"]))
