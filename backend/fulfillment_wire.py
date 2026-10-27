"""Compact, data-only fulfillment archives; public runtime state stays version 1.

Wire version 2 stores exact source bytes and actual attempt evidence once. Its
references never escape loading: each public source/attempt occurrence receives
independent mutable containers. Packing is a read-only serialization view used
only while the codec builds immutable archive bytes.
"""

import copy
import hashlib
import json
import math
import re

from backend import resources
from backend.fulfillment import ingest_csv, plan_digest, validate_plan, validate_source


FORMAT = "dataworks-fulfillment-wire"
VERSION = 2
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_UUID = re.compile(r"[0-9a-f]{32}\Z")
_STATE_FIELDS = {"source", "draft_plan", "last_attempt", "committed"}
_ATTEMPT_FIELDS = {
    "run_id", "source_id", "plan_id", "plan_revision", "plan_digest", "status",
    "engine_version", "accepted", "rejected", "report", "issues",
}
_SOURCE_FIELDS = {
    "format", "version", "source_id", "raw_sha256", "raw_bytes", "read_options",
    "columns", "rows", "row_ids",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, expected, description):
    _require(type(value) is dict and set(value) == set(expected),
             f"The fulfillment {description} has unsupported fields.")


def _identifier(value, pattern, description):
    _require(type(value) is str and bool(pattern.fullmatch(value)),
             f"The fulfillment {description} is invalid.")
    return value


def _json_digest(value):
    """Typed canonical equality without retaining a second large JSON string."""
    from backend.project_io import MAX_DEPTH, MAX_NODES

    stack = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _require(nodes <= MAX_NODES and depth <= MAX_DEPTH,
                 "The fulfillment evidence exceeds the project structure budget.")
        kind = type(item)
        if kind is dict:
            _require(all(type(key) is str for key in item),
                     "Fulfillment evidence needs text JSON keys.")
            stack.extend((child, depth + 1) for child in item.values())
        elif kind is list:
            stack.extend((child, depth + 1) for child in item)
        else:
            _require(kind in (str, int, float, bool, type(None))
                     and (kind is not float or math.isfinite(item)),
                     "Fulfillment evidence must contain finite JSON data.")
    digest = hashlib.sha256()
    try:
        for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False,
                                      sort_keys=True, separators=(",", ":")).iterencode(value):
            digest.update(chunk.encode("utf-8"))
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("Fulfillment evidence must contain supported finite JSON.") from exc
    return digest.hexdigest()


def _source_shape(source):
    _keys(source, _SOURCE_FIELDS, "source")
    _require(source["format"] == "dataworks-csv-source"
             and type(source["version"]) is int and source["version"] == 1,
             "The fulfillment source version is unsupported.")
    _identifier(source["source_id"], _HASH, "source identifier")
    _identifier(source["raw_sha256"], _HASH, "source digest")
    _require(type(source["raw_bytes"]) is bytes,
             "Fulfillment source evidence must preserve original bytes.")
    _keys(source["read_options"], {"encoding", "delimiter"}, "source reading options")
    _require(all(type(source["read_options"][key]) is str for key in ("encoding", "delimiter")),
             "Fulfillment source reading options must be explicit text.")
    _require(type(source["columns"]) is list and all(type(value) is str for value in source["columns"])
             and type(source["rows"]) is list
             and all(type(row) is list and all(type(value) is str for value in row) for row in source["rows"])
             and type(source["row_ids"]) is list and all(type(value) is str for value in source["row_ids"]),
             "Fulfillment source records have unsupported types.")


def _attempt_shape(attempt, source, plan):
    _keys(attempt, _ATTEMPT_FIELDS, "attempt")
    _identifier(attempt["run_id"], _UUID, "run identifier")
    _require(type(attempt["engine_version"]) is int and attempt["engine_version"] == 1,
             "The fulfillment engine version is unsupported.")
    _require(attempt["source_id"] == source["source_id"]
             and attempt["plan_id"] == plan["plan_id"]
             and type(attempt["plan_revision"]) is int
             and attempt["plan_revision"] == plan["revision"]
             and attempt["plan_digest"] == plan_digest(plan),
             "Fulfillment wire references disagree with their source or plan.")
    _require(attempt["status"] in ("completed", "blocked", "decision_required", "failed")
             and type(attempt["issues"]) is list,
             "The fulfillment attempt status or issues are unsupported.")
    if attempt["status"] == "completed":
        _require(type(attempt["accepted"]) is list and type(attempt["rejected"]) is list
                 and type(attempt["report"]) is dict
                 and len(attempt["accepted"]) + len(attempt["rejected"]) == len(source["rows"]),
                 "A completed fulfillment attempt needs its complete output evidence.")
    else:
        _require(all(attempt[field] is None for field in ("accepted", "rejected", "report")),
                 "An unsuccessful fulfillment attempt cannot contain completed outputs.")


def pack_fulfillment(project):
    """Build a compact read-only wire view; do not mutate the runtime project."""
    from backend.fulfillment_state import MAX_HISTORY

    _keys(project, {"version", "history", *_STATE_FIELDS}, "project")
    _require(type(project["version"]) is int and project["version"] == 1,
             "The fulfillment runtime project version is unsupported.")
    _require(type(project["history"]) is list and len(project["history"]) <= MAX_HISTORY,
             "The fulfillment project has too many undo checkpoints.")
    _require(sum(resources.estimate_bytes(snapshot) for snapshot in project["history"])
             <= resources.MAX_HISTORY_BYTES, "Fulfillment undo checkpoints exceed their memory budget.")
    sources, plans, attempts = {}, {}, {}
    canonical_sources = {}
    attempt_digests = {}

    def source_ref(source):
        if source is None:
            return None
        _source_shape(source)
        identifier = source["source_id"]
        if identifier not in canonical_sources:
            canonical_sources[identifier] = validate_source(source)
        _require(source == canonical_sources[identifier],
                 "Repeated fulfillment source identifiers contain inconsistent evidence.")
        sources.setdefault(identifier, {"raw_bytes": source["raw_bytes"],
                                         "read_options": dict(source["read_options"])})
        return identifier

    def plan_ref(plan):
        owned = validate_plan(plan)
        digest = plan_digest(owned)
        if digest in plans:
            _require(owned == plans[digest], "Repeated fulfillment plan digests contain inconsistent plans.")
        else:
            plans[digest] = owned
        return digest

    def attempt_ref(attempt, source_id, digest):
        if attempt is None:
            return None
        _require(source_id is not None, "A fulfillment attempt has no source reference.")
        _attempt_shape(attempt, canonical_sources[source_id], plans[digest])
        identifier = attempt["run_id"]
        fingerprint = _json_digest(attempt)
        if identifier in attempts:
            _require(attempt_digests[identifier] == fingerprint,
                     "Repeated fulfillment run identifiers contain inconsistent evidence.")
        else:
            # The codec reads this view without mutation and returns immutable
            # bytes. Avoid another large output copy during serialization.
            attempts[identifier] = attempt
            attempt_digests[identifier] = fingerprint
        return identifier

    def state_ref(value):
        _keys(value, _STATE_FIELDS, "checkpoint")
        source_id = source_ref(value["source"])
        digest = plan_ref(value["draft_plan"])
        last = attempt_ref(value["last_attempt"], source_id, digest)
        committed = value["committed"]
        if committed is not None:
            _keys(committed, {"source", "plan", "attempt"}, "committed result")
            committed_source = source_ref(committed["source"])
            _require(committed_source is not None, "A committed fulfillment result has no source.")
            committed_plan = plan_ref(committed["plan"])
            committed_attempt = attempt_ref(committed["attempt"], committed_source, committed_plan)
            _require(committed_attempt is not None and attempts[committed_attempt]["status"] == "completed",
                     "Only a completed fulfillment attempt can be committed.")
            committed = {"source": committed_source, "plan": committed_plan, "attempt": committed_attempt}
        if last is not None and attempts[last]["status"] == "completed":
            _require(committed is not None and last == committed["attempt"],
                     "The latest completed attempt must identify the committed result.")
        return {"source": source_id, "draft_plan": digest, "last_attempt": last, "committed": committed}

    current = state_ref({key: project[key] for key in _STATE_FIELDS})
    history = [state_ref(snapshot) for snapshot in project["history"]]
    return {"format": FORMAT, "version": VERSION, "sources": sources, "plans": plans,
            "attempts": attempts, "current": current, "history": history}


def unpack_fulfillment(wire):
    """Inflate strict references to independent public version-1 state values."""
    from backend.fulfillment_state import MAX_HISTORY

    _keys(wire, {"format", "version", "sources", "plans", "attempts", "current", "history"}, "wire payload")
    _require(wire["format"] == FORMAT and type(wire["version"]) is int and wire["version"] == VERSION,
             "The fulfillment wire format/version is unsupported.")
    _require(type(wire["history"]) is list and len(wire["history"]) <= MAX_HISTORY,
             "The fulfillment wire has too many undo checkpoints.")
    tables = {key: wire[key] for key in ("sources", "plans", "attempts")}
    for name, table in tables.items():
        _require(type(table) is dict and len(table) <= 2 * (MAX_HISTORY + 1),
                 f"The fulfillment {name} table exceeds its checkpoint reference budget.")
        for key in table:
            _identifier(key, _UUID if name == "attempts" else _HASH, name + " table identifier")
    used = {key: set() for key in tables}

    def reference(value, name, *, nullable=False):
        if nullable and value is None:
            return None
        _identifier(value, _UUID if name == "attempts" else _HASH, name + " reference")
        _require(value in tables[name], f"The fulfillment wire has a dangling {name} reference.")
        used[name].add(value)
        return value

    def check_state(value):
        _keys(value, _STATE_FIELDS, "wire checkpoint")
        source_id = reference(value["source"], "sources", nullable=True)
        reference(value["draft_plan"], "plans")
        last = reference(value["last_attempt"], "attempts", nullable=True)
        _require(last is None or source_id is not None, "A fulfillment attempt has no source reference.")
        if value["committed"] is not None:
            _keys(value["committed"], {"source", "plan", "attempt"}, "wire committed result")
            for key, table in (("source", "sources"), ("plan", "plans"), ("attempt", "attempts")):
                reference(value["committed"][key], table)

    check_state(wire["current"])
    for snapshot in wire["history"]:
        check_state(snapshot)
    for name, table in tables.items():
        _require(set(table) == used[name], f"The fulfillment wire contains unused {name} entries.")

    sources, plans, attempts = {}, {}, {}
    for identifier, item in tables["sources"].items():
        _keys(item, {"raw_bytes", "read_options"}, "wire source")
        _keys(item["read_options"], {"encoding", "delimiter"}, "source reading options")
        source = ingest_csv(item["raw_bytes"], **item["read_options"])
        _require(source["source_id"] == identifier,
                 "A fulfillment source table key does not match its original bytes and options.")
        sources[identifier] = source
    for digest, item in tables["plans"].items():
        plan = validate_plan(item)
        _require(plan_digest(plan) == digest, "A fulfillment plan table key does not match its plan.")
        plans[digest] = plan
    for identifier, item in tables["attempts"].items():
        _keys(item, _ATTEMPT_FIELDS, "wire attempt")
        _require(item["run_id"] == identifier, "A fulfillment attempt table key does not match its run.")
        source_id, digest = item["source_id"], item["plan_digest"]
        _require(type(source_id) is str and source_id in sources
                 and type(digest) is str and digest in plans,
                 "A fulfillment attempt has dangling source or plan evidence.")
        _attempt_shape(item, sources[source_id], plans[digest])
        _json_digest(item)
        attempts[identifier] = item

    def inflate(value):
        source_id, digest, last = value["source"], value["draft_plan"], value["last_attempt"]
        if last is not None:
            _attempt_shape(attempts[last], sources[source_id], plans[digest])
        committed_ref = value["committed"]
        committed = None
        if committed_ref is not None:
            source_key, plan_key, run_key = (committed_ref[key] for key in ("source", "plan", "attempt"))
            _attempt_shape(attempts[run_key], sources[source_key], plans[plan_key])
            _require(attempts[run_key]["status"] == "completed",
                     "Only a completed fulfillment attempt can be committed.")
            committed = {"source": copy.deepcopy(sources[source_key]),
                         "plan": copy.deepcopy(plans[plan_key]),
                         "attempt": copy.deepcopy(attempts[run_key])}
        if last is not None and attempts[last]["status"] == "completed":
            _require(committed_ref is not None and last == committed_ref["attempt"],
                     "The latest completed attempt must identify the committed result.")
        return {"source": copy.deepcopy(sources[source_id]) if source_id is not None else None,
                "draft_plan": copy.deepcopy(plans[digest]),
                "last_attempt": copy.deepcopy(attempts[last]) if last is not None else None,
                "committed": committed}

    current = inflate(wire["current"])
    history = []
    history_bytes = 0
    for snapshot in wire["history"]:
        owned = inflate(snapshot)
        history_bytes += resources.estimate_bytes(owned)
        _require(history_bytes <= resources.MAX_HISTORY_BYTES,
                 "Inflated fulfillment undo checkpoints exceed their memory budget.")
        history.append(owned)
    return {"version": 1, **current, "history": history}
