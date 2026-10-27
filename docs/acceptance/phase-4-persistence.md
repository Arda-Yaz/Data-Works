# Phase 4: compact evidence and identity guarantees

The initial 100,000-row all-valid diagnostic peaked at 1,011,609,600 bytes while
decoding/validating a project in one process. That was a diagnostic rather than
the mixed F14 workload, but it exposed repeated evidence retention near the
1 GiB ceiling. The changes below reduce duplication without raising limits.

## Archive representation

Public fulfillment project values remain version 1. Their source, draft plan,
last attempt, committed result, and coherent undo snapshots keep the same shape
and independent mutable ownership. The data-only archive codec uses the compact
representation in [fulfillment_wire.py](../../backend/fulfillment_wire.py):

```text
format: dataworks-fulfillment-wire
version: 2
sources: source ID -> {raw_bytes, read_options}
plans: canonical digest -> complete versioned plan
attempts: run UUID -> actual complete attempt evidence
current: source/plan/attempt/committed references
history: bounded coherent reference checkpoints
```

The archive stores original source bytes and explicit parsing options once.
Parsing reconstructs derived header/rows/row IDs, and source identities are checked.
Actual accepted/rejected records, reasons, raw quarantine evidence, duplicate
references, report totals, and attempt status are stored once per run. Loading
does not replace recorded outputs with newly calculated results. Full semantic
validation compares the stored evidence to deterministic execution before active
session replacement; a mismatch fails instead of repairing the recorded result.

Inflation gives every public source/attempt/plan occurrence independent containers.
Mutating a returned latest attempt cannot alter its committed copy or undo state.
The temporary packing view reads owned state while producing immutable bytes;
it does not become a public mutable reference graph.

The compact payload remains inside envelope version 2. Legacy version 1 generic
projects and the earlier noncompact version 2 fulfillment representation still
load. A Phase 3 reader rejects the new wire shape before session mutation; a
pre-fulfillment reader rejects envelope version 2. Unknown formats/versions,
dangling or unused references, mismatched table keys, and unsupported fields fail.
The existing 50 MiB compressed, 100 MiB JSON, depth, node, source, and history
budgets remain unchanged. Wire-table cardinalities are bounded by checkpoint
references, and expanded undo history must fit its byte budget.

## Validation and runtime ownership

A per-validation cache parses each distinct source once. Every repeated occurrence
is still compared with canonical original evidence and independently copied.
Typed canonical JSON SHA-256 fingerprints avoid retaining large serialized
attempt strings in replay/identity caches. They preserve distinctions such as
`true`, `1`, and `1.0` that Python value equality would otherwise collapse.

The same validation context enforces:

- One run UUID has one source/plan/status/output meaning across current state,
  committed state, and retained history. Identical checkpoint repetitions are valid.
- One `(plan_id, revision)` has one canonical digest. An imported known earlier
  revision is valid if its meaning is identical; editing parameters needs a new
  revision. A forged revision cannot hide different rules under an existing label.

History-free export validates the explicitly selected current representation
before copying; it does not replay/copy undo snapshots that will be omitted.
History-inclusive export and all restoration validate the retained history.
Checkpoint budgeting estimates the new snapshot before allocating a full copy;
an oversized checkpoint is pruned according to the existing oldest-first policy.
The current committed result is retained independently of pruned undo history.

Deferred UI exports capture the owned committed run by reference. State transitions
replace that run instead of mutating it, and CSV exporters build new frames.
This avoids copying every output record on each bounded-preview rerun while
keeping download data associated with its original run identity.

## Evidence and scope

Fifteen compact-wire tests and expanded recovery/ownership tests cover legacy
reads, independent copies, typed corruption, inconsistent UUIDs/revisions,
reference tampering, unchanged raw evidence, and failure before session mutation.
The [Phase 4 results](phase-4-results.md) qualify measured workload limits and
remaining environment/customer release gates. Bounded undo can discard older
checkpoints; downloaded archives remain the persistence/restart boundary.
