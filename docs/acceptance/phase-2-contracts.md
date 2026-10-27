# Phase 2: fulfillment execution and persistence contracts

Date: 2026-10-26. Scope: the local, single-user workflow in [Phase 1](phase-1.md).
The owner approved Phases 2 and 3 together. Phase 2 defines the contracts and
executable checks; Phase 3 supplies the working implementation. Real customer
data, dashboard compatibility, observed value, and workload sign-off remain open.

## Ownership and identity

Source evidence, a draft plan, a run attempt, and a committed result are distinct
owned values. Pure backend functions never mutate their arguments. Streamlit
adapters prepare a complete replacement before assigning session state.

A source contains exact `raw_bytes`, explicit encoding/delimiter, parsed header,
raw string records, raw-file SHA-256, and a source ID derived from the raw bytes
and canonical read options. Identical bytes with identical options identify the
same source; different options or bytes identify a different source. Filenames
and pandas index labels are not identity. Row IDs are `<source_id>:<ordinal>`,
using 1-based parsed record ordinals excluding the header. Quoted newlines do not
change this record-based convention. Empty physical CSV records are ignored;
records of empty cells remain source records. Preserve duplicate business keys.

Malformed encoding/CSV, missing/duplicate header names, unequal field counts, and
resource limits fail source ingestion before replacing current state. Valid CSV
with missing or extra business columns is a blocked execution attempt. The same
central upload/row/column/cell budgets apply as to existing application ingestion.

## Plan version 1

The saved JSON plan is `dataworks-fulfillment-plan`, version 1, with a UUID
`plan_id`, positive integer `revision`, `statistical_policy: "none"`, and five
ordered, versioned steps. JSON keys and operation parameters are strict: unknown
fields, unsupported versions/steps, duplicate JSON keys, nonfinite numbers, and
executable/custom values are rejected. Plan JSON is bounded to 1 MiB.

| Ordered step | Persisted behavior |
|---|---|
| `schema` | Exact six columns from Phase 1; extra/missing columns block; source column order may differ. |
| `normalize` | Explicit `%Y-%m-%d` and `%Y/%m/%d` date formats; trim/uppercase SKU; trim/lowercase status and exact alias mapping. |
| `validate_rows` | Preserve identifiers or reject surrounding whitespace; positive signed-64 integer quantities; missing/invalid values quarantine; unseen nonempty status requires a decision. |
| `deduplicate` | Key `(order_id,line_id)`; canonical exact duplicates keep first with traceable quarantine; conflicting payloads require a decision. |
| `reconcile` | Accepted rows stay in source order; completed outputs partition all source IDs and reconcile report totals. |

Date lexemes must match an enabled format exactly before calendar validation.
Quantity parsing must avoid float conversion and precision loss. All independent
row reasons are retained. Unknown status decisions take precedence even on a row
with ordinary row errors; no completed output is committed in that attempt.
Duplicate conflicts are assessed only among otherwise valid canonical rows.

Canonical JSON with sorted keys, UTF-8, finite values, and compact separators
defines the SHA-256 plan digest. No timestamps, widget state, input filename, or
learned statistic enters the plan. Revising a supported mapping preserves plan
ID, increments revision, and changes digest. Importing a saved plan never silently
rewrites its parameters, ID, or revision. Generic statistical recipes remain
separate; unsupported statistical steps cannot enter this fulfillment plan.

## Attempt and commitment

Every attempt records a fresh run UUID, engine version, source ID, plan ID/revision
and digest, status, and stage-attributed issues. Terminal states are:

- `completed`: exact accepted records, rejected raw records/reasons/row IDs,
  duplicate references, and final report exist. Final validation succeeded.
- `blocked`: structural compatibility prevents execution.
- `decision_required`: unseen status or conflicting duplicates require an
  attributed plan/source change followed by a new attempt.
- `failed`: unexpected execution failure; record stage and exception class,
  without exposing a cell-bearing exception message as an application error.

Noncompleted attempts have `accepted`, `rejected`, and `report` set to `None`.
They cannot masquerade as a completed zero-row output. Business exclusions are
row rejections, not failed runs. Completed reports include the exact quantities,
status subtotals, line counts, reason assignments, and distinct accepted orders
specified by the independently authored Phase 1 oracle.

Compute the full candidate, ledger, and report before commitment. A blocked,
decision-required, or failed attempt updates diagnostic attempt state but retains
the preceding committed result with its own source and plan. A result is current
only when its source ID and plan digest match current source/draft plan; display
latest attempt status separately. A failed rerun must not relabel a previous
success as belonging to the failed attempt.

## Project extension and recovery

Reuse the bounded, data-only `.dataworks` JSON/ZIP codec. Existing generic
archives retain envelope version 1. Archives containing fulfillment evidence
require envelope version 2 and an independently versioned `fulfillment` payload:

```text
version: 1
source: current source or null
draft_plan: validated plan
last_attempt: latest attempt or null
committed: {source, plan, attempt} or null
history: bounded coherent snapshots, each excluding nested history
```

A fulfillment-only project uses an explicitly discriminated payload mode rather
than fabricating a generic dataset. Mixed projects retain existing generic
dataset/experiment/chart fields plus the extension. Loading an older project
without the extension creates empty fulfillment state. A present but invalid
extension must fail before any active session mutation; it cannot be ignored.
No migration invents historical row reasons, fitted values, or run evidence.

Version 2 is necessary because the old version 1 application would silently
ignore added fulfillment evidence in mixed projects and could discard it on
resaving. Its existing exact-version check rejects version 2. The new reader
accepts legacy version 1 and fulfillment version 2, rejects fulfillment payloads
mislabelled as version 1, and rejects version 2 without the extension. This
compatibility guard refines the initial optional-extension design during Phase 3;
the typed codec and legacy source data representation are unchanged.

Before restoration, recompute source identity from original bytes/options, check
the strict plan and digest, verify attempt/source/plan associations, row partition,
raw rejection evidence, duplicate references, and report semantics. Deterministic
re-execution may check completed evidence; persisted run UUID remains unchanged.
Unsupported future project/extension/engine versions fail clearly and atomically.

Source changes, plan changes, and successful commitments checkpoint coherent
source/plan/attempt/committed state. Undo restores one checkpoint; failed attempts
do not manufacture new successful checkpoints. Count/byte budgets match current
undo limits (10 snapshots and 128 MiB), releasing oldest entries first. Exporting
without history preserves current evidence but removes undo capability.

Any archive-visible state change increments the project revision and invalidates
prepared project bytes. Thus an unsuccessful attempt preserves committed results
but requires a newly prepared archive to include the attempt. Preparing bytes,
downloading/retaining an archive, and successfully reopening it are distinct.
No server durability or crash recovery is claimed.

## Public backend boundary and checks

`backend.fulfillment` owns `default_plan`, `validate_plan`, `revise_plan`,
`plan_digest`, `dumps_plan`, `loads_plan`, `ingest_csv`, `validate_source`, and
`execute`. It has no Streamlit dependency. Accepted records expose ordinal,
source-row ID, and canonical six-field values. Rejected records expose ordinal,
row ID, raw six-field dictionary, reason codes, attributed issues, and duplicate
references when applicable. Attempt issues use `code`, `stage`, and relevant
`columns`, `source_records`, `key`, `differing_columns`, or `normalized_value`.

`backend.fulfillment_state` owns pure project transitions/validation. Existing
`StateManager` adapts those transitions to Streamlit and existing archive export/
restore. Shared dataset preparation moves out of the Streamlit-backed state
module; generic replay retains its current behavior without importing Streamlit.

Executable acceptance tests consume the frozen Phase 1 fixtures directly. During
Phase 2 they must fail honestly while the new engine is absent; no skip,
`expectedFailure`, fixture-derived production output, or fallback mock substitutes
for the implementation. Phase 3 must turn them green and retain the existing
regression suite. Persistence, fresh-process reuse, interaction tests, and failure
injection complement the fixture tests. Phase 4 retains broader workload and
recovery proof; this contract alone is not release evidence.
