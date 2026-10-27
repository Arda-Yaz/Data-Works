# Phase 3: synthetic fulfillment workflow results

Date: 2026-10-26. The first complete fulfillment workflow is implemented and
verified against the frozen synthetic expectations. Phases 2 and 3 were approved
together; work pauses before Phase 4. Production release approval remains withheld.

## Implemented behavior

- [Execution engine](../../backend/fulfillment.py): owned raw-byte CSV sources,
  source-scoped row IDs, strict versioned JSON plans, explicit column/date/status/
  quantity policies, complete candidate execution, and reconciled report totals.
- [Fulfillment state](../../backend/fulfillment_state.py): separate current source,
  draft plan, attempt, and committed result; coherent bounded undo; atomic
  commitment; strict restoration validation of persisted evidence and associations.
- [Session integration](../../backend/state.py): fulfillment-only and mixed
  `.dataworks` projects using the existing bounded, data-only codec. Old projects
  without fulfillment data still open. Any present invalid extension fails before
  active state replacement. Recorded attempts invalidate prepared archive bytes.
  Fulfillment archives use envelope version 2 so old version 1 builds reject them
  explicitly instead of silently discarding their evidence when resaving.
- [Workflow page](../../frontend/pages/data_fulfillment.py): explicit file reading,
  saved-plan import/download, strict mapping revisions, run/undo, latest-attempt
  status, correctly attributed earlier results, bounded previews, accepted and
  quarantine CSV exports, and a provenance-bearing JSON report.
- [Project page](../../frontend/pages/data_project.py): save/reopen with or without
  undo snapshots, including fulfillment-only projects with no generic DataFrame.
- [Independent preparation](../../backend/dataset_preparation.py): existing generic
  schema/replay preparation no longer imports the Streamlit state adapter.
- Dataset clearing preserves the independent fulfillment workspace and project
  name. Explicit whole-project clearing retains its existing separate behavior.

No statistical imputation or outlier removal is allowed in the fulfillment plan.
Quantity parsing retains signed-64 precision; report sums use Python integers.
Cancelled lines remain accepted, exact canonical duplicates identify their keeper,
and conflicting duplicates/unseen statuses require explicit decisions.

## Actual synthetic results

The implementation outputs match the independently authored Phase 1 oracle:

| Input | Source lines | Accepted | Quarantined | Reason assignments | Accepted orders | Accepted quantity |
|---|---:|---:|---:|---:|---:|---:|
| Month A | 12 | 4 | 8 | 8 | 3 | 5 |
| Month B | 8 | 5 | 3 | 3 | 4 | 11 |
| Month C invalid rows | 9 | 2 | 7 | 9 | 2 | 3 |

Missing required quantity and unexpected warehouse columns block the attempt.
Conflicting quantities for one order-line key and unseen `backordered` status
require decisions. Each preserves the preceding committed B result without
presenting partial outputs as a completed run. Injected failures after processing
an initial row and at final reporting likewise retain preceding valid results.

Every completed source record has exactly one accepted/quarantined outcome;
the C record with three reasons is counted once. Rejected CSVs preserve original
values, all reasons/stages, source-row identity, and exact-duplicate references.
Accepted CSVs preserve leading-zero IDs and maximal signed-64 quantities.

## Verification and independent reproduction

See [the evidence index](evidence/phase-3/README.md) for full command logs, timings,
test counts, retained project files, source hashes, and remaining limits.
The full regression/Streamlit suite, syntax checks, dependency compatibility,
and independent frozen-fixture audit pass. No skipped or expected-failure tests
substitute for the new implementation. Existing ingestion/cleaning/EDA/ML/chart/
project/resource regressions remain included.

[The reproducible runner](../../scripts/verify_fulfillment.py) writes a real saved
plan and real `.dataworks` archives, checks A, reopens A in another process, uses
its unchanged plan on B, and reopens B in a third process. Saved run IDs, source
identities, reports, row evidence, and plan digest survive reopening. The backend
child processes do not import Streamlit. It also verifies all C dispositions and
prior-result preservation. The separate F02 test imports the standalone saved
plan file into a fresh process before processing B.

Use Python 3.12 from the repository root and a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py
.\.venv\Scripts\python.exe scripts/verify_fulfillment.py --output-dir .test-tmp/fulfillment-proof
```

## Issues found and corrected

Verification and independent review caught and fixed header-only source budget
enforcement, Python equality accepting altered integer/boolean/float evidence,
completed-attempt association with an absent or older commitment, and the generic
dataset-clear action erasing fulfillment state. Regression checks cover these
invariants. Archive version guarding also prevents older readers from silently
discarding new evidence in mixed projects. Mapping imports reject duplicate JSON keys/nonfinite values, and
mixed duplicate-reference previews preserve integral references as text.

The isolated virtual environment lacks its own pip module. System pip explicitly
targeting that interpreter reports no broken requirements; no packages changed.
Focused fresh-process tests needed the existing runner's repository-local scratch
directory under this sandbox. The final suite uses that supported runner.

## Remaining gates

This evidence covers a narrow synthetic workflow on the current Windows/Python
3.12 environment. It does not establish customer-held-out compatibility, target
workload runtime/peak memory, Linux execution, real dashboard import, unaided
operator usability, preparation-time savings, or shared-host OIDC/TLS operation.
Synthetic Month B was authored alongside its expected outputs; it is not a real
unseen customer export.

Phase 4 must broaden reuse/recovery/resource evidence at the declared workload.
The first release is still proposed as local single-user use with retained
downloaded projects. Server durability/autosave is not implemented. Generic
statistical recipe reuse still requires its own explicit fitted/refit policy;
the fulfillment plan rejects those unsupported steps rather than implying them.
Real operator confirmation, two permissioned consecutive exports, the dashboard
contract, licensing, and observed value remain open before release claims.
