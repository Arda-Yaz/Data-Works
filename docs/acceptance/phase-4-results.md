# Phase 4: reuse, recovery, and measured workload results

Date: 2026-10-27. Engineering verification passed on the recorded installed
Windows/Python environment. The owner requested a pause before Phase 5 and a
suggested Git commit message. Production release approval remains withheld;
changed dependency pins and real customer/operator validation are still open.

## Correctness, compatibility, and recovery

The full regression suite passes **516 tests** with no skips or expected failures.
This includes 41 new Phase 4 checks: 18 recovery/ownership cases, 15 compact-wire
cases, five benchmark arithmetic/gate cases, and three statistical-characterization
cases. Syntax, frozen-fixture integrity, and the actual saved-plan/project
reproduction runner also pass. The Phase 1 oracle and fixture bytes are unchanged.

The new cases cover interleaved source/plan changes, blocked/failed attempts,
coherent undo, stale-result attribution, fresh-process reopen with/without
history, explicit category decisions, changed types, renamed fields, duplicate
references, and corrupted or plausibly forged output evidence. Invalid restored
state fails before active session or prepared archive replacement.

Two actual traceability defects were found and corrected:

- One UUID could previously describe a failed attempt and a different committed
  success, or two different historical runs. Repeated identities now require
  identical typed semantic evidence throughout current/committed/history state.
- One plan ID/revision could previously describe two parameter sets. A revision
  now has one digest throughout the project. An identical earlier known revision
  can still be imported intentionally; changed parameters require a new revision.

Generic median/IQR replay was checked against the separately frozen arithmetic
examples: B refits to median 250 and IQR upper cap 70, not frozen-A values 25/7.
Generic frozen fitted policies remain unsupported and are not advertised as
verified reuse. Fulfillment plans continue to reject statistical operations.

## Workload results

The [frozen protocol](phase-4-protocol.md) uses 100,000 mixed-quality rows,
4,230,060 input bytes, the same six fields, one warm-up, and ten measured samples.
Each sample runs/exports in one fresh process and reopens the actual archive in
another. The reference profile includes both retained undo checkpoints, matching
the default application save option. All eleven samples were correct, retained
their intended history, and preserved saved run/source/plan identity on reopening.
Code and input fingerprints remained unchanged during the full run.

| Measurement | Observed | Phase 1 limit |
|---|---:|---:|
| Execution, nearest-rank p95 | 8.175 seconds | 30 seconds |
| Export/serialization/compression, p95 | 15.269 seconds | 30 seconds |
| Cold reopen/import/read/validation, p95 | 16.871 seconds | 30 seconds |
| OS whole-process peak RSS, including warm-up/imports | 751.23 MiB | 1,024 MiB |

The full history-inclusive F14 profile **passes** on the reference environment.
With ten measured samples the declared nearest-rank p95 is their maximum.
History-free export passed a separate one-sample pilot; it has no ten-sample
percentile claim. The measured workload is this exact 4.23 MB mixed file, not a
guarantee for every possible file below the admission limits or other hardware.

Every sample matched independent counts: 50,000 accepted, 50,000 quarantined,
70,000 reason assignments, 40,000 distinct accepted orders, and quantity 120,000.
Canonical values, exact-duplicate references, status subtotals, and source/run/plan
associations were checked as well as totals. Source-row rejection counts never
double-count a line with multiple reasons.

Reference host: Windows 11 build 26300, AMD64 Family 25 Model 97 Stepping 2,
12 logical CPUs, 16,282,554,368 bytes RAM, Python 3.12.14. These are backend
process measurements with explicit environment overrides and full provenance,
not browser latency or unspecified-concurrency guarantees.

## Application-scale workflow

A separate 100,000-row headless Streamlit AppTest completed actual source loading,
execution, project preparation, clearing, and reopening through the real pages,
StateManager, and archive codec. Only upload transport was mocked with actual
CSV/archive bytes. Source, plan, run identity, counts, 10,000 multi-reason rows,
and 10,000 exact duplicates survived reopening.

UI execution took 9.755 seconds, project preparation 13.242 seconds, and reopening
15.510 seconds. Whole-process peak was 732.83 MiB, including imports, prior stages,
and test/session overhead. Previews remained bounded to 100 source/accepted/
quarantined records. This is one headless application profile, not real-browser
usability evidence, a cold fresh-process UI restart, or a ten-sample UI p95 claim.

## Retaining a large previous month

An additional fresh-process diagnostic reopened the actual 100,000-row A project,
loaded a distinct 100,000-row October B using the unchanged plan, injected a final
reporting failure, and retried successfully. The failed B attempt preserved A's
run, source, report, and row identities and exposed no completed candidate output.
Retry produced the expected 50,000 accepted/50,000 quarantined rows and 70,000
reasons, with October dates and new source-scoped row identities.

Peak memory was **949.47 MiB**, below 1 GiB but higher than the first-month F14
reference profile. Setting B while retaining A took 9.210 seconds; the injected
failure took 4.694 seconds and retry 13.075 seconds. This is one supplemental
diagnostic, not an additional ten-sample percentile.

The existing undo-byte budget pruned **two checkpoints to zero** during that
large source replacement. Prior committed A remained available until B succeeded;
undo availability was not guaranteed beyond the retained budget. Preserve a
downloaded project before replacing large sources when an earlier checkpoint
must remain recoverable. No history limit was raised or pruning hidden.

## Changes supporting the proof

[Compact persistence](phase-4-persistence.md) stores original source bytes and
actual attempt evidence once per identity, retains explicit wire compatibility,
and restores independently owned public values. No archive/history/node/source
budgets were raised. Actual output evidence is preserved and checked; it is not
silently replaced by recomputation after reopening.

Per-validation source caching and typed semantic fingerprints reduce redundant
parsing/retained JSON strings while still checking every occurrence. Explicit
history-free export omits history before validation/copying. Checkpoint budgeting
avoids allocating snapshots immediately pruned by the byte cap. Deferred CSV
exports retain their owned committed run instead of duplicating all records on
each UI rerun. Recovery, ownership, corruption, and legacy compatibility tests
protect these changes.

The initial all-valid diagnostic approached the memory ceiling. It was a different
diagnostic workload, so its memory value is not an apples-to-apples performance
improvement claim. Current mixed-workload metrics are the acceptance evidence.

## Environment decision and release gates

The owner asked to preserve changed dependency pins unless application errors
were demonstrated. Those edits remain unchanged. The available package source
could not resolve `streamlit==1.66.1` during a dry-run; no packages were installed.
That is an installation-validation limitation, not a demonstrated application bug.

| Package | Declared pin | Installed and tested |
|---|---|---|
| Streamlit | 1.66.1 | 1.55.0 |
| pandas | 3.1.0 | 2.3.3 |
| NumPy | 2.5.4 | 2.4.2 |
| Plotly | 7.1.0 | 6.6.0 |
| scikit-learn | 1.9.1 | 1.8.0 |

The other seven declared pins match the recorded installed environment. Passing
tests/measurements qualify only the installed versions; they do not validate a
clean installation of the five changed pins. That environment gate remains open.

Real operator confirmation, two permissioned representative exports, customer-
held-out compatibility, dashboard import, observed preparation-time benefits,
Linux execution, licensing, and applicable deployment controls also remain open.
Synthetic correctness and faster execution do not establish product demand.

See [the evidence index](evidence/phase-4/README.md) for commands, raw outputs,
full measurements, source/fixture hashes, and retained local artifacts. The
[suggested commit message](phase-4-commit-message.txt) covers the engineering
changes; staging and commits are unchanged. Pause for approval before Phase 5.
