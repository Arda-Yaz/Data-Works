# Phase 4 evidence index

Client date: 2026-10-27, Europe/Istanbul. Command timestamps use UTC. Source began
at HEAD `7408223e8542f2bcc39086e9e94b38ee32fcc131` with the earlier phases' local
work and the owner's dependency-pin edits. The [starting inventory](starting-state.json)
records that exact state. HEAD alone does not identify the tested source.

## Final verification

| Check | Observed result | Evidence |
|---|---|---|
| Full regression/Streamlit suite | 516 tests, `OK`, no skips | [output](regression-suite.txt), [commands/source hashes](verification.json) |
| Syntax, actual A/B/C reproduction, frozen oracle audit | All exit 0 | [metadata](verification.json), [workflow output](synthetic-workflow.txt), [fixture audit](frozen-fixture-audit.txt) |
| Expanded recovery/state | 38 checks passed: 18 new recovery and 20 retained state checks | [output](recovery-tests.txt), [metadata](recovery-tests.json) |
| Compact-wire/codec/version checks | 38 focused checks passed, including 15 new wire checks | [metadata](wire-focused-checks.json) |
| Full F14 reference profile | Passed: 100,000 rows, one warm-up, ten measured samples, history included | [full results](benchmark-full-with-history/results.json), [invocation](benchmark-full-with-history-invocation.json), [raw output](benchmark-full-with-history-stdout.txt) |
| 100,000-row headless application workflow | Passed source load/run/save/clear/reopen and exact evidence checks | [results](ui-workload-100000/results.json), [command](ui-workload-100000/command.json) |
| Declared dependency installation | Open: resolution failed; owner requested preserving pins | [dry-run output](declared-dependency-resolution.txt), [metadata](declared-dependency-resolution.json) |
| Installed package compatibility | No broken requirements, exit 0; distinct from declared-pin installation | [output](installed-dependency-check.txt), [metadata](installed-dependency-check.json) |
| Source/fixture/requirements/index integrity | See final source and artifact inventory | [final-integrity.json](final-integrity.json) |

The full F14 reference run took 451.618 seconds for all worker pairs. Nearest-rank
p95 values are execution **8.174651s**, export **15.269182s**, and cold reopen
**16.870697s**. Whole-process maximum, including warm-up/imports/oracle checks,
was **787,718,144 bytes (751.23 MiB)**. All eleven samples matched expected
values/counts and preserved run/source/plan identity; code/input fingerprints
were stable. Both retained undo snapshots were exported and reopened in each.

The source CSV contains 100,000 records and 4,230,060 bytes, SHA-256
`0044ea22bebda2be5c3242b249c3f32a184be3049fcfa80964b47991ee15f86e`.
Independent totals are 50,000 accepted, 50,000 quarantined, 70,000 reasons,
40,000 distinct accepted orders, and quantity 120,000. The generator/oracle and
threshold math have five separate passing tests. [The protocol](../../phase-4-protocol.md)
fixes the reference host, archive profile, workload, and threshold interpretation.

## Supplemental evidence

- [History-free pilot](benchmark-pilot-100k-nohistory-v1/results.json): one sample,
  no warm-up; execution 8.469s, export 12.778s, reopen 14.980s, peak 737 MiB.
- [History-inclusive pilot](benchmark-pilot-100k-withhistory-v1/results.json): one
  sample, no warm-up; execution 10.352s, export 15.365s, reopen 17.205s, peak 752 MiB.
- [Initial diagnostic](initial-pilot.json) and [its script](initial-pilot.py): an
  earlier all-valid workload exposed duplicated retention near the memory ceiling.
  This is a different diagnostic, not a before/after improvement comparison.
- [Application profile](ui-workload-100000/results.json): one real-page AppTest
  run with only upload transport mocked. Execution 9.755s, preparation 13.242s,
  reopening 15.510s, and process peak 732.83 MiB. Identity, counts, multi-reason
  quarantine, and duplicate references survive reopening. It is not real-browser
  usability evidence or a ten-sample UI percentile.
- [Large retained-month recovery](large-month-reuse.json), with its
  [reproduction script](large-month-reuse.py): reopened a full prior A, loaded a
  distinct B with the unchanged plan, injected failure, preserved A, and retried
  B successfully. Peak was **949.47 MiB**, higher than the reference profile but
  below 1 GiB. Undo checkpoints were explicitly pruned from two to zero under
  the existing byte cap. This is one separate diagnostic, not a percentile claim.

Pilots and the application check are explicitly supplemental; they do not replace
the full one-warm-up/ten-sample reference protocol. Timings include the boundaries
declared by each script, rather than conflating a backend call, headless page
interaction, and human preparation time.

## Reproduce

Use the recorded Python 3.12 environment and new output directories:

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py
.\.venv\Scripts\python.exe scripts/verify_fulfillment.py --output-dir .test-tmp/fulfillment-proof
.\.venv\Scripts\python.exe scripts/benchmark_fulfillment.py --include-history --output-dir .test-tmp/fulfillment-benchmark
.\.venv\Scripts\python.exe scripts/verify_fulfillment_workload_ui.py --output-dir .test-tmp/fulfillment-ui
.\.venv\Scripts\python.exe docs/acceptance/evidence/phase-4/large-month-reuse.py --benchmark-dir .test-tmp/fulfillment-benchmark --output .test-tmp/large-month-reuse.json
```

Inspect each utility's `--help` for supplemental profiles. The full benchmark
uses sequential worker pairs and fails partial/incorrect/oversized/slow samples;
smaller row/run counts cannot become an F14 pass. Avoid other DataWorks workloads
while measuring the single-user reference profile.

Generated CSVs and `.dataworks` files remain locally available but are ignored
by this evidence directory's `.gitignore`. Commit-ready evidence consists of
the deterministic generators, frozen oracle, hashes, JSON measurements, command
logs, and source inventory. Regenerate binary artifacts using the exact command;
their presence alone is not a release claim.

## Environment and release meaning

The full report records installed versions, declared pins, and their mismatch.
Five owner-edited pins remain unchanged; testing used Streamlit 1.55.0, pandas
2.3.3, NumPy 2.4.2, Plotly 6.6.0, and scikit-learn 1.8.0. The other seven installed
packages match their pins. No packages were changed by this phase. The available
source failed to resolve Streamlit 1.66.1; that leaves clean-install validation
open rather than proving a runtime application defect.

The full benchmark passes on the recorded Windows host/environment and exact
synthetic workload. Changed pins, customer exports/dashboard import, operator
validation/time savings, Linux, licensing, and target hosting remain separate
release gates. The [Phase 4 report](../../phase-4-results.md) and
[persistence notes](../../phase-4-persistence.md) distinguish these limits.
Staging/commits are unchanged; the [commit message](../../phase-4-commit-message.txt)
is a suggestion for review. Pause before Phase 5.
