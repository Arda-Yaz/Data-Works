# Phase 4: frozen reuse, recovery, and workload protocol

Date: 2026-10-27. The owner authorized Phase 4 and requested a suggested Git
commit message. Pause before Phase 5; no staging, commit, push, or release is
authorized by this message request.

## Correctness and recovery

Retain the frozen Phase 1 fixtures and independent expectations. Expand tests
across source changes, plan revisions, blocked/failed runs, coherent undo, and
fresh-process save/reopen with and without history. Reject source/plan/report/
duplicate/type/order tampering before active session replacement. A run UUID
must identify exactly one attempt meaning across current and historical state.
One `(plan_id, revision)` must identify exactly one parameter set; importing a
known earlier revision with identical meaning remains valid.

Fulfillment plans continue to forbid imputation/outlier steps. Separate frozen
arithmetic examples characterize existing generic recipe replay as refitting
median/IQR statistics per input. Generic recipes still do not persist a frozen/
refit policy; those characterization checks do not approve generic cross-file
reuse or add statistical transformations to fulfillment reporting.

## F14 reference workload

Use the deterministic mixed-quality generator in
[`benchmark_fulfillment.py`](../../scripts/benchmark_fulfillment.py). Freeze:

- 100,000 parsed rows, the declared six columns, UTF-8 CSV, 4,230,060 input bytes.
- Raw input SHA-256:
  `0044ea22bebda2be5c3242b249c3f32a184be3049fcfa80964b47991ee15f86e`.
- Every ten rows contain five accepted lines and five rejected lines, including
  an exact normalized duplicate and a row with three independent reasons.
- Expected totals: 50,000 accepted, 50,000 rejected, 70,000 reason assignments,
  40,000 distinct accepted orders, and quantity 120,000. Status line totals are
  20,000 fulfilled, 20,000 pending, 10,000 cancelled; quantities are 60,000,
  30,000, and 30,000 respectively. Check canonical values and duplicate references
  as well as aggregate counts.
- Reference archive profile **includes undo history**, matching the default
  application save option. An explicit history-free pilot is supplemental;
  it does not establish a ten-sample history-free percentile.

Use one warm-up and ten measured samples. Each sample executes/exports in a
fresh process, exits, then reopens its actual archive in another fresh process.
Verify run/source/plan identity survives reopening and every saved result matches
the independent workload oracle. New default plan UUIDs are allowed per sample;
supported default parameters, input bytes, and measured implementation are fixed.
Saved-plan reuse across different files is proved separately by F02/recovery tests.

Execution timing covers `run_project` candidate computation/commit transition;
ingestion/setup are separately recorded. Export includes selected-state validation,
typed serialization, and compression; writing bytes is separately recorded.
Cold reopen includes backend imports, file reading, decoding, and full semantic
validation in the fresh process. These are backend timings, not browser latency.

Apply the Phase 1 thresholds without weakening them: nearest-rank p95 execution,
export, and reopen each at most 30 seconds; input at most 20 MiB; OS-reported
whole-process peak RSS at most 1 GiB, including imports and oracle checks. With ten
samples, nearest-rank p95 is the maximum measured value. Exclude warm-up timings
from p95 but include warm-up RSS in the memory gate. Failed/partial/pilot samples
cannot count as an F14 pass. Record source hashes and fail if code/input changes
during measurement. Do not run other DataWorks workloads concurrently.

Reference host: Windows 11 build 26300, AMD64 Family 25 Model 97 Stepping 2,
12 logical CPUs, 16,282,554,368 bytes physical RAM, Python 3.12.14. Record exact
installed package versions. This host is the acceptance reference, not a promise
for unspecified hardware, concurrency, inputs, or browser sessions.

## Application-scale check

Run a separate single-sample, headless Streamlit AppTest on the same 100,000-row
input: real source-load/run pages, project preparation, clearing, and reopening
actual archive bytes. Mock only the upload transport. Verify the same source,
plan, run ID, counts, multi-reason records, duplicate references, and bounded
previews. Record phase durations and whole-process peak RSS. It complements the
backend protocol and does not claim real-browser interaction or UI p95 evidence.

## Environment and release boundary

The owner changed five dependency pins after Phase 3 and requested that they be
preserved unless demonstrated to cause application errors. The installed packages
remain the earlier tested versions. Available-source resolution could not find
`streamlit==1.66.1`; that records an installation-validation limitation, not a
demonstrated runtime defect. Preserve the requirements file and report both
declared and installed versions. Performance/correctness results qualify only
the recorded installed environment; a clean install of changed pins remains open.

No archive, source, or history budgets may be globally raised to make the
benchmark pass. Compact persistence may deduplicate original source bytes and
actual attempts, preserving public ownership, typed evidence, legacy reads,
explicit wire versions, and validation before session replacement.

Real operator confirmation, permissioned consecutive exports, dashboard contract,
business-time baseline, customer-held-out files, Linux checks, and applicable
deployment controls remain separate release gates. Phase 4 cannot establish those
from synthetic files or from faster benchmark results.
