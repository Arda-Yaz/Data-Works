# Phase 3 evidence index

Date: 2026-10-26, Europe/Istanbul. Recorded command timestamps use UTC.
Current source starts from HEAD `7408223e8542f2bcc39086e9e94b38ee32fcc131`
with the working-tree changes inventoried in [final-integrity.json](final-integrity.json).
No commit, push, dependency installation, deployment, or release approval occurred.

## Current verification

| Check | Observed result | Evidence |
|---|---|---|
| Full current regression/Streamlit suite | 475 tests, `OK`, exit 0; no skips | [full output](regression-suite-current.txt), [command and source hashes](verification-current.json) |
| Current syntax | Exit 0 | [output](syntax-check-current.txt), [metadata](verification-current.json) |
| Dependency compatibility | No broken requirements, exit 0 | [output](dependency-check.txt), [metadata](verification-results.json) |
| Frozen fixture audit | Passed; Phase 1 inputs/oracle unchanged | [output](frozen-fixture-audit.txt) |
| Actual workflow and fresh-process A/B reopen/reuse | Passed; saved plan digest unchanged; no Streamlit imports in backend children | [results](workflow-current/results.json), [command](workflow-current-command.json), [full output](workflow-current.txt) |
| Multi-reason C ledger with undo history | Actual archive reopened and validated in a separate process | [result](multireason-reopen.json) |
| Source/fixture/index integrity | See final inventory | [final-integrity.json](final-integrity.json) |

The final test runner reported 9.666 seconds, with 13.341 seconds including
startup/import work. These are test timings, not F14 workload measurements.
The suite contains the previous 395 tests plus 80 new checks: 17 fulfillment
acceptance, 21 source/plan contract, 20 state/session, 10 page/export, 4 independent
preparation, and 8 archive-version checks. It includes preservation under injected
late failure, corrupt project/association/type rejection, explicit decision
outcomes, bounded undo, old-project import, and unchanged-plan fresh-process reuse.

The virtual environment still lacks its own pip module. The recorded system
pip command explicitly targets that Python 3.12 interpreter; it does not check a
different application's packages. No dependencies or requirement pins changed.
The supported runner supplies disposable repository-local scratch space for
Streamlit/fresh-process tests; direct unittest invocation under this sandbox
encountered an unwritable system temporary directory during development.

## Retained current proof artifacts

- [Saved plan](workflow-current/saved-plan.json).
- [Month A project](workflow-current/month-a.dataworks), envelope version 2.
- [Month B project](workflow-current/month-b.dataworks), envelope version 2.
- [Month C with undo history](workflow-current/month-c-with-history.dataworks),
  envelope version 2, including a row with three reasons counted once.

The original expected outputs were not regenerated from these archives. The
implementation, tests, and reproduction script compare against the independently
authored Phase 1 oracle. Current A/B reuse retained digest
`bde9417927d56f184c09b71d43a2b2a51313f32bacb18c4eb878c066e77f79fc`.
Source identities and run IDs are included in the result files; reopening retains
the saved run identity rather than replacing it with validation replay identity.

## Reproduce

From the repository root, use Python 3.12 and a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py
.\.venv\Scripts\python.exe scripts/verify_fulfillment.py --output-dir .test-tmp/fulfillment-proof
```

The first command runs the existing/new tests together. The second writes actual
saved plans/archives and launches independent child processes. It fails on output,
disposition, identity, plan-digest, fixture-integrity, or reopening discrepancies.
The parent directory may exist; the specified output directory must be new.

To verify retained C evidence directly:

```powershell
.\.venv\Scripts\python.exe scripts/verify_fulfillment.py --project docs/acceptance/evidence/phase-3/workflow-current/month-c-with-history.dataworks --expected-input month_c_invalid_rows.csv
```

Open the app with `python -m streamlit run app.py`, select **Order Fulfillment**,
and use the same synthetic CSVs to exercise the UI. **Project & Exports** opens
the retained `.dataworks` files. The Streamlit checks automate those interactions;
no real-operator observation or actual browser usability study is claimed.

## Development history and limits

`verification-results.json` records the initial 466-test run. `verification-final.json`
records the subsequent 467-test run after dataset-clear isolation. Both were
superseded by **verification-current.json** after the archive-version guard.
The `workflow/` directory and earlier artifact-revalidation record contain
pre-release version 1 fulfillment archives. They are historical diagnostics,
not current release artifacts; the new reader deliberately rejects that unsafe
version/extension combination. Current proof files are under **workflow-current/**.

The new archive guard keeps legacy version 1 import while requiring version 2
for fulfillment evidence. Older readers reject version 2 before they can ignore
and discard the added ledger in a mixed project. Unknown versions, downgraded
fulfillment payloads, unsupported extension state, and malformed associations fail
before active state replacement.

The workflow is verified against synthetic inputs on this Windows environment.
Real customer-held-out files, dashboard imports, operator demand/preparation-time
benefit, Linux CI, target workload p95/peak RSS, and shared-host authentication/
proxy behavior remain unverified. No statistical operations enter this plan.
Generic statistical reuse still requires its own explicit fitting-policy work.
Phase 4 requires the owner's approval; release sign-off remains withheld.
