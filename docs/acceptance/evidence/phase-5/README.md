# Phase 5 evidence

Date: 2026-10-29. Synthetic engineering verification only; real-operator
observation and production release approval remain pending. Start from the
[protocol](../../phase-5-protocol.md) and [results](../../phase-5-results.md).

Final evidence:

- `verification-final/results.json`: complete source hashes, twelve actual
  installed/declared packages, unchanged Git HEAD/index, syntax, frozen fixture
  hashes, command statuses, and source stability during checks.
- `verification-final/regression.txt`: **531 tests, OK**, no skips.
- `verification-final/fresh-process-workflow.txt` and `verification-final/workflow/results.json`:
  actual saved-plan/project A→B reuse and reopening, deliberate incompatibility,
  decisions, and rejection reconciliation against the independent oracle.
- `ui-workload-100000-final/results.json`: actual 100,000-row UI execution,
  history-free archive preparation, clearing, reopening, identity/correctness
  checks, timings, OS peak RSS, host, packages, and code/input hashes.
- `ui-workload-final-invocation.json` and stdout/stderr transcripts: final UI
  workload command, process return code, and captured output.
- `workspace-tests.{txt,json}`: focused 11-journey run and its source hashes before
  the last browser-driven caption adjustment. The final complete regression
  above includes those same journeys against the final code.
- [Browser smoke record](browser-smoke.md), its actual final accessibility state,
  and the two PNG captures: narrow rendered browser and keyboard check using the
  labeled [synthetic fixture harness](browser_app.py).
- `final-integrity.json`: final source/environment/fixture/evidence audit and
  comparison with the earlier measured backend implementation.

`verification/` and `ui-workload-100000/` retain the earlier passing run before
the preview-toolbar wording correction. Their hashes distinguish that state;
they do not replace final-code verification. Large generated CSV and `.dataworks`
files remain local and are ignored by this folder's `.gitignore`; the generators,
oracle, input hashes, and textual evidence are retained.

Reproduce engineering checks from the repository with its recorded environment.
Choose a new output directory for every invocation:

```powershell
python docs/acceptance/evidence/phase-5/verify.py --output-dir .test-tmp/phase5-verification
python scripts/verify_fulfillment_workload_ui.py --rows 100000 --output-dir .test-tmp/phase5-ui
```

For the optional rendered smoke, the local harness uses loopback only:

```powershell
python -m streamlit run docs/acceptance/evidence/phase-5/browser_app.py --server.address=127.0.0.1 --server.port=8515 --server.headless=true --browser.gatherUsageStats=false
```

Run the documented synthetic sequence, then stop the temporary server. This is a
test harness, not the production entry point or deployment guidance. The same
real pages are routed by `app.py`; its mixed dataset/fulfillment navigation and
separate undo controls are exercised in the regression suite.

The owner asked to preserve changed pins. These commands do not prove a clean
installation of that manifest. No Stage6 publication, Git staging/commit, or
release approval is performed by these checks.
