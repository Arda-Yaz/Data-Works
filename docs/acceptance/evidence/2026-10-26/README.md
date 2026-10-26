# Phase 1 baseline evidence

Client date: 2026-10-26, Europe/Istanbul. Command timestamps are recorded in UTC.
Starting HEAD: `4b8ab1c04b355c331d850fcf6ed627ae767ba3ba` on `main`.
The tested source includes pre-existing staged, unstaged, and untracked work;
HEAD alone does not identify that source. See the complete starting inventory.

## Observed results

| Check | Result | Evidence |
|---|---|---|
| Interpreter | Python 3.12.14, Windows, project `.venv` | [environment.json](environment.json) |
| Direct dependency pins | All 12 installed versions match `requirements.txt` | [environment.json](environment.json) |
| Project-local `python -m pip check` | Could not run: the `.venv` has no pip module | [dependency-check.txt](dependency-check.txt) |
| System pip targeting project interpreter | Exit 0, no broken requirements | [dependency-check-alternate.txt](dependency-check-alternate.txt) |
| CI syntax command | Exit 0 | [syntax-check.txt](syntax-check.txt), [command metadata](baseline-results.json) |
| Existing regression suite | Exit 0; 395 tests; `OK` | [full output](regression-suite.txt), [command metadata](baseline-results.json) |

The test runner reported 4.980 seconds for 395 tests; the enclosing subprocess
took 8.245 seconds including startup/import work. These are baseline test timings,
not workload performance measurements. Streamlit bare-mode context warnings and
deliberately exercised diagnostic error events appear in the unabridged log;
the suite reported no failures or skips. No dependencies were installed or changed.

This verifies the existing suite on this Windows environment. It does not verify
Linux execution, a clean dependency installation, a real OIDC callback, customer
dashboard imports, the new fulfillment workflow, fresh-process acceptance reuse,
performance budgets, or real-user value. The CI configuration describes intended
Windows/Linux checks; no remote CI run is claimed here.

## Reproduction

From the repository root, use its Python 3.12 environment:

```powershell
$env:DATAWORKS_REQUIRE_AUTH = 'false'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m compileall -q app.py backend frontend scripts tests
.\.venv\Scripts\python.exe scripts/run_tests.py
```

For dependency checks, use `.\.venv\Scripts\python.exe -m pip check` when pip
is installed in the environment. This environment instead used the available
system pip with `pip --python .\.venv\Scripts\python.exe check`. Inspect the
target interpreter and installed versions; do not substitute another interpreter
and claim the same environment was checked. Standard dependency installation
instructions remain in the repository README.

The `checks` records preserve the original commands, durations, and exit codes.
The initial unavailable pip check is retained alongside its successful alternate;
it has not been rewritten as a successful original invocation.

## Starting state and artifact integrity

- [starting-state.json](starting-state.json): source-file SHA-256/size inventory,
  index blob identities, raw porcelain status, platform, and capture time before
  Phase 1 authoring. It excludes ignored environments and Git internals; it is
  a hash inventory, not a content backup or a guarantee of a clean checkout.
- [git-status-before.txt](git-status-before.txt) and
  [diff-stat-before.txt](diff-stat-before.txt): recorded pre-existing changes.
- [fixture-audit.py](fixture-audit.py) and [fixture-audit.json](fixture-audit.json):
  independent standard-library checks of fixture structure, preserved tokens,
  date/status normalization, partitions, duplicate references, report arithmetic,
  statistical examples, and declared blocker payloads. They import no production
  code. Run `.\.venv\Scripts\python.exe docs/acceptance/evidence/2026-10-26/fixture-audit.py`.
  These checks audit the oracle, not application behavior.
- [final-integrity.json](final-integrity.json): final comparison of the starting
  source/index inventory and hash list of Phase 1 artifacts.

The separate [fixture manifest](../../../../tests/fixtures/fulfillment_v1/manifest.json)
freezes synthetic inputs and expectations for later acceptance implementation.
Phase 2 must not derive replacement expectations from application output to make
an implementation pass. Changed business requirements need a reviewed version.

An independent review also checked the raw fixtures and expected arithmetic
without production imports. The fixtures do not cover every declared edge rule:
signed-64 boundaries, whitespace-bearing IDs, all missing-field combinations,
header permutations/duplicates, mixed unknown-status errors, repeated pandas
indices, injected failures, undo, and fresh-process application execution still
need executable cases in the next phases. The specification declares the
unknown-status precedence; the supplied status fixture isolates that decision.
