# Reproduce the fulfillment verification

This is a local verification candidate for one monthly order-line workflow.
Production approval, operator acceptance, and redistribution rights are separate
gates. See [RELEASE_GATES.md](RELEASE_GATES.md). The sources, original synthetic
fixtures, independent expected outputs, and measured limits are included; no
customer demand or measured preparation-time benefit is claimed.

## Setup on Windows

Use Python 3.12, create a new environment, and install the unchanged declared pins:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/verify_release.py --output-dir ..\dataworks-verification
```

Choose a new output directory each time. Exit 0 means the synthetic engineering
checks passed in the recorded environment; the JSON still says `release_ready:
false`. Exit 2 means the Python/dependency gate blocked execution. Exit 1 means an
integrity, syntax, or execution check failed. Read `results.json` and actual logs
in the output directory. Nothing is installed or published by the verifier.

The verifier checks a bundle manifest when present, frozen fixture hashes,
executable source stability, all regression tests with no skips, and actual
fresh-process saved-plan reuse/project reopening. It clears child `PYTHONPATH`,
disables user site packages, and uses the selected source tree as its working
directory. It needs no Git metadata in an extracted bundle. Its hashes establish
content integrity, not a signature or trusted publisher identity.

The preserved declared pins installed into a fresh isolated Windows/Python 3.12
environment on 2026-10-30, and all 559 tests plus the actual fresh-process
fulfillment proof passed. Its complete distribution snapshot is
`environment/declared-clean-windows-py312.json`. The original environment also
passed the full suite. Installation transcripts and final package measurements
are in the separate Phase 6 handoff.

The earlier verified environment is recorded in
`requirements-verified-windows-py312.txt` and
`environment/verified-windows-py312.json`. Those are an observed environment
profile, not a wheel/hash lock or proof of a clean installation. To examine that
historical environment explicitly, install its profile into another environment
and pass `--allow-environment-mismatch`. That flag records the mismatch and never
turns it into a declared-pin pass. The Phase 6 handoff separately records the fresh
declared-pin installation and its actual checks.

Some Windows execution sandboxes prohibit local sockets. Streamlit 1.66.1's AppTest
creates an asyncio event loop that requires a local socket pair. Run the tests in
an authorized environment permitting local IPC; do not disable application CORS,
XSRF, or authentication to work around a test-host constraint.

## Short walkthrough

Start `python -m streamlit run app.py`, then open **Order Fulfillment**.

1. Load `tests/fixtures/fulfillment_v1/month_a.csv`, review the plan, and run it.
   Expect 12 source lines, four accepted, eight quarantined, three accepted orders,
   and quantity 5. Inspect a reason: that filter affects only its preview. Native
   table-toolbar exports contain that preview; use the named fulfillment download
   buttons for the full accepted/quarantine outputs and report.
2. Download the plan. In **Project & Exports**, prepare and retain the actual
   `.dataworks` download. Start a fresh session and reopen it. Preparation alone
   is not saving; the app cannot confirm that the browser retained a download.
3. Load Month B and run the unchanged plan. Expect eight source lines, five
   accepted, three quarantined, four accepted orders, and quantity 11. Source/run
   identity changes while plan identity/digest stay unchanged.
4. Load the missing-column or extra-column C fixture. Execution blocks and the
   earlier B report remains attributed to B. Conflicting duplicates require
   corrected source records; the interface never selects a winner. Unknown
   statuses require an explicit reviewed mapping, a plan revision, and a new run.
5. C invalid rows completes with two accepted and seven quarantined rows. Its
   nine reasons do not count as nine rejected records. Undo restores coherent
   source/plan/attempt/result state while a checkpoint is retained. Large changes
   can release all undo checkpoints; retain projects for longer recovery.

The [frozen fixture README](tests/fixtures/fulfillment_v1/README.md) identifies
original synthetic provenance and the manually specified oracle. The generic
ingestion regression also uses an [original fixture](tests/fixtures/ingestion_v1/README.md).
The unknown-provenance `surat_uncleaned.csv` is excluded from verification bundles
and no longer required by tests; it is left untouched in the original repository.

Project serialization preserves both pandas nullable-string (`pd.NA`) and
NaN-backed string semantics, including index/column storage. Older two-field
string descriptors remain readable. NaN-backed strings use an additional explicit
descriptor field; older DataWorks builds reject that unsupported descriptor
instead of silently changing its null semantics. Use this build to reopen it.

Fulfillment contains no learned imputation/outlier policy. Generic statistical
recipe replay refits per input; frozen fitted policies remain unsupported. See
the [contracts](docs/acceptance/phase-2-contracts.md) and
[persistence decisions](docs/acceptance/phase-4-persistence.md).

## Workload and local package

The declared F14 protocol uses 100,000 mixed rows, one warm-up and ten measured
samples, whole-process peak RSS, and actual cold-process archive reopening:

```powershell
python scripts/benchmark_fulfillment.py --include-history --output-dir ..\dataworks-benchmark
python scripts/verify_fulfillment_workload_ui.py --rows 100000 --output-dir ..\dataworks-ui-profile
```

Smaller/fewer-sample pilots are not a full performance pass. Measurements qualify
their recorded hardware/packages/input/code. The UI profile is one headless
application interaction, not a ten-sample UI percentile or observed user benefit.

Build the local ZIP from the repository root, then use a new checked extraction:

```powershell
python scripts/build_verification_bundle.py --output ..\dataworks-verification.zip --extract-to ..\dataworks-source-copy
```

The builder selects tracked files plus explicitly named new verification files,
then applies path/type limits. It excludes secrets configuration, Git metadata,
environments, generated runs, and the unattributed CSV. Review `BUNDLE-MANIFEST.json`
for the exact inventory. The accompanying SHA256 identifies the ZIP; it is not a
publisher signature. Repeat builds from identical file bytes and origin metadata
are deterministic on the recorded Python/zlib runtime. Fixed ZIP timestamps do
not imply equivalence across every compression runtime.

For a separate-copy check, select an interpreter from a known environment, change
to the extracted source directory, and run its own `scripts/verify_release.py`.
Use an output directory outside that source copy. This proves package source
completeness on that host; independent human and second-host reproduction require
their own attributed evidence. Final Phase 6 results and ZIP checksums are handed
off separately rather than embedded recursively in the artifact being verified.
