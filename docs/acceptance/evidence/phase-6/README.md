# Phase 6 evidence index

Date: 2026-10-30 (Europe/Istanbul). Local engineering verification only. The final
phase result record and integrity audit state the observed results and remaining
gates; production approval is withheld. No Git staging, commit, public upload,
or deployment is performed.

## Local artifact

`dataworks-verification-final.zip` contains 229 source files plus `BUNDLE-MANIFEST.json`.
It is 591,890 bytes, SHA256
`2093427202d02ca4b2b561481052a7aae3df11ee8e3b0274e793d3cc720c1774`.
`dataworks-verification-final.zip.sha256` provides the same checksum. The builder's
`bundle-final-build.json` proves byte-identical repeated construction and records the
canonical source snapshot fingerprint. `final-bundle-review.json` records
a separate agent's exact inventory/content review. This is not human reproduction
or a signed authenticity guarantee.

The ZIP and `extracted-final/` tree stay local and are ignored by this folder's
`.gitignore`; code, original fixtures, manifests, and textual evidence remain
reviewable. The artifact excludes the legacy root CSV, secrets configuration,
Git metadata, Python environments, generated runs, and this phase's result
artifacts to avoid self-referential fingerprints.

## Environment and compatibility

- `declared-dependency-resolution.{json,txt}`: failed sandboxed dry-run; it cannot
  establish that a pin is unavailable from the package source.
- `declared-dependency-resolution-network.{json,txt}`: successful resolution
  outside the network sandbox, without installing anything.
- `declared-clean-install.{json,txt}`: actual installation into a new isolated
  `.test-tmp/phase6-declared-env`, preserving the existing `.venv` and manifest.
- `declared-dependency-check.{json,txt}`: actual compatibility check of that new
  environment. Complete distributions are recorded in
  `environment/declared-clean-windows-py312.json` in the repository/bundle.
- `declared-env-diagnostic.txt`: the original socket-sandbox stall, including
  the stack in asyncio's local socket-pair creation. It was interrupted; it is
  not passing regression evidence.
- `declared-env-workspace-network/`: initial actual declared-version failures,
  17 failures/three errors in 551 tests, before compatibility fixes. The failed
  transcript is retained rather than replaced.
- `workspace-declared-final/` and `workspace-original-final/`: complete 559-test
  success and actual saved-plan/project fresh-process proof after fixes. Default
  strict environment checking was used for the declared environment; the earlier
  environment explicitly used `--allow-environment-mismatch`.
- `page-text-{original,declared}-check.*` and `ingestion-fixture-check.*`: focused
  compatibility/provenance checks. Full final checks include later assertions.

## Separate-copy reproduction and workload

- `package-declared-final/`: strict declared-environment verification from the extracted
  package; all 559 tests, actual fresh-process proof, and manifest/source/fixture
  stability checks passed.
- `package-original-final/`: the same extracted package passed all 559 tests and the
  fresh-process proof with an explicit historical-environment override.
- `package-old-default-final/` and its invocation transcript: default mode
  correctly returned exit 2 for the historical dependency mismatch after passing
  package integrity. A blocked gate is not counted as a passing regression run.
- `bundle-final-copy-rebuild.json`: building from the extracted copy with the declared
  interpreter reproduced the exact original ZIP bytes/checksum.
- `benchmark-declared/results.json` and sibling invocation/stdout/stderr files:
  the complete history-inclusive F14 protocol passed, with execution/export/cold
  reopen p95 of 7.353/12.838/17.211 seconds and 754.45 MiB process peak.
- `ui-workload-declared/results.json` and sibling transcripts: the single
  100,000-row actual-page profile passed with 737.74 MiB peak.
- `large-month-declared.json` and sibling transcripts: prior large-month results
  survived injected failure, retry succeeded, undo was pruned from two to zero,
  and the highest observed phase peak was 952.93 MiB. The reused script's fixed
  historical `client_date` label is distinguished from the new invocation's actual
  2026-10-30 date and UTC execution times.
- `final-integrity.json`: final package/source/environment/fixture/Git/evidence
  audit, including checksum and retained external release gates.

The checked extracted source tree has no Git metadata, application environment,
legacy root CSV, or generated projects. The final package checks use that tree's
own verifier and scripts, with outputs outside the extracted tree. The selected
interpreter's site-packages are recorded and child `PYTHONPATH` is cleared. This
is a separate source copy/process on the same host, not an independent person.

The declared-environment workload runner uses the original F14 mixed input and
independent oracle, one warm-up and ten measured samples, and history-inclusive
actual archive reopening. A separate 100,000-row AppTest profile exercises the
pages and history-free archive. Large prior-month reuse/failure/retry is a
separate diagnostic, not another ten-sample percentile. Refer to their actual
results and command transcripts for statuses and measurements.

Reproduce from the source package using a new output directory each time:

```powershell
python scripts/verify_release.py --output-dir ..\verification-results
python scripts/benchmark_fulfillment.py --include-history --output-dir ..\benchmark-results
python scripts/verify_fulfillment_workload_ui.py --rows 100000 --output-dir ..\ui-results
```

Historical profiles require the explicit environment-mismatch flag. A clean
installation is established by installation evidence, not inferred from matching
versions or an override. The original fixture manifest and owner-edited
`requirements.txt` remain unchanged. Actual operator/customer/dashboard evidence,
independent human reproduction, Linux, licensing, and applicable shared-host
controls remain separate release gates.

Earlier candidate artifacts and checks without the final suffix remain as
history. A final documentation audit replaced three historical archive links
with explicit local-file references because generated projects are omitted from
the ZIP. The final 155 packaged local Markdown links all resolve; executable
sources and measured workload code were unchanged by that documentation fix.
