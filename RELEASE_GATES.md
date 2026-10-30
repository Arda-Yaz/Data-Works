# Engineering release decision

**Production release approval is withheld.** This package is for local,
single-user synthetic engineering verification. Passing its verifier is not
permission to publish, redistribute, deploy publicly, or claim customer benefit.
The Phase 6 result record supplies the actual package/checksum and observed checks.

| Gate | Required evidence / current boundary |
|---|---|
| Execution and data integrity | Independent synthetic expected outputs, accepted/rejected partition, attributed reasons, explicit conflicts, exact identifiers/dates/quantities, and failure-preserved prior result. Re-run the included verifier for this artifact. |
| Persistence and compatibility | Actual saved plan/project, new-process unchanged-plan reuse, strict unsupported/corrupt evidence handling, coherent undo, and reopened source/plan/run identity. Covered by synthetic checks; actual customer compatibility remains separate. |
| Dependency installation | The preserved pins installed into a fresh Windows/Python 3.12 environment on 2026-10-30. All 559 tests, fresh-process proof, and dependency compatibility check passed. Exact distributions and actual installation evidence are recorded separately. Historical and declared environments remain distinguished. |
| Workload | Record F14's one warm-up plus ten measured samples under each claimed environment. Limits: execution/export/reopen p95 at most 30 seconds each and peak RSS at most 1 GiB for the exact declared workload. Prior-environment measurements do not validate new pins. |
| Operator and downstream consumer | One real operations analyst confirms the recurring job; obtain two permissioned consecutive exports and the dashboard's required format/meaning. Not yet obtained. |
| Unaided completion and value | Observe task/recovery without directing button choices; compare dashboard totals and active preparation time against the existing process. Proposed gate: at least 50% lower active time on each export and no unexplained count differences. Not observed. |
| Independent reproduction | A different person reproduces the package and understands failure behavior/limits. Another process/agent on the same host is useful engineering evidence, not this human gate. Pending. |
| Platform | Windows/Python 3.12 evidence is attributed in phase results. Windows/Linux CI is configured; an observed successful Linux run is still required before a Linux claim. |
| Licensing and redistribution | Owner chooses the project's license and confirms distribution rights/dependency obligations. No project license is currently established. Original synthetic provenance is documented; the legacy unattributed CSV is excluded from the ZIP. |
| Shared deployment, if selected | Real OIDC provider callback, verified allowlist claims, TLS/WebSocket proxy, ingress/resource/session controls, and target-host operation. Current mocks and loopback configuration do not establish shared-host readiness. |

The first scoped workflow is local use with retained `.dataworks` downloads.
Sessions are not autosaved or durable across service restart. Undo is bounded;
its limit is not a process memory guarantee. Native table exports are previews.
Accepted CSV must still be checked against the real dashboard contract, including
spreadsheet-safe versus raw text behavior.

See [REPRODUCING.md](REPRODUCING.md), the
[acceptance specification](docs/acceptance/phase-1.md),
[operator observation sheet](docs/acceptance/phase-5-observation.md), and
[deployment controls](DEPLOYMENT.md). No synthetic metric establishes demand,
product-market fit, or a funding valuation.
