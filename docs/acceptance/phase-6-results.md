# Phase 6: reproducible package and engineering release decision

Date: 2026-10-30 (Europe/Istanbul). Phase 6 engineering verification is complete;
production release approval remains **withheld**. No public publication,
deployment, staging, or commit is performed. This record is separate from the
ZIP it identifies so the artifact does not contain its own recursive checksum.

## Package and reproducibility boundary

The [local verification ZIP](evidence/phase-6/dataworks-verification-final.zip) is
591,890 bytes, with 229 source files plus a content manifest. SHA256:

```text
2093427202d02ca4b2b561481052a7aae3df11ee8e3b0274e793d3cc720c1774
```

Two builds from identical source bytes and origin metadata produced identical
ZIP bytes on the recorded Python/zlib runtime. A separate agent reviewed the
exact inventory and every file hash/size. The canonical source snapshot is
`5d50a95f04c83ed722418a4e4027b91ebc726ebf975588310dc9e6921f77009e`.
The manifest distinguishes the starting Git revision `bb3f526` from the actual
content snapshot, which includes the current uncommitted implementation.

Rebuilding from the extracted, Git-free copy with the clean declared interpreter
also produced the identical ZIP/checksum. This demonstrates the included builder
works from its own content inventory rather than relying on the original checkout.

The bundle contains application code, tests, original synthetic fixtures and
independent expectations, architecture/persistence decisions, setup instructions,
short walkthrough, environment snapshots, and explicit release gates. It excludes
secrets configuration, Git metadata, environments, generated runs, and the legacy
CSV whose provenance is unknown. Its sole regression dependency was replaced by
an original four-row generic-ingestion fixture with exact text/null expectations.
The unrelated root CSV remains untouched.

The verifier checks listed content, executable-source coverage, fixture identity,
source stability, runtime/dependencies, nonempty/no-skip regression discovery,
and actual new-process saved-plan/project reproduction. It blocks mismatched
versions by default; the historical environment requires an explicit override.
It never turns test success into `release_ready: true`. Content hashes identify
bytes; they do not authenticate a publisher or establish distribution rights.

The extracted package passed all 559 tests and the actual fresh-process fulfillment
proof twice: strict mode using the freshly installed declared environment, and
explicit mismatch mode using the historical environment. The historical environment
was separately blocked with exit 2 in default mode, as required; bundle integrity
passed before that environment gate. Both successful runs rechecked manifest,
fixture, and source stability after execution and retained their actual logs.

Separate-copy results are recorded in the evidence index and final integrity
audit. The extracted copy has no Git metadata, environment, legacy CSV,
or generated projects; its own scripts locate its own sources. Reusing a selected
interpreter on the same Windows host is not independent human or second-host
reproduction. Those gates remain open even when all engineering checks pass.

## Declared environment and compatibility fixes

The owner's dependency pins remain unchanged. The initial sandboxed package
resolution failed, but a retry outside the network restriction resolved all
pins. Actual installation then succeeded in a new isolated Windows/Python 3.12.14
environment with system site-packages disabled. The original `.venv` was not
upgraded. The new environment's dependency compatibility check passed.

| Package | Historical environment | Fresh declared environment |
|---|---:|---:|
| Streamlit | 1.55.0 | 1.66.1 |
| pandas | 2.3.3 | 3.1.0 |
| NumPy | 2.4.2 | 2.5.4 |
| Plotly | 6.6.0 | 7.1.0 |
| scikit-learn | 1.8.0 | 1.9.1 |

The other seven direct pins match both environments. Full distributions are
recorded separately for each profile. Installation success did not imply runtime
compatibility: the first actual declared-environment suite had 17 failures and
three errors in 551 tests. Those failed logs are retained.

Pandas 3's default NaN-backed strings exposed real preservation defects. Automatic
text/identifier inference now preserves existing StringDtype storage and null
semantics, while an explicitly requested object conversion remains explicit.
Project serialization distinguishes NaN-backed from nullable `pd.NA` strings and
preserves object/string index and column storage without reinference. Older
two-field string descriptors remain readable; older DataWorks builds reject the
new unsupported NaN descriptor instead of silently changing its semantics.

Some tests had implicitly relied on pandas' old default object construction;
those intended object/None inputs now request it explicitly. Exact value/schema
assertions remain. Added checks cover both string null modes, source ownership,
and archive round trips. Streamlit test paths now use absolute repository paths,
and session reset assertions use public membership rather than a removed proxy.

Both environments now pass **559 tests**, with no failures, errors, or skips,
and the actual fresh-process fulfillment proof. The increase from Phase 5's 531
tests comprises 12 verifier, eight bundle, three codec, four ingestion/preparation,
and one text-storage case. The frozen fulfillment oracle is unchanged.

The new Streamlit AppTest initializes an asyncio loop that needs a local socket
pair. Stack traces diagnosed the sandboxed startup stall. Checks outside that
socket restriction completed; no application security setting was disabled and
no package downgrade or event-loop workaround was introduced.

## Workload qualification

The preserved declared environment passed the full F14 protocol: the exact
100,000-row mixed-quality input, one warm-up and ten measured samples, retained
history, and real archive reopening in separate processes. All eleven samples
matched the independent oracle and source/plan/run identities. Input and relevant
code fingerprints stayed unchanged throughout the run.

| Measurement, declared environment | Observed | Limit |
|---|---:|---:|
| Execution, nearest-rank p95 | 7.353 seconds | 30 seconds |
| Export, p95 | 12.838 seconds | 30 seconds |
| Cold reopen, p95 | 17.211 seconds | 30 seconds |
| Whole-process peak including imports/warm-up | 754.45 MiB | 1,024 MiB |

Counts remain 50,000 accepted, 50,000 quarantined, 70,000 reasons, 40,000 accepted
orders, and quantity 120,000; canonical values, status subtotals, and duplicate
references are checked, not just totals. The new measurements qualify the actual
new environment rather than extending Phase 4's older-package performance claim.
They apply to the recorded host/input/profile, not every admitted file or shared
concurrency. The separate UI and retained-month diagnostics have their own
observations in the evidence and are not additional ten-sample percentiles.

The single 100,000-row AppTest profile passed actual source loading, execution,
archive preparation, clearing, and reopening through the pages. Execution took
7.343 seconds, preparation 11.498 seconds, and reopening 18.085 seconds; whole-
process peak was 737.74 MiB. Identities/outputs matched and previews stayed bounded
to 100 rows. This is a history-free archive profile, not a UI p95 or real-browser
or operator study.

A separate retained-month diagnostic reopened the actual large A archive, loaded
a distinct B using the unchanged plan, injected a final reporting failure, and
retried. A's committed source/plan/run/report survived failure with no completed
B output. Retry produced the expected B rows with October dates and new source-
scoped identities. Peak was **952.93 MiB**, the highest observed in this phase,
below 1 GiB. Cold A reopening took 14.580 seconds, replacing source while retaining
A 7.835 seconds, injected failure 4.101 seconds, and successful retry 11.322 seconds.
The existing undo budget pruned two checkpoints to zero; no limit was raised.
Retain a downloaded project when recovery beyond that budget is needed.

That diagnostic reuses an unchanged historical script whose `client_date` field
is fixed to the earlier protocol date. Its invocation metadata and actual UTC
execution timestamps identify this run as 2026-10-30; the raw field is not used
as a claim that the new measurement occurred on the historical date.

## Release decision and remaining work

Windows declared-pin installation, compatibility/correctness, package integrity,
and the measured backend workload now have attributed evidence. The software has
substantially stronger engineering proof, but the original release conditions
include evidence that cannot be generated by synthetic tests or self-approval:

- Real operator confirmation, two permissioned consecutive representative exports,
  and the actual dashboard contract/accepted import.
- Unaided task/recovery observation and preparation-time comparison against the
  existing process. No demand, achieved savings, or market-fit claim is made.
- Independent human reproduction of the package and failure behavior.
- Owner licensing/distribution decision and applicable dependency obligations.
- Observed Linux verification before claiming that platform. Shared use additionally
  needs actual OIDC callback, TLS/proxy, isolation, and target-host checks.

The engineering decision is **local verification candidate; production release
withheld**. See [the evidence index](evidence/phase-6/README.md),
[reproduction guide](../../REPRODUCING.md), [release gates](../../RELEASE_GATES.md),
and [suggested Git message](phase-6-commit-message.txt). No next phase is started.
