# Phase 5: fulfillment workspace results

Date: 2026-10-29 (Europe/Istanbul). Workspace implementation and engineering
verification are complete on the recorded installed environment. The original
Stage 5 unaided-operator exit gate remains open. Pause before Phase 6 for the
owner's approval; production release approval remains withheld.

## Implemented workspace

The overview shows the current source, draft plan, latest attempt for those
inputs, and successful result separately, with a concrete next action. Source
previews and advanced mapping details collapse so routine actions are reachable.
Incompatible columns, unknown statuses, conflicts, and failed stages have readable
guidance while exact evidence and identities remain available.

Unseen statuses have an explicit meaning selector starting without a business
choice. Each control belongs to its run and source status. Applying a reviewed
choice creates a plan revision before rendering, clears the old attempt, and
requires another run. Conflicting duplicates identify their business key, source
records, and differing fields; the interface never chooses a winner.

Quarantine inspection filters only the bounded preview, with matching/total counts.
Reports and dedicated fulfillment downloads remain bound to the full successful
run, including when newer inputs fail. The table toolbar's own CSV export is
explicitly labeled as preview-only. No execution scope or backend rules changed.

Undo shows retained checkpoints and budget limits, with a notice when a transition
evicts or skips checkpoints. Dataset undo is labeled separately. Archive status
distinguishes session-only work from current prepared bytes; it never claims
receipt of a browser download. History-export preferences survive navigation,
and omitted history explains what cannot be recovered after reopening.

## Problems found and resolved

- Mid-render reruns left old decision widgets in the installed AppTest tree after
  a shorter render with no prior result. Applying the decision in a callback
  before rendering resolves that lifecycle problem. Consecutive unknown statuses,
  changed selections, source replacement, and undo now require the proper explicit
  decision for their context.
- First-ever failure wording implied an earlier success existed. Failure guidance
  now says the attempt committed no result, while the overview shows whether a
  previous successful result actually exists.
- A no-history archive could be described as noncurrent after navigation removed
  its widget preference. A separate session preference preserves that choice.
- The real browser exposed native preview-download buttons that made generic
  full-download captions ambiguous. Captions now name the full-result buttons
  explicitly; the native table export retains its preview behavior.

Same-page reopening over a differently named project was investigated. Existing
widget-reset logic already preserves the restored name; a regression check now
demonstrates it. No backend or persistence-format change was necessary.

## Verification results

The complete suite passes **531 tests**, with no failures, errors, or skips: the
516 earlier tests plus 11 new workspace journeys and four new project-page cases.
The checks inspect actual downloaded CSV/report bytes, independent expected
values, coherent undo, failure preservation, and actual archive reopening.
Source syntax, frozen fixture hashes, fresh-process saved-plan reuse/reopening,
and `git diff --check` pass. Git HEAD and staging remain unchanged from `91d6774`.

A final 100,000-row headless application profile loaded, ran, prepared an archive
without history, cleared the session, and reopened the actual archive through
the real pages. It preserved source/plan/run identities, 50,000 accepted rows,
50,000 quarantined rows, 70,000 reasons, 40,000 accepted orders, quantity 120,000,
10,000 multi-reason rows, and 10,000 duplicate references. Preview tables remained
bounded to 100 rows. Both correctness and the observed 1 GiB memory gate passed.

| Final UI profile measurement | Observed |
|---|---:|
| Execution | 9.093 seconds |
| Archive preparation | 13.240 seconds |
| Actual archive reopening | 16.264 seconds |
| Whole-process peak, including imports/AppTest | 735.32 MiB |

This is one headless UI profile on the recorded Windows/Python host, not a new
ten-sample F14 percentile or cold-process UI restart. The Phase 4 benchmark's
recorded backend/workload/requirements hashes still match; its published backend
measurements remain evidence for that exact reference profile, not new browser
latency claims. Earlier Phase 5 runs are retained and distinguished from the final
wording verification.

A separate [rendered browser smoke](evidence/phase-5/browser-smoke.md) checked the
actual pages with labeled synthetic transport and arrow/Tab/Enter interaction.
It verified a reviewed plan revision without automatic execution or replacement
of the previous report. This adds browser evidence, not a real-user completion
study or an accessibility conformance claim.

## Remaining gates and engineering decision

Changed dependency pins remain untouched, as requested. Tests use Streamlit 1.55.0,
pandas 2.3.3, NumPy 2.4.2, Plotly 6.6.0, and scikit-learn 1.8.0, rather than the five
changed manifest versions. A clean installation of those pins remains unverified.
The actual environment and all twelve installed/declared versions are recorded.

Real operator confirmation, two permissioned representative exports, dashboard
compatibility, unaided completion/recovery, and measured preparation-time benefit
remain pending. [The observation sheet](phase-5-observation.md) gives the tasks
and measures to collect without inventing outcomes. Linux, licensing, and any
applicable hosting controls remain release gates from earlier phases.

The UI changes are implemented and synthetically verified. Stage 5 operator
acceptance and production readiness are **not established**. See the
[evidence index](evidence/phase-5/README.md) and
[suggested Git commit message](phase-5-commit-message.txt). No commit was created.
