# Phase 5: fulfillment workspace verification protocol

Date: 2026-10-29 (Europe/Istanbul). The owner approved continuing Phase 5 after
the Phase 4 boundary. Pause before Phase 6. The starting checkout was clean at
`91d6774` (the owner committed the earlier fulfillment work). This phase does
not stage files, create a commit, publish a package, or approve production release.

## Scope and intended behavior

Refine the existing fulfillment and project pages around the proven state model.
Keep execution, versioned persistence, business rules, budgets, dependency pins,
and the frozen Phase 1 inputs/oracle unchanged.

- Show current source, draft plan, latest attempt for those inputs, and the last
  successful result separately, with a concrete next action.
- Explain incompatible columns, unmapped statuses, conflicting duplicates, and
  failed stages in ordinary language; retain exact evidence and identifiers.
- Require an explicit meaning for each unseen status. A choice belongs to its
  run and source status, creates a plan revision, and requires a new execution.
- Make source sampling and quarantine filtering scope explicit. Totals and all
  downloads continue to describe the full committed run, even after failure.
- Explain retained undo limits and actual checkpoint releases. Never imply that
  a large source replacement is guaranteed to be undoable.
- Distinguish session state from prepared archive bytes and an actually retained
  download. Preserve the selected history-export option across navigation.

## Engineering verification

Use actual page interactions with synthetic inputs through Streamlit AppTest.
Mock only file upload transport and, where checked, capture the actual download
callbacks to inspect their bytes. Inject one reporting failure through the
existing engine seam. Check decisions, stale state, filtering, undo retention,
save options, and actual archive reopening against the independent oracle.

Run the complete regression suite using `python scripts/run_tests.py`, syntax
checks, the existing fresh-process fulfillment reproduction runner, fixture
integrity checks, and `git diff --check`. Run the existing 100,000-row headless
UI profile against the changed pages, recording exact implementation hashes and
installed packages. It is one UI profile, not a new ten-sample F14 percentile.
The Phase 4 backend measurements remain historical evidence for their recorded
implementation and environment; compare backend hashes before reusing that claim.

## Operator observation gate

An operations analyst and two permissioned representative monthly exports have
not been identified. Synthetic engineering checks cannot satisfy the original
Stage 5 unaided-user exit gate. Do not claim user completion, dashboard acceptance,
measured preparation-time savings, or release readiness without that evidence.
Use the separate observation sheet to record the real task, assistance, mistakes,
recovery, counts, and timings once an operator is available.

The changed dependency pins remain preserved by the owner's instruction. Tests
qualify the installed environment only; no clean installation of those pins is
claimed. Phase 6 and production approval remain separate owner decisions.
