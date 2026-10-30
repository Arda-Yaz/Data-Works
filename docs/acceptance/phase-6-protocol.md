# Phase 6: reproducible package and release audit

Date: 2026-10-30 (Europe/Istanbul). The owner approved this phase. Starting
checkout was clean at `bb3f526`; the owner committed Phase 5. No staging, commit,
public upload, deployment, or release approval is implied by preparing a local
verification bundle.

Prepare an explicit source/fixture/document allowlist, an exact-content manifest,
deterministic ZIP construction, setup instructions, and a short walkthrough.
Exclude secrets, Git metadata, environments, generated runs, and the unattributed
legacy CSV. Replace its sole regression-test dependency with original synthetic
data while retaining the tested text/missingness behavior. Keep the fulfillment
oracle, production behavior, budgets, and owner-edited dependency pins unchanged.

Verify the bundle twice for deterministic bytes, extract its checked members into
a new directory, then run the actual regression and fresh-process fulfillment
proof from that separate source tree. Record interpreter/packages, source hashes,
commands, actual output logs, and environment differences. Use no original-checkout
imports or Git assumptions in the extracted verifier. Verification by another
agent/process on the same host is not an independent human, clean installation,
or second-host reproduction.

The verifier must block a package-version mismatch by default. An explicit
recorded-environment override may run engineering checks while keeping declared-
pin installation unverified. Capture a fresh dependency-resolution attempt; do
not change pins to hide its result. Install into a separate environment only if
packages can actually be resolved. Reusing installed packages must be labeled.

Audit real operator/value/dashboard evidence, licensing/provenance, Windows/Linux
coverage, current dependency installation, and applicable deployment controls.
Synthetic success cannot satisfy external gates. Publish the local artifact and
its checksum to this chat for review, with a Git commit message and a release
decision grounded in the observed results. Public redistribution remains gated
by the owner's licensing/provenance decisions and verified target environment.
