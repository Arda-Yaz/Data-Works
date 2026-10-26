# Phase 1: monthly order-fulfillment acceptance specification

Prepared 2026-10-26 (Europe/Istanbul). Status: engineering specification and
synthetic fixtures prepared; application acceptance execution and real-operator
validation remain pending. Phase 2 requires the owner's approval.

## Target and evidence boundary

The proposed user is an operations analyst at a small online retailer who
prepares a monthly order-fulfillment report from one order-line CSV exported by
the same order-management system. The operations manager consumes the result
through an existing fulfillment dashboard. The intended benefit is less manual
preparation and fewer unexplained changes in reported order counts.

The owner selected this target; no real operator has yet confirmed it. The
engineering fixtures establish checkable requirements, not demand, achieved time
savings, or compatibility with a customer's actual dashboard. Before claiming
those outcomes, obtain one operator's confirmation, their current preparation
steps, two consecutive representative exports, permission/provenance for using
them, and the dashboard's required fields, types, status meanings, and date format.

The first proposed release target is local, single-user use with downloaded
`.dataworks` archives. This is a scope proposal, not a new persistence capability.
Shared hosting, OIDC, autosave, and crash recovery need their own verified gates.

## First-workflow business rules

These proposed rules are precise enough to implement and review. They need
confirmation against the real source/dashboard before a customer release.

1. Require exactly `order_id`, `line_id`, `order_date`, `sku`, `quantity`, and
   `fulfillment_status`. Input column order may vary; output uses that order.
   Missing, renamed, extra, or duplicate headers block the attempt. Do not
   silently ignore additional fields. Malformed CSV/encoding also blocks parsing.
2. Read identifiers as text. Preserve leading zeros and the original identifier
   bytes; never infer numeric IDs. Null, empty, or whitespace-only IDs are row
   errors (`missing_order_id` or `missing_line_id`). Nonempty IDs containing
   leading/trailing whitespace are quarantined as `identifier_whitespace`
   rather than silently changing identity. The supplied fixtures use clean IDs.
3. Accept only exact `YYYY-MM-DD` and `YYYY/MM/DD` calendar dates. Export dates as
   `YYYY-MM-DD`. Missing dates are `missing_order_date`; impossible, ambiguous,
   or unsupported dates are `invalid_order_date`. Both reject the row. Do not
   infer locale, timezone, or reporting period from a file name.
4. Trim SKU whitespace and uppercase letters. A missing/blank SKU rejects the
   row. No other SKU rewriting or product lookup is assumed.
5. Quantity must be a base-10 integer in signed-64 range, greater than zero.
   Trim surrounding whitespace; optional leading sign is permitted. Empty input
   is `missing_quantity`; malformed/fractional text is `invalid_quantity`;
   values outside signed-64 range are `quantity_out_of_range`; zero/negative
   values are `nonpositive_quantity`. Never impute quantities or remove numeric
   outliers in this workflow.
6. Trim/lowercase statuses, then map `shipped`, `dispatched`, `complete`, and
   `fulfilled` to `fulfilled`; `open`, `awaiting`, and `pending` to `pending`;
   `canceled` and `cancelled` to `cancelled`. Missing/blank status rejects the
   row. Any unseen nonempty value requires a decision and prevents commitment,
   even if the same row has other validation errors; those errors may still
   appear as uncommitted diagnostics. It must not be silently mapped or
   discarded. Cancelled lines remain accepted.
7. Evaluate all independent row rules at one declared validation stage, retaining
   all applicable reasons for each row. Reject invalid rows before duplicate
   classification. Duplicate classification examines the canonical valid rows.
8. The duplicate key is `(order_id, line_id)`. If every canonical field agrees,
   retain the earliest source record and quarantine later copies as
   `exact_duplicate`, referencing the retained source record. If any canonical
   payload differs, produce a conflict report and require an explicit decision;
   never automatically choose a winner. A conflict or unseen-status decision
   must be resolved through an attributed input/plan change and a new attempt.
9. Preserve input record order among accepted rows. Retain raw rejected records,
   immutable source-row identity, reason code(s), stage/operation attribution,
   and duplicate references. Transformations must not alter source evidence.

Blocked, decision-required, and internal-error attempts never replace the last
committed result. They may expose diagnostic candidate information labeled as
uncommitted. They do not report a completed accepted/rejected partition.

## Independent expected results

Fixtures and the manually authored oracle live in
[`tests/fixtures/fulfillment_v1`](../../tests/fixtures/fulfillment_v1/README.md).
Their versioned manifest freezes the inputs and expectations. No expected value
was obtained by running DataWorks transformations.

| Input | Source rows | Accepted | Rejected | Reason assignments | Accepted distinct orders | Accepted quantity |
|---|---:|---:|---:|---:|---:|---:|
| Month A | 12 | 4 | 8 | 8 | 3 | 5 |
| Month B | 8 | 5 | 3 | 3 | 4 | 11 |
| Month C invalid records | 9 | 2 | 7 | 9 | 2 | 3 |

Count reconciliation is by source records: accepted and rejected identities are
disjoint and their union equals all source identities for a completed run. A row
with three reasons is one rejected row. Distinct order counts cannot be added
across dispositions, because an order can have several lines and duplicates.
Reports must distinguish source lines, accepted lines, distinct accepted orders,
and quantities; status subtotals reconcile to their corresponding accepted totals.

Month B is a synthetic unchanged-plan challenge, not a claim of genuinely unseen
customer validation. Build/save the plan on A, record its version and digest,
then execute B in a fresh process without changing the plan or fitted artifacts.
Real held-out verification requires a subsequent representative customer export.

## Acceptance matrix

Every case below is a declared future acceptance requirement, not a passing
application result. Existing regression evidence is recorded separately.

| ID | Check and expected evidence |
|---|---|
| F01 | Month A accepted values, order, logical schema, raw rejected records, reasons, duplicate references, and report exactly match `expected.json`. |
| F02 | Save the A plan, exit, load it in a fresh process, apply unchanged to B; match B expectations and prove the plan digest did not change. B source identities differ from A. |
| F03 | `month_c_missing_quantity.csv` blocks with the missing field identified; prior committed B data/report/identity remain intact. |
| F04 | `month_c_extra_column.csv` blocks with `warehouse` identified. A permuted six-column B succeeds with canonical output order. Duplicate/renamed headers block. |
| F05 | C invalid-record input completes with exactly two accepted and seven quarantined records; invalid quantities/dates never silently coerce or fill. |
| F06 | Slash dates and status aliases normalize before canonical validation and duplicate classification. A records 3/4 are one retained line and one exact duplicate. |
| F07 | C record 2 carries three reasons but counts once. Source identities survive repeated pandas index labels, reset indices, and duplicate business keys. |
| F08 | C conflicting duplicates require a decision; identify key, both source records, and differing quantity. No automatic winner or partial commitment. |
| F09 | C unseen status requires a decision identifying `backordered`; no guessed mapping. Any decision is bound to the reviewed source and plan. |
| F10 | Inject an internal failure after an early successful transformation and before commitment; retain prior committed accepted/rejected data, report, plan/source identities, and archive preparation status consistently. |
| F11 | Export/reopen in a separate process reproduces source, plan, committed outputs, rejection evidence, and report. Prepared bytes alone are not durable saving; retain the actual archive. Unsupported/corrupt associations fail before mutation. |
| F12 | Undo restores coherent data/plan/report associations. Editing input or a draft plan visibly marks older results as earlier results. Omitting history on export explicitly limits undo. |
| F13 | Accepted CSV dates/IDs/statuses match the declared downstream format; raw and spreadsheet-safe exports have explicit semantics. Rejected export retains evidence. Dashboard integration remains pending its real format. |
| F14 | At the provisional workload below, measure repeated run/export/reopen wall time and whole-process peak RSS on a recorded host. Retention budgets alone do not prove performance. |
| F15 | One actual operator confirms the recurring problem; observe the existing process and DataWorks on two consecutive representative exports, verify identical agreed dashboard totals, and measure active preparation time. |

## Statistics and existing-feature scope

The fulfillment plan contains no median imputation, interpolation, z-score, or
IQR transformation. Preserve existing ingestion, EDA, chart, ML, cleaning, undo,
and export behavior through regression tests.

Generic reusable statistical steps still need an explicit per-step refit/frozen
policy, fitted values where applicable, fitting population, and versioned
semantics before being advertised as safely reusable. The separate arithmetic
characterization fixture distinguishes B median 250 (refit) from 25 (frozen A),
and B IQR cap 70 (refit) from 7 (frozen A). It does not claim either policy is
already persisted or implemented. Unsupported reusable operations must be clearly
rejected; do not broaden this first workflow to make demonstrations richer.

## Proposed measurable limits and value gate

Provisional engineering workload: one UTF-8 CSV, 100,000 parsed records, six
columns, at most 20 MiB input, within existing application limits. Future F14
measurement uses one warm-up and ten timed runs on a recorded Python 3.12 host;
nearest-rank p95 execution time must be at most 30 seconds. Measure export and
fresh-process reopen separately, at most 30 seconds each, and whole-process peak
RSS at most 1 GiB including imports. These are proposed acceptance thresholds,
not measured results or promises for unspecified hardware/concurrency. Freeze
the actual reference host and representative workload before performance sign-off.

For F15, record active preparation minutes, total elapsed time, manual decisions,
and corrections using the same two exports and agreed dashboard checks. Proposed
value threshold: at least 50% lower active preparation time on each export with
zero unexplained line/order-count differences. No existing-process timings have
been observed. A synthetic walkthrough or one operator does not establish market
fit, demand at scale, or funding value.

## Phase boundary and outstanding decisions

Phase 1 supplies this specification, original fixtures, independently calculated
expectations, the [code-level gap map](gap-map.md), and [baseline evidence](evidence/2026-10-26/README.md).
It makes no production behavior changes. Phase 2 will specify the versioned
source/plan/run/commit/persistence contracts and turn these requirements into
executable acceptance checks after owner approval.

Outstanding: confirm a real operator and process; obtain two permissioned exports
and dashboard contract; confirm proposed mapping/date/duplicate/unknown-column
policies and local release scope; freeze measured workload/reference host; choose
project licensing before redistribution. Real-user value and release sign-off
remain open while those requirements are unverified.
