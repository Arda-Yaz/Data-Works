# Synthetic monthly fulfillment fixtures, version 1

These are original synthetic engineering fixtures created for DataWorks on
2026-10-26. They contain no customer exports, personal data, or third-party data.
The proposed operator and workflow were supplied by the project owner in this
chat. No actual operator, observed preparation time, or dashboard integration
has been validated. Repository licensing remains the owner's decision.

- `month_a.csv`: establish a plan and compare against independent expectations.
- `month_b.csv`: unchanged-plan compatibility challenge with fresh records.
- `month_c_invalid_rows.csv`: row quarantine, including three reasons on one row.
- Other `month_c_*.csv`: schema blockers and decisions that prevent commitment.
- `expected.json`: manually specified canonical records, source-record references,
  row dispositions, and report totals. It is an acceptance oracle, not a run result.
- `statistical_characterization.json`: arithmetic examples for generic cleaning
  policy tests. The fulfillment plan must not impute quantities or remove outliers.
- `manifest.json`: frozen SHA-256 inventory of fixture inputs and expectations.

Record numbers are 1-based parsed data rows, excluding the CSV header. They must
eventually map to immutable source-scoped identities; they are not pandas index
labels or business keys. Rejected exports must retain the original record and
its reasons; accepted records retain input order after exclusions.

The executable behavior and phase gates are specified in
[the Phase 1 acceptance specification](../../../docs/acceptance/phase-1.md).
No production runner for these new acceptance cases is implemented in Phase 1.

Month B is a synthetic challenge, authored alongside its expected results. It
is not a genuinely unseen customer export. Later implementation must save the
Month A plan before exercising B, record its digest, and prove B caused no plan
modification. Changing fixtures or expected results requires a documented reason,
a new manifest, and review; never regenerate expectations from production output.
