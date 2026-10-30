# Original synthetic ingestion regression fixture

Created for DataWorks on 2026-10-30. These four rows are original synthetic test
values, with no customer data and no copied descriptions from the repository's
unattributed `surat_uncleaned.csv`. They are not a property dataset or a business
accuracy claim. The root CSV remains untouched and is not needed for these tests.

The independent expectation is preservation through parsing and inferred dtype
application: all nonempty tokens remain exact text, including compound values
`5 out of 10`, `12.5 credits`, `900 sq units`, the literal `NA`, and leading-zero
token `000900`. Each column has exactly one missing cell. No mixed compound-text
column is automatically coerced to numeric values. The regression declares every
expected cell and checks complete frame equality after inference.

This fixture replaces the regression's dependency on an input whose upstream
provenance and license have not been established. It does not grant a license to
the application or resolve redistribution rights for that unrelated root file.
