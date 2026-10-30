# Data-Works

Data-Works is a Streamlit application for uploading tabular datasets, reviewing schema, cleaning data, exploring distributions, building charts, and running quick machine learning experiments.

## Features

- Upload CSV, TSV, TXT, Excel, JSON, and JSON Lines files with explicit reading options.
- Infer column data types and roles before analysis.
- Track cleaning operations with undo support.
- Handle missing values, duplicates, outliers, and allowed-value mismatches.
- Normalize text, map categories, and inspect rows using literal search and missing-value filters.
- Explore univariate statistics, correlations, and group-by summaries.
- Create and save Plotly charts in the current session.
- Train and compare basic classification and regression models.
- Retain completed experiments across reruns and download reports or fitted models.
- Export cleaned CSV, save/reopen projects, and verify recorded transformations.
- Prepare monthly order-fulfillment files with saved plans, traceable quarantine,
  reconciled reports, and complete-run commitment.

## Architecture

- `app.py`: Streamlit entry point and page navigation.
- `backend/state.py`: shared session state, dataset metadata, history, and undo.
- `backend/ingestion.py`: file parsing, schema inference, and dtype casting.
- `backend/cleaning.py`: data cleaning helpers.
- `backend/eda.py`: exploratory analysis helpers.
- `backend/visualization.py`: Plotly figure builders.
- `backend/preprocessing.py`: train/test split and model preprocessing.
- `backend/ml.py`: model registry, training, metrics, and ML plots.
- `backend/experiments.py`: retained experiment reports and model exports.
- `backend/operations.py`: parameterized transformation history and replay.
- `backend/text_transform.py`: explicit text normalization and exact category mapping.
- `backend/project_io.py`: versioned, data-only project archives.
- `backend/fulfillment.py`: source evidence, saved plans, and complete fulfillment execution.
- `backend/fulfillment_state.py`: committed results, recovery, and coherent fulfillment undo.
- `backend/fulfillment_wire.py`: compact persisted evidence with strict reference validation.
- `frontend/pages/`: Streamlit page implementations.
- `frontend/components/`: reusable UI widgets.

## Setup

Use **Python 3.12**. The CI workflow is configured for Windows and Linux; recorded
local verification uses Windows. Linux execution remains unverified in the
acceptance evidence. See [REPRODUCING.md](REPRODUCING.md) for the verification
package, recorded environment, and open release gates.

The unchanged declared pins passed a fresh Windows installation and 559 tests on
2026-10-30. The earlier environment also passed. The local verification package
and its strict environment/integrity checks are described in that reproduction
guide; passing them does not establish operator acceptance or production approval.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

## Notes

- Work lives in Streamlit session state until downloaded. Use **Project & Exports** to save a `.dataworks` file and open it in another session.
- Machine learning preprocessing fits encoders, imputers, and scalers on the training split, then applies them to the test split.
- The latest five completed ML experiments are retained. Reports preserve the settings and dataset version used by each run, even after further cleaning.
- Project files include original/working data, schema, operation parameters, saved charts, and experiment reports. Undo snapshots are optional. Reopened reports retain their results; retrain to obtain live models.
- Fitted model downloads include preprocessing and package versions in a separate joblib artifact. Only load trusted model files; the app never imports model binaries.
- CSV exports offer spreadsheet-safe text (default) or raw values. Safe export prefixes formula-like text with an apostrophe; project files preserve original types and values.
- Transformation history can be verified against the original upload and exported as a recipe. Older or custom operations without complete parameters are marked non-replayable.
- Project loading accepts only bounded JSON archives, without extracting files or executing serialized Python objects. Limits: 50 MiB compressed, 100 MiB JSON, and five million data values across frames, history, and reports. Omit undo snapshots when needed.
- Dataset changes clear dependent charts. Save a project before changing data if you need to retain its charts.

## Monthly order fulfillment

Open **Order Fulfillment** for the six-column order-line workflow: `order_id`,
`line_id`, `order_date`, `sku`, `quantity`, and `fulfillment_status`. Load the CSV,
review or import the saved plan, and click **Run fulfillment plan**. Input column
order may vary; missing/extra columns block execution. Dates must be explicit
`YYYY-MM-DD` or `YYYY/MM/DD`; identifiers preserve leading zeros. Quantities are
positive signed-64 integers and are never imputed. SKU/status normalization and
the complete rules are declared in [the acceptance specification](docs/acceptance/phase-1.md).

Exact canonical duplicates retain the first `(order_id, line_id)` record and
quarantine later copies with a source reference. Conflicting payloads and unseen
statuses require an explicit decision. Invalid rows retain raw values, source-row
identities, all applicable reasons, and validation stages. Completed reports
reconcile source lines to accepted/quarantined lines and distinguish distinct
orders from line and quantity totals. An unsuccessful attempt preserves the last
successful result; older results retain their source/plan associations.

The workspace overview separates the current source and plan from the latest
attempt and last successful result, and gives the next action. For an unseen
status, choose its meaning under **Resolve an unseen status**, apply the reviewed
mapping, then run again. Each decision creates a plan revision; no meaning is
preselected. Conflicting duplicates identify their source records and differing
fields so the source can be corrected explicitly.

**Inspect quarantine reason** filters the bounded preview only. The displayed
report totals and fulfillment download buttons cover the full successful run.
Streamlit's table toolbar exports only that displayed preview. The
page shows retained fulfillment undo checkpoints and warns when a change releases
checkpoints under the budget. Dataset undo in the sidebar is labeled separately.

Download accepted CSV, quarantine CSV, and the report from that page. Use
**Project & Exports** to save/reopen either a fulfillment-only or mixed project.
Archives retain source bytes/read options, saved plans, results, and optional
bounded undo history. Existing projects without fulfillment state still open.
Fulfillment archives use format version 2 and require a fulfillment-aware build;
older version 1 builds reject them rather than ignoring their new evidence.
Reopening verifies the new evidence against its source and plan before replacing
the current session. Retain the downloaded archive; preparation alone does not
provide durable storage.

The archive status distinguishes session-only changes from a prepared current
archive. DataWorks cannot confirm that the browser retained a download. The
chosen undo-history export option persists across navigation, and omitting
history visibly explains what will be unavailable after reopening.

The synthetic reproduction command writes actual saved plans/projects and checks
fresh-process reopening and unchanged-plan reuse. Choose a new output directory:

```powershell
python scripts/verify_fulfillment.py --output-dir .test-tmp/fulfillment-proof
```

The [synthetic fixtures](tests/fixtures/fulfillment_v1/README.md) define engineering
correctness. Real-operator confirmation, customer exports, dashboard integration,
observed time savings, and workload performance remain separate release gates.

The repeatable workload protocol uses 100,000 mixed-quality rows, one warm-up,
and ten measured samples, including real archive reopening in fresh processes.
Use a new output directory and the history-inclusive reference profile:

```powershell
python scripts/benchmark_fulfillment.py --include-history --output-dir .test-tmp/fulfillment-benchmark
```

The runner records code/input hashes, installed package versions, wall times,
whole-process peak memory, and correctness checks. Smaller pilots are explicitly
separate from a full protocol pass. See [the frozen protocol](docs/acceptance/phase-4-protocol.md).
Its results qualify only the recorded hardware, packages, and workload.

Fulfillment archives store each original source and actual attempt once, then
restore independently owned values. [The persistence notes](docs/acceptance/phase-4-persistence.md)
describe compatibility, identity checks, and unchanged resource limits. Existing
generic statistical recipe replay refits statistics on each input; frozen fitted
statistics are unsupported. The fulfillment plan excludes statistical operations.

The [Phase 5 observation sheet](docs/acceptance/phase-5-observation.md) defines the
still-pending unaided operator check. Automated synthetic interaction checks do
not establish customer usability, dashboard compatibility, or achieved savings.

## Uploading and transforming arbitrary datasets

Choose a file, adjust **File-reading options**, then click **Load Dataset**. Delimited files support a single custom delimiter and a text encoding such as `utf-8-sig`, `cp1254`, or `utf-16`. Excel uploads let you choose a sheet; `openpyxl` reads `.xlsx` and `xlrd` reads legacy `.xls` files.

JSON accepts records arrays, flat column-oriented objects, or a single flat record. To select nested records, enter an object-key path such as `["data", "records"]`. JSON Lines accepts one flat record per line. Nested cells must be flattened outside the app; unsupported structures fail explicitly.

Decimal and thousands separators guide numeric schema inference and conversion. Raw uploaded text is retained until **Confirm Schema**, where the conversion preview shows the proposed values. Separators can also be changed per column in the schema editor. Invalid digit grouping, numeric overflow, and underflow are rejected. Leading-zero identifiers and literal `NA`/`NULL` strings are preserved; empty delimited/Excel cells are missing.

Use **Inspect rows** on Upload to choose columns, search literal text, filter missing values, and browse at most 100 rows per page. These controls do not change the dataset.

The **Text & Categories** cleaning tab offers trimming, whitespace collapse, letter case, Unicode normalization, and exact category replacements. All are explicit operations with a preview and undo. Replacements run simultaneously, preserve unmatched/missing values, update allowed-value rules, and are included in saved projects and transformation recipes. The bundled CSV is only an unclean test fixture; these operations make no assumptions about its fields or domain.

## Performance and operational limits

EDA calculates only the active panel and reuses session-local results until data or controls change. Upload summaries calculate when opened. Charts display at most 10,000 sampled/reduced rows and 50 categories, with visible notices; category aggregates use the full dataset. Date lines retain the full ordered range when reduced.

Application budgets are centralized in `backend/resources.py`:

| Resource | Default limit |
|---|---|
| Upload | 50 MiB |
| Dataset | 1,000,000 rows, 10,000 columns, 5,000,000 cells, 128 MiB estimated storage |
| Undo | 10 snapshots and 128 MiB; oldest snapshots are released first |
| Saved charts | 10 charts, 5 MiB per chart, 20 MiB total JSON |
| Retained experiments | 5 runs, 64 MiB each, 128 MiB total estimated storage |
| ML run | 50,000 rows and 5,000,000 encoded values |
| Kernel SVM / KNN | 10,000 training rows / 20,000,000 train-test row pairs |

ML runs acquire one process-wide execution slot and use one numerical thread per library. Overlapping runs receive a retry message. These are practical workload budgets, not process memory guarantees; shared hosting also needs operating-system/container limits. See **Project & Exports → Session resource usage** for estimated retained storage.

## Tests and deployment

```shell
python scripts/run_tests.py
```

The runner uses cross-platform Python APIs and isolates Streamlit test files in
a disposable temporary directory. GitHub Actions is configured to install
dependencies, check compatibility and source syntax, then run the suite on
Windows and Linux. A configured job is not evidence that it passed; recorded
acceptance results qualify the stated Windows environment only.

See [DEPLOYMENT.md](DEPLOYMENT.md) for local startup, private shared deployment, optional native OIDC sign-in, JSON diagnostics, health probes, and hosting limits. Local startup binds to `127.0.0.1`. Shared deployment requires provider/proxy configuration and live callback verification. Licensing and the fixture's upstream provenance must be supplied by the project owner before redistribution.
