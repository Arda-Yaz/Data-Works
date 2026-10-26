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
- `frontend/pages/`: Streamlit page implementations.
- `frontend/components/`: reusable UI widgets.

## Setup

Use **Python 3.12**. This is the supported runtime checked by the Windows/Linux CI workflow.

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

The runner works across Windows/Linux and isolates Streamlit test files in a disposable temporary directory. GitHub Actions installs dependencies, checks compatibility and source syntax, then runs the suite on both operating systems.

See [DEPLOYMENT.md](DEPLOYMENT.md) for local startup, private shared deployment, optional native OIDC sign-in, JSON diagnostics, health probes, and hosting limits. Local startup binds to `127.0.0.1`. Shared deployment requires provider/proxy configuration and live callback verification. Licensing and the fixture's upstream provenance must be supplied by the project owner before redistribution.
