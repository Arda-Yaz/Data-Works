"""Run one headless Streamlit workload interaction and archive-reopen smoke.

Only the file-uploader boundary is supplied synthetic bytes. Real page rendering,
session adapters, execution, codec, and restoration run inside Streamlit AppTest.
This is one observed UI profile, not a real-browser check or ten-sample UI p95.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_fulfillment as benchmark


SCRIPT = '''
import streamlit as st
from backend.state import StateManager
from frontend.pages.data_fulfillment import data_fulfillment
from frontend.pages.data_project import data_project

StateManager.init_session()
page = st.selectbox("Workload verification page", ["Order Fulfillment", "Project & Exports"], key="workload_page")
if st.button("Clear verification session", key="workload_clear"):
    StateManager.clear()
    st.rerun()
if page == "Order Fulfillment":
    data_fulfillment()
else:
    data_project()
'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def implementation_hashes():
    paths = (
        "backend/fulfillment.py", "backend/fulfillment_state.py", "backend/state.py",
        "backend/project_io.py", "backend/resources.py", "frontend/pages/data_fulfillment.py",
        "frontend/pages/data_project.py", "frontend/project_status.py", "scripts/benchmark_fulfillment.py",
        "scripts/verify_fulfillment_workload_ui.py", "requirements.txt",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def clean_app(app, stage):
    exceptions = [element.value for element in app.exception]
    errors = [element.value for element in app.error]
    require(not exceptions and not errors,
            f"AppTest failed at {stage}: exceptions={exceptions!r}; errors={errors!r}")


def output_summary(app, rows):
    """Inspect every output against the benchmark's independently fixed oracle."""
    from backend.fulfillment import plan_digest
    from backend.fulfillment_state import current_result

    project = app.session_state["fulfillment"]
    attempt = benchmark.check_project(project, rows)
    require(current_result(project), "The committed result is incorrectly marked as earlier output.")
    require(project["last_attempt"]["run_id"] == attempt["run_id"], "Latest and committed successful run differ.")
    accepted_ids = {item["row_id"] for item in attempt["accepted"]}
    rejected_ids = {item["row_id"] for item in attempt["rejected"]}
    require(not accepted_ids & rejected_ids and accepted_ids | rejected_ids == set(project["source"]["row_ids"]),
            "Accepted/quarantine identities do not reconcile to the uploaded source.")
    for record in attempt["rejected"]:
        ordinal = record["source_record"]
        require(record["raw"] == dict(zip(project["source"]["columns"], project["source"]["rows"][ordinal - 1])),
                "Quarantine output lost original source evidence.")
    groups = rows // 10
    multiple_reasons = sum(len(item["reasons"]) > 1 for item in attempt["rejected"])
    duplicates = sum(item["reasons"] == ["exact_duplicate"] for item in attempt["rejected"])
    require(multiple_reasons == groups and duplicates == groups,
            "Multi-reason quarantine or exact-duplicate counts differ from the synthetic oracle.")
    metrics = {element.label: int(element.value.replace(",", "")) for element in app.metric}
    report = benchmark.expected_report(rows)
    require(metrics == {"Source lines": rows, "Accepted lines": report["accepted_rows"],
                        "Quarantined lines": report["rejected_rows"], "Accepted orders": report["accepted_distinct_orders"]},
            "Displayed report metrics differ from independently expected totals.")
    rendered_rows = [len(element.value) for element in app.dataframe]
    require(len(rendered_rows) >= 4 and all(count <= 100 for count in rendered_rows),
            "A Streamlit preview rendered beyond its declared 100-row scope.")
    require(app.session_state["df"] is None, "Fulfillment smoke unexpectedly created a generic dataset.")
    # Keep only small identity/report values. Retaining a whole old project or
    # decoding a second archive before clearing would artificially inflate RSS.
    return {"run_id": attempt["run_id"], "source_id": attempt["source_id"],
            "plan_id": attempt["plan_id"], "plan_revision": attempt["plan_revision"],
            "plan_digest": plan_digest(project["draft_plan"]), "report": report,
            "source_sha256": project["source"]["raw_sha256"],
            "multi_reason_quarantined_rows": multiple_reasons,
            "exact_duplicate_rows": duplicates, "rendered_table_rows": rendered_rows,
            "history_snapshots": len(project["history"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for incremental results and actual archive")
    parser.add_argument("--input", type=Path, help="Existing exact CSV produced by benchmark_fulfillment.generate_workload")
    parser.add_argument("--rows", type=int, default=100_000)
    arguments = parser.parse_args()
    try:
        expected = benchmark.expected_report(arguments.rows)
    except ValueError as exc:
        parser.error(str(exc))
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {
        "protocol_version": 1, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one headless Streamlit AppTest workload profile; not a real browser, fresh-process reopen, customer validation, or ten-sample UI p95",
        "f14_full_protocol_claimed": False,
        "configuration": {"rows": arguments.rows, "samples": 1, "include_history": False, "apptest_timeout_seconds": 120},
        "mocks": ["streamlit.file_uploader returns actual synthetic CSV/project bytes"],
        "measurement_limits": "Whole-process high-water RSS includes imports, AppTest/protobuf/session overhead and all prior stages. Source CSV bytes and actual archive bytes are retained for boundary uploads; no original full project copy or standalone decoded archive is retained.",
        "hardware": benchmark.hardware(), "installed_packages": benchmark.versions(),
        "dependency_claim": "Installed versions recorded only; no clean-install or declared-pin compatibility claim.",
        "implementation_sha256": implementation_hashes(), "phase_seconds": {}, "phase_peak_rss_bytes": {},
        "expected_report": expected,
    }

    def save():
        benchmark.write_report(output / "results.json", result)

    def perform(stage, operation, *, app=None):
        result["stage"] = stage
        save()
        begin = time.perf_counter()
        try:
            return operation()
        finally:
            result["phase_seconds"][stage] = time.perf_counter() - begin
            result["phase_peak_rss_bytes"][stage] = benchmark.peak_rss_bytes()
            if app is not None:
                clean_app(app, stage)
            save()

    previous_temp = tempfile.tempdir
    try:
        data = arguments.input.read_bytes() if arguments.input else benchmark.generate_workload(arguments.rows)
        if arguments.input:
            require(hashlib.sha256(data).digest() == hashlib.sha256(benchmark.generate_workload(arguments.rows)).digest(),
                    "--input is not the exact deterministic workload for --rows; this oracle cannot judge arbitrary files.")
        require(0 < len(data) <= benchmark.MAX_INPUT_BYTES, "Input exceeds the declared 20 MiB workload budget.")
        result["input"] = {"path": str(arguments.input.resolve()) if arguments.input else str(output / "workload.csv"),
                           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if not arguments.input:
            (output / "workload.csv").write_bytes(data)
        save()
        with tempfile.TemporaryDirectory(prefix="apptest-", dir=output) as directory:
            require(Path(directory).resolve().is_relative_to(output), "AppTest scratch path left its declared output directory.")
            tempfile.tempdir = directory
            begin = time.perf_counter()
            from streamlit.testing.v1 import AppTest
            result["phase_seconds"]["apptest_imports"] = time.perf_counter() - begin
            result["phase_peak_rss_bytes"]["apptest_imports"] = benchmark.peak_rss_bytes()
            app = AppTest.from_string(SCRIPT, default_timeout=120)
            uploads = {"fulfillment_source_file": BytesIO(data)}
            with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: uploads.get(kwargs.get("key"))):
                perform("initial_render", lambda: app.run(), app=app)
                perform("load_source", lambda: app.button(key="fulfillment_load").click().run(), app=app)
                require(len(app.session_state["fulfillment"]["source"]["rows"]) == arguments.rows,
                        "The loaded source record count differs from the actual input.")
                perform("execute_plan", lambda: app.button(key="fulfillment_run").click().run(), app=app)
                before = perform("oracle_before_save", lambda: output_summary(app, arguments.rows), app=app)
                require(before["source_sha256"] == result["input"]["sha256"], "The executed source differs from uploaded bytes.")
                result["before_archive"] = before
                perform("switch_to_project", lambda: app.selectbox(key="workload_page").set_value("Project & Exports").run(), app=app)
                perform("disable_history", lambda: app.checkbox(key="project_include_history").uncheck().run(), app=app)
                perform("prepare_archive", lambda: app.button(key="project_prepare").click().run(), app=app)
                archive_data = app.session_state["prepared_project_download"]["data"]
                archive_path = output / "fulfillment-without-history.dataworks"
                perform("archive_write", lambda: archive_path.write_bytes(archive_data))
                result["archive"] = {"path": str(archive_path), "bytes": len(archive_data),
                                     "sha256": hashlib.sha256(archive_data).hexdigest(), "include_history": False}
                uploads["fulfillment_source_file"] = None
                perform("clear_session", lambda: app.button(key="workload_clear").click().run(), app=app)
                require(app.session_state["fulfillment"]["source"] is None
                        and app.session_state["fulfillment"]["committed"] is None,
                        "The full session clear did not remove the old source and result.")
                uploads["project_file"] = BytesIO(archive_data)
                perform("enable_project_open", lambda: app.run(), app=app)
                perform("reopen_project", lambda: app.button(key="project_open").click().run(), app=app)
                perform("switch_back_to_fulfillment", lambda: app.selectbox(key="workload_page").set_value("Order Fulfillment").run(), app=app)
                after = perform("oracle_after_reopen", lambda: output_summary(app, arguments.rows), app=app)
                result["after_reopen"] = after
                fields = ("run_id", "source_id", "plan_id", "plan_revision", "plan_digest", "report", "source_sha256",
                          "multi_reason_quarantined_rows", "exact_duplicate_rows")
                require(all(before[field] == after[field] for field in fields),
                        "Reopening through the actual project page changed run/source/plan/output evidence.")
                require(after["history_snapshots"] == 0, "Archive reopened with history despite explicit omission.")
            del app
        require(implementation_hashes() == result["implementation_sha256"], "Implementation changed during the UI workload profile.")
        peak = benchmark.peak_rss_bytes()
        result["observed_gates"] = {"correctness_and_identity": True,
                                    "whole_process_peak_at_most_1_gib": peak <= benchmark.MAX_RSS_BYTES}
        result["status"] = "passed" if all(result["observed_gates"].values()) else "failed"
        result["stage"] = "complete"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:2000])
    finally:
        tempfile.tempdir = previous_temp
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["whole_profile_seconds"] = time.perf_counter() - started
        try:
            result["peak_rss_bytes"] = benchmark.peak_rss_bytes()
        except Exception as exc:
            result.update(status="failed", memory_measurement_error=type(exc).__name__)
        save()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
