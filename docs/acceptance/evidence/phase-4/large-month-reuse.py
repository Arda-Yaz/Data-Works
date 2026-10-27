"""One fresh-process 100k A -> B reuse/recovery diagnostic, not F14 p95.

Reuses the actual full-benchmark Month A archive. Only small identity/report
summaries survive transitions; no full previous-project deepcopy is retained.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_fulfillment as benchmark


OUTPUT = Path(__file__).with_suffix(".json")
BENCHMARK = Path(__file__).parent / "benchmark-full-with-history"
ROWS = 100_000


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def summary(project):
    attempt = project["committed"]["attempt"]
    return {"run_id": attempt["run_id"], "source_id": attempt["source_id"],
            "plan_digest": attempt["plan_digest"], "report": attempt["report"],
            "history_snapshots": len(project["history"])}


def run(benchmark_dir, output):
    require(not output.exists(), "Preserve the existing diagnostic; choose a new output path before rerunning.")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {"status": "running", "client_date": "2026-10-27",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "one fresh-process large prior-result reuse/recovery diagnostic; no ten-sample F14 or UI claim",
              "rows": ROWS, "input_archive": str(benchmark_dir / "sample-00.dataworks"),
              "input_csv": str(benchmark_dir / "workload.csv"), "phase_seconds": {}, "phase_peak_rss_bytes": {},
              "history_counts": {}, "installed_packages": benchmark.versions(),
              "implementation_sha256": benchmark.implementation_hashes(),
              "source_changes": ["2026-09 -> 2026-10", "2026/09 -> 2026/10"],
              "invalid_date_preserved": "2026-02-30",
              "history_policy": "10 checkpoints / 128 MiB; source change and success attempt checkpoints, evict oldest/oversized snapshots",
              "additional_export_reopen": "omitted: diagnostic focuses on retaining prior large A during B failure/retry; fresh-process archives are measured separately"}

    def save():
        benchmark.write_report(output, result)

    def measured(stage, operation):
        result["stage"] = stage
        save()
        begin = time.perf_counter()
        try:
            return operation()
        finally:
            result["phase_seconds"][stage] = time.perf_counter() - begin
            peak = benchmark.peak_rss_bytes()
            result["phase_peak_rss_bytes"][stage] = peak
            save()
            require(peak <= benchmark.MAX_RSS_BYTES, f"Whole-process peak exceeds 1 GiB at {stage}: {peak} bytes")

    save()
    try:
        begin = time.perf_counter()
        from backend.fulfillment import ingest_csv, plan_digest
        from backend.fulfillment_state import current_result, run_project, set_source, validate_project
        from backend.project_io import loads_project
        result["phase_seconds"]["imports"] = time.perf_counter() - begin
        project = measured("cold_reopen_a", lambda: validate_project(
            loads_project((benchmark_dir / "sample-00.dataworks").read_bytes())["fulfillment"]))
        benchmark.check_project(project, ROWS)
        before = summary(project)
        result["prior_a"] = before
        result["history_counts"]["reopened_a"] = len(project["history"])
        original_bytes = (benchmark_dir / "workload.csv").read_bytes()
        require(project["source"]["raw_sha256"] == hashlib.sha256(original_bytes).hexdigest(),
                "Actual archive A does not match actual benchmark source A.")
        next_bytes = original_bytes.replace(b"2026-09", b"2026-10").replace(b"2026/09", b"2026/10")
        require(next_bytes != original_bytes and next_bytes.count(b"2026-02-30") == ROWS // 10,
                "Synthetic B transformation failed or altered the deliberate invalid date.")
        del original_bytes
        result["next_input_sha256"] = hashlib.sha256(next_bytes).hexdigest()
        project = measured("set_source_b_retaining_a", lambda: set_source(project, ingest_csv(next_bytes)))
        new_source_id = project["source"]["source_id"]
        require(new_source_id != before["source_id"] and not current_result(project),
                "Month B did not create distinct source identity and an earlier A result.")
        require(plan_digest(project["draft_plan"]) == before["plan_digest"], "Plan changed while loading B.")
        result["history_counts"]["after_set_source_b"] = len(project["history"])
        with patch("backend.fulfillment._report", side_effect=RuntimeError("injected synthetic final-stage failure")):
            project = measured("injected_b_failure_retaining_a", lambda: run_project(project))
        failed = project["last_attempt"]
        require(failed["status"] == "failed" and failed["source_id"] == new_source_id
                and all(failed[key] is None for key in ("accepted", "rejected", "report")),
                "Injected B failure exposed completed outputs or the wrong source.")
        require(all(summary(project)[field] == before[field] for field in ("run_id", "source_id", "plan_digest", "report")),
                "B failure changed the prior A committed result.")
        require(all(item["row_id"].startswith(before["source_id"] + ":")
                    for item in project["committed"]["attempt"]["accepted"] + project["committed"]["attempt"]["rejected"]),
                "A committed rows were relabeled to another source.")
        benchmark.check_project(project, ROWS)
        result["failed_b"] = {"run_id": failed["run_id"], "source_id": failed["source_id"],
                              "status": failed["status"], "prior_a_preserved": True, "outputs_are_none": True}
        del failed
        result["history_counts"]["after_failed_b"] = len(project["history"])
        project = measured("retry_b_commit", lambda: run_project(project))
        after = summary(project)
        require(after["report"] == benchmark.expected_report(ROWS), "Month B report differs from independent arithmetic.")
        require(after["source_id"] == new_source_id and after["run_id"] != before["run_id"]
                and after["plan_digest"] == before["plan_digest"] and current_result(project),
                "Successful B reused a prior identity or changed the saved plan.")
        require(all(item["values"][2].startswith("2026-10-") for item in project["committed"]["attempt"]["accepted"]),
                "A September date survived in accepted Month B output.")
        require(all(item["row_id"].startswith(new_source_id + ":")
                    for item in project["committed"]["attempt"]["accepted"] + project["committed"]["attempt"]["rejected"]),
                "Successful Month B output reused Month A row identities.")
        result["completed_b"] = after
        result["history_counts"]["after_completed_b"] = len(project["history"])
        result["history_pruned"] = result["history_counts"]["after_set_source_b"] < result["history_counts"]["reopened_a"] + 1
        require(not any(name == "streamlit" or name.startswith("streamlit.") for name in sys.modules),
                "Pure reuse diagnostic imported Streamlit.")
        result.update(status="passed", stage="complete", unchanged_plan=True, accepted_dates_all_october=True)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:2000])
    finally:
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["total_seconds"] = time.perf_counter() - started
        result["peak_rss_bytes"] = benchmark.peak_rss_bytes()
        result["whole_process_peak_at_most_1_gib"] = result["peak_rss_bytes"] <= benchmark.MAX_RSS_BYTES
        if not result["whole_process_peak_at_most_1_gib"]:
            result["status"] = "failed"
        save()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK,
                        help="Existing full benchmark directory containing sample-00.dataworks and workload.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="New JSON output path; existing evidence is never overwritten")
    arguments = parser.parse_args()
    return run(arguments.benchmark_dir.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
