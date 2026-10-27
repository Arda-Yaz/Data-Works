"""Reproduce the synthetic fulfillment workflow using real persisted artifacts.

Run with --output-dir pointing to a new directory. Child processes reopen the A
and B projects and reuse the saved plan. No Streamlit session or mocks are used.
This is correctness evidence, not customer or workload-performance validation.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.fulfillment import dumps_plan, ingest_csv, plan_digest
from backend.fulfillment_state import (
    export_project, new_project, run_project, set_source, validate_project,
)
from backend.project_io import dumps_project, loads_project


FIXTURES = ROOT / "tests" / "fixtures" / "fulfillment_v1"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_fixture_manifest():
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        data = (FIXTURES / entry["path"]).read_bytes()
        require(len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"],
                "Frozen fixture changed: " + entry["path"])


def check_completed(project, filename):
    attempt = project["committed"]["attempt"]
    wanted = EXPECTED["completed_runs"][filename]
    require(attempt["status"] == "completed", "Run did not complete: " + filename)
    require(attempt["report"] == wanted["report"], "Report mismatch: " + filename)
    accepted = [{key: row[key] for key in ("source_record", "values")} for row in attempt["accepted"]]
    require(accepted == wanted["accepted"], "Accepted output mismatch: " + filename)
    rejected = []
    for record in attempt["rejected"]:
        item = {"source_record": record["source_record"], "reasons": sorted(record["reasons"])}
        if "duplicate_of_source_record" in record:
            item["duplicate_of_source_record"] = record["duplicate_of_source_record"]
        rejected.append(item)
    require(rejected == [{**row, "reasons": sorted(row["reasons"])} for row in wanted["rejected"]],
            "Rejected output mismatch: " + filename)
    return {"input": filename, "run_id": attempt["run_id"], "source_id": attempt["source_id"],
            "plan_digest": attempt["plan_digest"], "report": attempt["report"]}


def save(project, path):
    payload = {"mode": "fulfillment-only", "project_name": "Synthetic fulfillment acceptance",
               "fulfillment": export_project(project, include_history=False)}
    path.write_bytes(dumps_project(payload))


def reopen(path):
    payload = loads_project(path.read_bytes())
    require(payload.get("mode") == "fulfillment-only", "Unexpected project payload")
    return validate_project(payload["fulfillment"])


def child(arguments):
    project = reopen(arguments.project)
    before = plan_digest(project["draft_plan"])
    if arguments.input:
        project = run_project(set_source(project, ingest_csv(arguments.input.read_bytes())))
        require(plan_digest(project["draft_plan"]) == before, "Plan changed during reuse")
        require(arguments.write_project is not None, "Reuse needs an output project path")
        save(project, arguments.write_project)
        filename = arguments.input.name
    else:
        filename = arguments.expected_input
    result = check_completed(project, filename)
    result.update(status="passed", plan_digest_before=before,
                  streamlit_imported=any(name == "streamlit" or name.startswith("streamlit.") for name in sys.modules))
    require(not result["streamlit_imported"], "Execution imported Streamlit")
    print(json.dumps(result))


def invoke_child(*arguments):
    command = [sys.executable, str(Path(__file__).resolve()), *map(str, arguments)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, "Child workflow failed: " + result.stderr)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for actual plans, projects, and results")
    parser.add_argument("--project", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--write-project", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-input", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    check_fixture_manifest()
    if arguments.project:
        child(arguments)
        return
    if arguments.output_dir is None:
        parser.error("--output-dir is required")
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = new_project()
    plan = project["draft_plan"]
    digest = plan_digest(plan)
    (output / "saved-plan.json").write_bytes(dumps_plan(plan))
    project = run_project(set_source(project, ingest_csv((FIXTURES / "month_a.csv").read_bytes())))
    a = check_completed(project, "month_a.csv")
    save(project, output / "month-a.dataworks")
    reopened_a = invoke_child("--project", output / "month-a.dataworks", "--expected-input", "month_a.csv")
    reused_b = invoke_child("--project", output / "month-a.dataworks", "--input", FIXTURES / "month_b.csv",
                            "--write-project", output / "month-b.dataworks")
    reopened_b = invoke_child("--project", output / "month-b.dataworks", "--expected-input", "month_b.csv")
    require(all(item["plan_digest"] == digest for item in (a, reopened_a, reused_b, reopened_b)),
            "Persisted plan identity changed")
    project = reopen(output / "month-b.dataworks")
    prior = copy.deepcopy(project["committed"])
    noncommitting = []
    for filename, wanted in EXPECTED["noncommitting_attempts"].items():
        candidate = run_project(set_source(project, ingest_csv((FIXTURES / filename).read_bytes())))
        require(candidate["last_attempt"]["status"] == wanted["outcome"], "Wrong incompatible-input outcome")
        require(candidate["committed"] == prior, "An unsuccessful run replaced committed results")
        require(all(candidate["last_attempt"][key] is None for key in ("accepted", "rejected", "report")),
                "An unsuccessful attempt exposed completed outputs")
        validate_project(candidate)
        noncommitting.append({"input": filename, "outcome": wanted["outcome"], "prior_result_preserved": True})
    c = run_project(set_source(project, ingest_csv((FIXTURES / "month_c_invalid_rows.csv").read_bytes())))
    c_result = check_completed(c, "month_c_invalid_rows.csv")
    result = {"status": "passed", "scope": "synthetic correctness and fresh-process persistence; no customer/value/performance claim",
              "python": sys.version, "saved_plan_digest": digest, "month_a": a,
              "fresh_process_a_reopen": reopened_a, "fresh_process_b_reuse": reused_b,
              "fresh_process_b_reopen": reopened_b, "noncommitting_attempts": noncommitting,
              "month_c_invalid_rows": c_result}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
