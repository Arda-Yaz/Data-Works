"""Measure F14 using fresh processes and a deterministic mixed-quality CSV.

Default protocol: 100,000 rows, one warm-up, ten measured samples. Each sample
executes/exports in a new process and reopens in another new process. Peak RSS
is the operating system's whole-process high-water mark, including imports.
Smaller pilots never constitute an F14 pass. Archives omit history by default;
--include-history explicitly selects the separate history-inclusive profile.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COLUMNS = ("order_id", "line_id", "order_date", "sku", "quantity", "fulfillment_status")
MAX_INPUT_BYTES = 20 * 1024**2
MAX_SECONDS = 30.0
MAX_RSS_BYTES = 1024**3
PACKAGES = ("streamlit", "pandas", "numpy", "plotly", "scikit-learn", "scipy",
            "joblib", "threadpoolctl", "Authlib", "openpyxl", "xlrd", "watchdog")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_report(rows):
    """Independent arithmetic: each ten-row group has five accepted lines."""
    require(type(rows) is int and rows > 0 and rows % 10 == 0,
            "Choose a positive row count divisible by ten.")
    groups = rows // 10
    return {
        "source_rows": rows, "accepted_rows": 5 * groups, "rejected_rows": 5 * groups,
        "rejection_reason_assignments": 7 * groups, "accepted_distinct_orders": 4 * groups,
        "accepted_quantity": 12 * groups,
        "accepted_line_counts_by_status": {"fulfilled": 2 * groups, "pending": 2 * groups, "cancelled": groups},
        "accepted_quantity_by_status": {"fulfilled": 6 * groups, "pending": 3 * groups, "cancelled": 3 * groups},
    }


def workload_group(group):
    """Raw lexemes deliberately differ for a canonical exact duplicate."""
    first, second, third, fourth = (f"{group * 4 + offset:010d}" for offset in range(4))
    return [
        [first, "01", "2026-09-01", " sku-001 ", "2", " Shipped "],
        [first, "01", "2026/09/01", "SKU-001", "2", "complete"],
        [first, "02", "2026-09-01", "sku-002", "1", "Open"],
        [second, "01", "2026/09/02", "sku-003", "3", "CANCELLED"],
        [second, "02", "2026-09-03", "sku-004", "0", "pending"],
        [second, "03", "2026-02-30", "sku-005", "1", "pending"],
        ["", "01", "2026-09-04", "   ", "-1", "shipped"],
        [third, "01", "2026-09-05", "sku-006", "4", "Dispatched"],
        [third, "", "2026-09-06", "sku-007", "1", "pending"],
        [fourth, "01", "2026/09/07", "sku-008", "2", "Awaiting"],
    ]


def generate_workload(rows):
    expected_report(rows)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(COLUMNS)
    for group in range(rows // 10):
        writer.writerows(workload_group(group))
    data = stream.getvalue().encode("utf-8")
    require(len(data) <= MAX_INPUT_BYTES, "Generated workload exceeds the declared 20 MiB input limit.")
    return data


def nearest_rank(values, percentile=0.95):
    require(type(percentile) in (int, float) and math.isfinite(percentile) and 0 < percentile <= 1,
            "The percentile must be finite and between zero and one.")
    require(bool(values) and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values),
            "Percentiles require nonempty finite nonnegative measurements.")
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def assess(report):
    """Apply declared gates without turning a partial pilot into a full pass."""
    config, samples = report["configuration"], report["samples"]
    measured = [sample for sample in samples if sample["kind"] == "measured"]
    warmups = [sample for sample in samples if sample["kind"] == "warmup"]
    complete = len(measured) == config["runs"] and len(warmups) == config["warmups"]
    full = config["rows"] == 100_000 and config["runs"] == 10 and config["warmups"] == 1
    successful = bool(samples) and all(sample["status"] == "passed" for sample in samples)
    metrics = {}
    for name, part, field in (("execution_p95_seconds", "run", "execution_seconds"),
                              ("export_p95_seconds", "run", "export_seconds"),
                              ("reopen_p95_seconds", "reopen", "reopen_seconds")):
        values = [(sample.get(part) or {}).get(field) for sample in measured]
        metrics[name] = nearest_rank(values) if values and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values) else None
    peaks = [(sample.get(part) or {}).get("peak_rss_bytes") for sample in samples for part in ("run", "reopen")]
    metrics["peak_rss_bytes"] = max(peaks) if peaks and all(type(value) is int and value > 0 for value in peaks) else None
    gates = {
        "sample_protocol_complete": complete,
        "all_samples_correct": successful,
        "input_within_20_mib": 0 < report["workload"]["input_bytes"] <= MAX_INPUT_BYTES,
        "execution_p95_at_most_30_seconds": metrics["execution_p95_seconds"] is not None and metrics["execution_p95_seconds"] <= MAX_SECONDS,
        "export_p95_at_most_30_seconds": metrics["export_p95_seconds"] is not None and metrics["export_p95_seconds"] <= MAX_SECONDS,
        "reopen_p95_at_most_30_seconds": metrics["reopen_p95_seconds"] is not None and metrics["reopen_p95_seconds"] <= MAX_SECONDS,
        "whole_process_peak_at_most_1_gib": metrics["peak_rss_bytes"] is not None and metrics["peak_rss_bytes"] <= MAX_RSS_BYTES,
    }
    passed = full and all(gates.values())
    if not complete:
        status = "incomplete"
    elif full:
        status = "passed" if passed else "failed"
    else:
        status = "pilot_complete" if all(gates.values()) else "pilot_failed"
    return {"status": status, "full_protocol": full, "f14_gate_passed": passed,
            "archive_profile": "with_history" if config["include_history"] else "without_history",
            "measured_samples": len(measured), "warmup_samples": len(warmups),
            "metrics": metrics, "gates": gates}


def peak_rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                            "PagefileUsage", "PeakPagefileUsage")]]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def hardware():
    total_ram = None
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical", "total_pagefile", "available_pagefile", "total_virtual", "available_virtual", "extended_virtual")]]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total_ram = int(status.total_physical)
    else:
        try:
            total_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            pass
    return {"platform": platform.platform(), "architecture": platform.machine(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_cpus": os.cpu_count(), "physical_ram_bytes": total_ram,
            "python": sys.version, "python_executable": sys.executable}


def versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def implementation_hashes():
    paths = ("backend/fulfillment.py", "backend/fulfillment_state.py", "backend/project_io.py",
             "backend/fulfillment_wire.py", "backend/resources.py", "scripts/benchmark_fulfillment.py",
             "requirements.txt")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def declared_environment():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==", 1)
            pins[name] = version
    actual = versions()
    return {"declared_pins": pins, "actual_packages": actual,
            "matches_declared_pins": all(actual.get(name) == version for name, version in pins.items()),
            "scope": "Performance measurements use actual installed packages; matching declared dependencies is a separate release gate."}


def check_project(project, rows):
    attempt = project["committed"]["attempt"]
    require(attempt["status"] == "completed" and attempt["report"] == expected_report(rows),
            "Workload report differs from independently specified counts.")
    accepted_positions = {0, 2, 3, 7, 9}
    canonical = {
        0: (0, "01", "2026-09-01", "SKU-001", 2, "fulfilled"),
        2: (0, "02", "2026-09-01", "SKU-002", 1, "pending"),
        3: (1, "01", "2026-09-02", "SKU-003", 3, "cancelled"),
        7: (2, "01", "2026-09-05", "SKU-006", 4, "fulfilled"),
        9: (3, "01", "2026-09-07", "SKU-008", 2, "pending"),
    }
    for record in attempt["accepted"]:
        group, position = divmod(record["source_record"] - 1, 10)
        require(position in accepted_positions, "Unexpected accepted source record.")
        offset, *values = canonical[position]
        require(record["values"] == [f"{group * 4 + offset:010d}", *values],
                "Accepted canonical values differ from workload oracle.")
    reasons = {1: {"exact_duplicate"}, 4: {"nonpositive_quantity"}, 5: {"invalid_order_date"},
               6: {"missing_order_id", "missing_sku", "nonpositive_quantity"}, 8: {"missing_line_id"}}
    for record in attempt["rejected"]:
        group, position = divmod(record["source_record"] - 1, 10)
        require(position in reasons and set(record["reasons"]) == reasons[position],
                "Quarantine reasons differ from workload oracle.")
        if position == 1:
            require(record["duplicate_of_source_record"] == group * 10 + 1,
                    "Exact duplicate does not reference its first canonical record.")
    return attempt


def worker(arguments):
    started = time.perf_counter()
    result = {"status": "failed", "kind": arguments.worker, "stage": "imports"}
    try:
        from backend.fulfillment import ingest_csv, plan_digest
        from backend.fulfillment_state import export_project, new_project, run_project, set_source, validate_project
        from backend.project_io import dumps_project, loads_project
        result["import_seconds"] = time.perf_counter() - started
        if arguments.worker == "run":
            result["stage"] = "read_input"
            begin = time.perf_counter()
            data = arguments.input.read_bytes()
            result["input_read_seconds"] = time.perf_counter() - begin
            result["input_bytes"] = len(data)
            result["input_sha256"] = hashlib.sha256(data).hexdigest()
            result["stage"] = "ingestion"
            begin = time.perf_counter()
            source = ingest_csv(data)
            result["ingestion_seconds"] = time.perf_counter() - begin
            result["stage"] = "draft_setup"
            begin = time.perf_counter()
            project = set_source(new_project(), source)
            result["setup_seconds"] = time.perf_counter() - begin
            result["stage"] = "execution"
            begin = time.perf_counter()
            project = run_project(project)
            result["execution_seconds"] = time.perf_counter() - begin
            result["stage"] = "oracle"
            begin = time.perf_counter()
            attempt = check_project(project, arguments.rows)
            result["oracle_seconds"] = time.perf_counter() - begin
            result.update(source_id=source["source_id"], plan_digest=plan_digest(project["draft_plan"]),
                          run_id=attempt["run_id"], actual_report=attempt["report"],
                          runtime_history_snapshots=len(project["history"]))
            result["stage"] = "export"
            begin = time.perf_counter()
            payload = {"mode": "fulfillment-only", "project_name": "Synthetic F14 workload",
                       "fulfillment": export_project(project, include_history=arguments.include_history)}
            archive = dumps_project(payload)
            result["export_seconds"] = time.perf_counter() - begin
            result["exported_history_snapshots"] = len(payload["fulfillment"]["history"])
            require(result["exported_history_snapshots"] == (len(project["history"]) if arguments.include_history else 0),
                    "The exported history does not match the requested archive profile.")
            result["stage"] = "archive_write"
            begin = time.perf_counter()
            arguments.archive.write_bytes(archive)
            result["archive_write_seconds"] = time.perf_counter() - begin
            result.update(archive_bytes=len(archive), archive_sha256=hashlib.sha256(archive).hexdigest())
        else:
            result["stage"] = "reopen"
            # Cold reopen timing includes backend imports, archive reading, typed
            # decoding, and full source/plan/evidence validation in this process.
            payload = loads_project(arguments.archive.read_bytes())
            project = validate_project(payload["fulfillment"])
            result["reopen_seconds"] = time.perf_counter() - started
            result["stage"] = "oracle"
            attempt = check_project(project, arguments.rows)
            result.update(source_id=project["source"]["source_id"],
                          plan_digest=plan_digest(project["draft_plan"]), run_id=attempt["run_id"],
                          actual_report=attempt["report"], reopened_history_snapshots=len(project["history"]))
        require(not any(name == "streamlit" or name.startswith("streamlit.") for name in sys.modules),
                "Benchmark backend imported Streamlit.")
        result.update(status="passed", stage="complete")
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)[:1000])
    result["process_operation_seconds"] = time.perf_counter() - started
    result["packages"] = versions()
    result["implementation_sha256"] = implementation_hashes()
    try:
        result["peak_rss_bytes"] = peak_rss_bytes()
    except Exception as exc:
        result.update(status="failed", peak_rss_bytes=None, measurement_error=type(exc).__name__)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


def invoke_worker(arguments, kind, input_path, archive_path):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", kind,
               "--rows", str(arguments.rows), "--input", str(input_path), "--archive", str(archive_path)]
    if arguments.include_history:
        command.append("--include-history")
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error_type": "TimeoutExpired", "timeout_seconds": 180, "kind": kind}
    try:
        result = json.loads(process.stdout)
        require(type(result) is dict, "Worker did not return an object.")
    except (ValueError, TypeError):
        return {"status": "failed", "kind": kind, "error_type": "WorkerProtocolError",
                "returncode": process.returncode, "stdout_tail": process.stdout[-2000:], "stderr_tail": process.stderr[-2000:]}
    result["returncode"] = process.returncode
    if process.returncode != 0:
        result["status"] = "failed"
    return result


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for incremental results, input, and actual archives")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--include-history", action="store_true", help="Measure the separate history-inclusive archive profile")
    parser.add_argument("--worker", choices=("run", "reopen"), help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--archive", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    try:
        expected = expected_report(arguments.rows)
        require(arguments.runs > 0 and arguments.warmups >= 0, "Runs must be positive and warm-ups nonnegative.")
    except ValueError as exc:
        parser.error(str(exc))
    if arguments.worker:
        require(arguments.input is not None and arguments.archive is not None, "Worker paths are required.")
        return worker(arguments)
    if arguments.output_dir is None:
        parser.error("--output-dir is required")
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = generate_workload(arguments.rows)
    input_path = output / "workload.csv"
    input_path.write_bytes(data)
    report = {
        "protocol_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic single-user backend workload; no UI, customer, or business-value claim",
        "status": "running", "configuration": {"rows": arguments.rows, "runs": arguments.runs,
            "warmups": arguments.warmups, "include_history": arguments.include_history},
        "workload": {"input_bytes": len(data), "input_sha256": hashlib.sha256(data).hexdigest(),
            "columns": list(COLUMNS), "encoding": "utf-8-sig", "delimiter": ",", "expected_report": expected},
        "measurement": {"execution": "run_project complete candidate/commit transition; ingestion and setup recorded separately",
            "export": "export_project plus typed archive serialization/compression; archive write recorded separately",
            "reopen": "fresh process backend imports plus archive read, loads_project and validate_project",
            "rss": "OS whole-process high-water mark including imports and oracle checks, maximum across execution and reopen processes",
            "percentile": "nearest-rank p95 of measured samples; warm-up excluded from timing percentile",
            "timeout_seconds_per_process": 180},
        "reference_host": hardware(), "packages": versions(), "declared_environment": declared_environment(),
        "implementation_sha256": implementation_hashes(), "samples": [],
    }
    path = output / "results.json"
    write_report(path, report)
    try:
        for index in range(arguments.warmups + arguments.runs):
            kind = "warmup" if index < arguments.warmups else "measured"
            archive_path = output / f"sample-{index:02d}.dataworks"
            sample = {"index": index, "kind": kind, "status": "running", "archive": archive_path.name}
            report["samples"].append(sample)
            write_report(path, report)
            sample["run"] = invoke_worker(arguments, "run", input_path, archive_path)
            if sample["run"]["status"] == "passed":
                sample["reopen"] = invoke_worker(arguments, "reopen", input_path, archive_path)
                same_identity = all(sample["run"].get(field) == sample["reopen"].get(field)
                                    for field in ("run_id", "source_id", "plan_digest"))
                unchanged_input = sample["run"].get("input_sha256") == report["workload"]["input_sha256"]
                unchanged_code = all(sample[part].get("implementation_sha256") == report["implementation_sha256"] for part in ("run", "reopen"))
                sample["status"] = "passed" if sample["reopen"]["status"] == "passed" and same_identity and unchanged_input and unchanged_code else "failed"
                if not same_identity:
                    sample["identity_mismatch"] = True
                if not unchanged_input:
                    sample["input_changed"] = True
                if not unchanged_code:
                    sample["implementation_changed"] = True
            else:
                sample.update(status="failed", reopen=None)
            report["assessment"] = assess(report)
            write_report(path, report)
            print(json.dumps({"index": index, "kind": kind, "status": sample["status"],
                              "execution_seconds": sample["run"].get("execution_seconds"),
                              "export_seconds": sample["run"].get("export_seconds"),
                              "reopen_seconds": (sample["reopen"] or {}).get("reopen_seconds"),
                              "peak_rss_bytes": sample["run"].get("peak_rss_bytes")}), flush=True)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["assessment"] = assess(report)
        write_report(path, report)
        return 130
    report["assessment"] = assess(report)
    report["status"] = report["assessment"]["status"]
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(path, report)
    print(json.dumps(report["assessment"], indent=2))
    return 0 if report["status"] in ("passed", "pilot_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
