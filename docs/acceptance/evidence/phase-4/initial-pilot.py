"""Initial pre-optimization diagnostic; all-valid rows isolate state/codec costs."""
import csv
import ctypes
from ctypes import wintypes
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from backend.fulfillment import ingest_csv
from backend.fulfillment_state import new_project, set_source, run_project, export_project, validate_project
from backend.project_io import dumps_project, loads_project


class Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]


def peak():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


def main():
    output = Path(__file__).with_name("initial-pilot.json")
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    result = {"rows": rows, "scope": "initial all-valid diagnostic, not the mixed F14 acceptance workload", "stages": []}
    def measure(name, action):
        start = time.perf_counter()
        try:
            value = action()
        except Exception as exc:
            result["stages"].append({"name": name, "seconds": time.perf_counter()-start, "peak_rss_bytes": peak(), "error_type": type(exc).__name__, "error": str(exc)})
            output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
            print(json.dumps(result["stages"][-1]), flush=True)
            raise
        result["stages"].append({"name": name, "seconds": time.perf_counter()-start, "peak_rss_bytes": peak()})
        output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result["stages"][-1]), flush=True)
        return value
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["order_id", "line_id", "order_date", "sku", "quantity", "fulfillment_status"])
    writer.writerows([[f"{i:09d}", "01", "2026-10-01", "SKU-001", "1", "shipped"] for i in range(rows)])
    data = stream.getvalue().encode("utf-8")
    del stream, writer
    result["input_bytes"] = len(data)
    source = measure("ingest", lambda: ingest_csv(data))
    project = measure("draft", lambda: set_source(new_project(), source))
    project = measure("execute", lambda: run_project(project))
    if project["last_attempt"]["status"] != "completed":
        raise RuntimeError(project["last_attempt"]["status"])
    payload = measure("export_without_history", lambda: export_project(project, include_history=False))
    archive = measure("encode", lambda: dumps_project({"mode":"fulfillment-only", "project_name":"Initial diagnostic", "fulfillment":payload}))
    measure("decode_validate", lambda: validate_project(loads_project(archive)["fulfillment"]))


if __name__ == "__main__":
    main()
