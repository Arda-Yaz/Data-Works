"""Capture Phase 5 regression and fresh-process proof in a new output directory."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [ROOT / "app.py", ROOT / "requirements.txt"]
    for directory in ("backend", "frontend", "scripts", "tests"):
        paths.extend((ROOT / directory).rglob("*.py"))
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "synthetic engineering checks; no real-operator or declared-pin installation claim",
              "python": sys.version, "implementation_sha256": source_hashes(), "checks": []}
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    result["packages"] = [{"package": line.split("==")[0], "declared": line.split("==")[1],
                           "installed": importlib.metadata.version(line.split("==")[0])}
                          for line in requirements if "==" in line]
    result["head_before"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["index_before"] = hashlib.sha256(subprocess.check_output(["git", "ls-files", "--stage"], cwd=ROOT)).hexdigest()

    def run(name, command):
        begin = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / f"{name}.txt").write_bytes(completed.stdout)
        check = {"name": name, "command": command, "exit_code": completed.returncode,
                 "seconds": time.perf_counter() - begin, "log": f"{name}.txt"}
        result["checks"].append(check)
        print(json.dumps(check), flush=True)
        return completed.returncode == 0

    passed = run("regression", [sys.executable, "scripts/run_tests.py"])
    passed = run("fresh-process-workflow", [sys.executable, "scripts/verify_fulfillment.py", "--output-dir", str(output / "workflow")]) and passed
    passed = run("diff-check", ["git", "diff", "--check"]) and passed
    syntax = []
    for name in result["implementation_sha256"]:
        if name.endswith(".py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8-sig"), filename=name)
            syntax.append(name)
    result["syntax_checked"] = syntax
    fixtures = ROOT / "tests/fixtures/fulfillment_v1"
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    result["fixtures"] = [{"path": item["path"], "matches": digest(fixtures / item["path"]) == item["sha256"]}
                          for item in manifest["files"]]
    result["source_unchanged_during_checks"] = source_hashes() == result["implementation_sha256"]
    result["head_after"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["index_unchanged"] = result["index_before"] == hashlib.sha256(subprocess.check_output(["git", "ls-files", "--stage"], cwd=ROOT)).hexdigest()
    passed = (passed and all(item["matches"] for item in result["fixtures"])
              and result["source_unchanged_during_checks"] and result["index_unchanged"]
              and result["head_before"] == result["head_after"])
    result["status"] = "passed" if passed else "failed"
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["status"], flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
