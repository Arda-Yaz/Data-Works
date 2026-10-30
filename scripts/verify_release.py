"""Verify a local DataWorks checkout or extracted bundle without installing it.

The default gate requires installed direct dependencies to match requirements.txt.
An explicit --allow-environment-mismatch permits synthetic engineering checks on
other recorded installed versions. Neither mode establishes a clean installation,
independent human reproduction, user acceptance, or production release readiness.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import chain
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MAX_MANIFEST_BYTES = 4 * 1024**2
MAX_MANIFEST_FILES = 20_000
MAX_BUNDLE_BYTES = 512 * 1024**2


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_name(name):
    """Reject path aliases that differ between Windows and POSIX filesystems."""
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Invalid relative evidence path")
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("Invalid relative evidence path")
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
                *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for part in name.split("/"):
        if (part.endswith((".", " ")) or part.split(".")[0].upper() in reserved
                or any(ord(character) < 32 or character in '<>"|?*' for character in part)):
            raise ValueError("Invalid or ambiguous Windows evidence path")
    return relative


def safe_file(root, name):
    """Accept portable package-relative regular files, never linked evidence."""
    relative = portable_name(name)
    path = root.joinpath(*relative.parts)
    for candidate in (path, *path.parents):
        if candidate == root.parent:
            break
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError("Linked evidence is unsupported")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Evidence file is missing or outside the source tree")
    return path


def strict_json(path):
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Evidence manifest exceeds its size limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate manifest key")
            result[key] = value
        return result

    def finite(value):
        raise ValueError("Nonfinite manifest value")

    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique, parse_constant=finite)


def source_hashes(root):
    paths = ["app.py", "requirements.txt"]
    for directory in ("backend", "frontend", "scripts", "tests"):
        folder = root / directory
        for path in chain((folder,), folder.rglob("*")):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError("Linked source paths are unsupported")
            if path.is_file() and path.suffix == ".py":
                paths.append(path.relative_to(root).as_posix())
    return {name: file_digest(safe_file(root, name)) for name in sorted(set(paths))}


def check_bundle(root):
    manifest_path = root / "BUNDLE-MANIFEST.json"
    if not manifest_path.exists():
        if manifest_path.is_symlink() or (hasattr(manifest_path, "is_junction") and manifest_path.is_junction()):
            raise ValueError("Linked bundle manifests are unsupported")
        return {"status": "not_present", "scope": "Live checkout verification; no bundle identity asserted."}
    manifest = strict_json(safe_file(root, "BUNDLE-MANIFEST.json"))
    if (not isinstance(manifest, dict) or manifest.get("format") != "dataworks-verification-bundle"
            or type(manifest.get("version")) is not int or manifest["version"] != 1
            or manifest.get("release_ready") is not False):
        raise ValueError("Unsupported or misleading bundle manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not 0 < len(entries) <= MAX_MANIFEST_FILES:
        raise ValueError("Invalid bundle file count")
    origin = manifest.get("origin")
    if (not isinstance(origin, dict) or not isinstance(origin.get("source_snapshot_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", origin["source_snapshot_sha256"]) is None):
        raise ValueError("Invalid bundle source fingerprint")
    snapshot = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if snapshot != origin["source_snapshot_sha256"]:
        raise ValueError("Bundle source fingerprint does not match its entries")
    names, portable_names, total = set(), set(), 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid bundle file entry")
        name, size, wanted = entry["path"], entry["bytes"], entry["sha256"]
        if not isinstance(name, str) or name.casefold() in portable_names or name.casefold() == "bundle-manifest.json":
            raise ValueError("Duplicate or invalid bundle file identity")
        if type(size) is not int or not 0 <= size <= MAX_BUNDLE_BYTES:
            raise ValueError("Invalid bundle file size")
        if not isinstance(wanted, str) or re.fullmatch(r"[0-9a-f]{64}", wanted) is None:
            raise ValueError("Invalid bundle file digest")
        names.add(name)
        portable_names.add(name.casefold())
        total += size
        if total > MAX_BUNDLE_BYTES:
            raise ValueError("Bundle exceeds its declared evidence budget")
        path = safe_file(root, name)
        if path.stat().st_size != size or file_digest(path) != wanted:
            raise ValueError("Bundle evidence does not match its manifest")
    if not set(source_hashes(root)).issubset(names):
        raise ValueError("Bundle manifest omits executable sources")
    return {"status": "passed", "manifest_sha256": file_digest(manifest_path),
            "files": len(entries), "bytes": total, "origin": origin,
            "scope": "Listed files and executable-source coverage checked; not a signature or trust assertion."}


def check_fixtures(root):
    folder = root / "tests/fixtures/fulfillment_v1"
    manifest_path = safe_file(folder, "manifest.json")
    manifest = strict_json(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, list) or not 0 < len(entries) <= MAX_MANIFEST_FILES:
        raise ValueError("Invalid fixture manifest")
    results, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or not {"path", "bytes", "sha256"}.issubset(entry):
            raise ValueError("Invalid fixture entry")
        name = entry["path"]
        if not isinstance(name, str) or name in names:
            raise ValueError("Duplicate fixture identity")
        names.add(name)
        path = safe_file(folder, name)
        matches = (type(entry["bytes"]) is int and path.stat().st_size == entry["bytes"]
                   and file_digest(path) == entry["sha256"])
        results.append({"path": name, "matches": matches})
    return {"status": "passed" if all(item["matches"] for item in results) else "failed",
            "manifest_sha256": file_digest(manifest_path), "files": results}


def environment(root):
    packages = []
    for line in (root / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            raise ValueError("Verification requires exact direct dependency pins")
        name, required = match.groups()
        try:
            distribution = importlib.metadata.distribution(name)
            installed = distribution.version
            location = str(distribution.locate_file(""))
        except importlib.metadata.PackageNotFoundError:
            installed, location = None, None
        packages.append({"name": name, "declared": required, "installed": installed,
                         "matches": installed == required, "distribution_location": location})
    if not packages:
        raise ValueError("No declared dependencies found")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "python_3_12": sys.version_info[:2] == (3, 12), "packages": packages,
            "all_installed": all(item["installed"] is not None for item in packages),
            "all_direct_versions_match": all(item["matches"] for item in packages)}


def run_check(root, output, name, command, *, timeout):
    started = time.perf_counter()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=timeout, check=False)
        data, returncode, error_type = completed.stdout, completed.returncode, None
    except subprocess.TimeoutExpired as exc:
        data, returncode, error_type = exc.stdout or b"", None, "TimeoutExpired"
    except OSError as exc:
        data, returncode, error_type = b"", None, type(exc).__name__
    (output / f"{name}.txt").write_bytes(data)
    return {"name": name, "command": list(command), "cwd": str(root), "exit_code": returncode,
            "error_type": error_type, "seconds": time.perf_counter() - started,
            "log": f"{name}.txt", "status": "passed" if returncode == 0 else "failed"}


def release_gates(environment_result):
    return {
        "declared_direct_versions": "passed" if environment_result["all_direct_versions_match"] else "mismatch",
        "clean_installation_of_declared_pins": "not_verified",
        "independent_human_reproduction": "not_observed",
        "real_operator_acceptance_and_time_savings": "not_observed",
        "permissioned_consecutive_customer_exports_and_dashboard": "not_verified",
        "linux_target_execution": "not_verified",
        "licensing_and_distribution_permission": "owner_decision_required",
        "applicable_deployment_controls": "target_environment_verification_required",
    }


def verify(root, output, *, allow_environment_mismatch=False):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"format": "dataworks-release-verification", "version": 1,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
              "scope": "Synthetic engineering verification on recorded installed packages; not production approval.",
              "allow_environment_mismatch": allow_environment_mismatch,
              "status": "failed", "release_ready": False, "checks": []}
    before = None
    exit_code = 1
    try:
        result["bundle"] = check_bundle(root)
        before = source_hashes(root)
        result["implementation_sha256"] = before
        result["fixtures"] = check_fixtures(root)
        if result["fixtures"]["status"] != "passed":
            raise ValueError("Frozen fixture mismatch")
        result["environment"] = environment(root)
        actual = result["environment"]
        result["release_gates"] = release_gates(actual)
        blocked = (not actual["python_3_12"] or not actual["all_installed"]
                   or (not actual["all_direct_versions_match"] and not allow_environment_mismatch))
        if blocked:
            result.update(status="blocked", block_reason="Unsupported Python, missing dependency, or unapproved installed-version mismatch.")
            exit_code = 2
        else:
            syntax = []
            for name in before:
                if name.endswith(".py"):
                    ast.parse((root / name).read_text(encoding="utf-8-sig"), filename=name)
                    syntax.append(name)
            result["syntax_checked"] = syntax
            regression = run_check(root, output, "regression", [sys.executable, "scripts/run_tests.py"], timeout=600)
            result["checks"].append(regression)
            transcript = (output / regression["log"]).read_text(encoding="utf-8", errors="replace")
            count = re.search(r"Ran (\d+) tests?", transcript)
            regression["tests_run"] = int(count.group(1)) if count else None
            regression["no_skips_or_expected_failures"] = not re.search(r"skipped=|expected failures=|unexpected successes=", transcript)
            if (regression["tests_run"] is None or regression["tests_run"] == 0
                    or not regression["no_skips_or_expected_failures"]):
                regression["status"] = "failed"
            workflow = run_check(root, output, "fulfillment", [sys.executable, "scripts/verify_fulfillment.py",
                                 "--output-dir", str(output / "workflow")], timeout=120)
            result["checks"].append(workflow)
            if workflow["status"] == "passed":
                document = strict_json(safe_file(output, "workflow/results.json"))
                if not isinstance(document, dict) or document.get("status") != "passed":
                    workflow["status"] = "failed"
            result["status"] = "passed" if all(item["status"] == "passed" for item in result["checks"]) else "failed"
            exit_code = 0 if result["status"] == "passed" else 1
    except Exception as exc:
        # The fixed exception class diagnoses the verification stage without
        # copying source cells or arbitrary subprocess exception messages.
        result.update(status="failed", error_type=type(exc).__name__)
        exit_code = 1
    finally:
        if before is not None:
            try:
                after = source_hashes(root)
                changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
                result["source_changes_during_checks"] = changed
                if changed:
                    result["status"], exit_code = "failed", 1
            except Exception as exc:
                result.update(status="failed", stability_error_type=type(exc).__name__)
                exit_code = 1
        for key, check in (("bundle", check_bundle), ("fixtures", check_fixtures)):
            if key in result:
                try:
                    unchanged = check(root) == result[key]
                    result[f"{key}_unchanged_during_checks"] = unchanged
                    if not unchanged:
                        result["status"], exit_code = "failed", 1
                except Exception as exc:
                    result.update(status="failed", **{f"{key}_unchanged_during_checks": False,
                                                       f"{key}_stability_error_type": type(exc).__name__})
                    exit_code = 1
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result, exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for actual logs, artifacts, and results.json")
    parser.add_argument("--allow-environment-mismatch", action="store_true",
                        help="Explicitly run checks on recorded installed versions; declared-pin/clean-install gates remain open")
    args = parser.parse_args(argv)
    try:
        result, exit_code = verify(ROOT, args.output_dir, allow_environment_mismatch=args.allow_environment_mismatch)
    except (OSError, ValueError) as exc:
        parser.error(f"A new writable output directory is required ({type(exc).__name__}).")
    print(json.dumps({"status": result["status"], "release_ready": False,
                      "results": str(args.output_dir.resolve() / "results.json")}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
