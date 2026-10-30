"""Release verification gates must not turn incomplete evidence into approval."""

import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_release as verifier


class ReleaseVerificationTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name) / "source"
        self.root.mkdir()
        for directory in ("backend", "frontend", "scripts", "tests/fixtures/fulfillment_v1"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / "app.py").write_text("pass\n", encoding="utf-8")
        (self.root / "requirements.txt").write_text("sample==1.0\n", encoding="utf-8")
        (self.root / "scripts/run_tests.py").write_text("pass\n", encoding="utf-8")
        (self.root / "scripts/verify_fulfillment.py").write_text("pass\n", encoding="utf-8")
        fixture = self.root / "tests/fixtures/fulfillment_v1/month_a.csv"
        fixture.write_bytes(b"id,value\n001,1\n")
        self.fixture = fixture
        manifest = {"files": [{"path": fixture.name, "bytes": fixture.stat().st_size,
                               "sha256": verifier.file_digest(fixture)}]}
        (fixture.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.output = Path(self.scratch.name) / "results"

    def environment(self, *, matches=True, installed=True, supported=True):
        return {"python_3_12": supported, "all_installed": installed,
                "all_direct_versions_match": matches,
                "packages": [{"name": "sample", "declared": "1.0", "installed": "1.0" if matches else "0.9"}]}

    def successful_check(self, root, output, name, command, *, timeout):
        (output / f"{name}.txt").write_bytes(b"Ran 1 test in 0.001s\n\nOK\n")
        if name == "fulfillment":
            (output / "workflow").mkdir()
            (output / "workflow/results.json").write_text('{"status":"passed"}', encoding="utf-8")
        return {"name": name, "command": command, "status": "passed", "exit_code": 0, "log": f"{name}.txt"}

    def test_default_mismatch_blocks_before_any_subprocess_and_records_no_release(self):
        with patch.object(verifier, "environment", return_value=self.environment(matches=False)), \
                patch.object(verifier, "run_check") as run:
            result, code = verifier.verify(self.root, self.output)
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["release_ready"])
        self.assertEqual(result["release_gates"]["declared_direct_versions"], "mismatch")
        self.assertEqual(result["release_gates"]["clean_installation_of_declared_pins"], "not_verified")
        run.assert_not_called()
        self.assertEqual(json.loads((self.output / "results.json").read_text())["status"], "blocked")

    def test_explicit_override_allows_engineering_without_claiming_install_or_release(self):
        with patch.object(verifier, "environment", return_value=self.environment(matches=False)), \
                patch.object(verifier, "run_check", side_effect=self.successful_check):
            result, code = verifier.verify(self.root, self.output, allow_environment_mismatch=True)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["release_ready"])
        self.assertEqual(result["source_changes_during_checks"], [])
        self.assertEqual([item["name"] for item in result["checks"]], ["regression", "fulfillment"])
        self.assertEqual(result["release_gates"]["independent_human_reproduction"], "not_observed")
        self.assertEqual(result["release_gates"]["declared_direct_versions"], "mismatch")

    def test_missing_packages_and_unsupported_python_cannot_be_overridden(self):
        for number, actual in enumerate((self.environment(installed=False), self.environment(supported=False))):
            with self.subTest(case=number), patch.object(verifier, "environment", return_value=actual), \
                    patch.object(verifier, "run_check") as run:
                result, code = verifier.verify(self.root, self.output.with_name(f"blocked-{number}"),
                                               allow_environment_mismatch=True)
                self.assertEqual((result["status"], code), ("blocked", 2))
                run.assert_not_called()

    def test_changed_fixture_or_invalid_syntax_fails_before_checks(self):
        self.fixture.write_bytes(b"changed input")
        with patch.object(verifier, "run_check") as run:
            result, code = verifier.verify(self.root, self.output)
        self.assertEqual((result["status"], code), ("failed", 1))
        self.assertEqual(result["fixtures"]["status"], "failed")
        run.assert_not_called()
        self.fixture.write_bytes(b"id,value\n001,1\n")
        (self.root / "app.py").write_text("def broken(\n", encoding="utf-8")
        with patch.object(verifier, "environment", return_value=self.environment()), \
                patch.object(verifier, "run_check") as run:
            result, code = verifier.verify(self.root, self.output.with_name("syntax"))
        self.assertEqual((result["error_type"], code), ("SyntaxError", 1))
        run.assert_not_called()

    def test_skipped_tests_or_missing_test_discovery_are_not_a_passing_suite(self):
        for number, transcript in enumerate((b"Ran 1 test\nOK (skipped=1)\n", b"Ran 0 tests\nOK\n", b"OK\n")):
            def check(root, output, name, command, *, timeout):
                value = self.successful_check(root, output, name, command, timeout=timeout)
                if name == "regression":
                    (output / value["log"]).write_bytes(transcript)
                return value
            with self.subTest(transcript=transcript), patch.object(verifier, "environment", return_value=self.environment()), \
                    patch.object(verifier, "run_check", side_effect=check):
                result, code = verifier.verify(self.root, self.output.with_name(f"skip-{number}"))
            self.assertEqual((result["status"], code), ("failed", 1))
            self.assertEqual(result["checks"][0]["status"], "failed")

    def test_subprocess_failure_and_changed_source_cannot_be_hidden(self):
        def check(root, output, name, command, *, timeout):
            value = self.successful_check(root, output, name, command, timeout=timeout)
            if name == "regression":
                (root / "backend/added.py").write_text("pass\n", encoding="utf-8")
                value.update(status="failed", exit_code=1)
            return value
        with patch.object(verifier, "environment", return_value=self.environment()), \
                patch.object(verifier, "run_check", side_effect=check):
            result, code = verifier.verify(self.root, self.output)
        self.assertEqual((result["status"], code), ("failed", 1))
        self.assertEqual(result["source_changes_during_checks"], ["backend/added.py"])
        self.assertEqual(result["checks"][0]["exit_code"], 1)

    def test_real_subprocess_log_retains_failure_and_clears_original_pythonpath(self):
        self.output.mkdir()
        command = [sys.executable, "-c", "import os,sys; print(os.environ.get('PYTHONPATH','ABSENT')); print('synthetic stderr',file=sys.stderr); sys.exit(7)"]
        with patch.dict(os.environ, {"PYTHONPATH": "original-checkout-must-not-leak"}):
            result = verifier.run_check(self.root, self.output, "failure", command, timeout=10)
        self.assertEqual((result["status"], result["exit_code"]), ("failed", 7))
        log = (self.output / "failure.txt").read_text()
        self.assertIn("ABSENT", log)
        self.assertIn("synthetic stderr", log)
        self.assertNotIn("original-checkout-must-not-leak", log)

    def bundle(self):
        paths = sorted(path for path in self.root.rglob("*") if path.is_file())
        manifest = {"format": "dataworks-verification-bundle", "version": 1, "release_ready": False,
                    "origin": {"git_head": "test-only"}, "files": [
                        {"path": path.relative_to(self.root).as_posix(), "bytes": path.stat().st_size,
                         "sha256": verifier.file_digest(path)} for path in paths]}
        manifest["origin"]["source_snapshot_sha256"] = hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        (self.root / "BUNDLE-MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_bundle_checks_actual_files_and_rejects_forged_entries_and_source_omission(self):
        manifest = self.bundle()
        self.assertEqual(verifier.check_bundle(self.root)["status"], "passed")
        variants = [
            {**manifest, "version": True}, {**manifest, "release_ready": True},
            {**manifest, "files": [*manifest["files"], manifest["files"][0]]},
            {**manifest, "files": [*manifest["files"], {**manifest["files"][0], "path": manifest["files"][0]["path"].upper()}]},
            {**manifest, "files": [{"path": "../outside", "bytes": 0, "sha256": "0" * 64}]},
            {**manifest, "files": [item for item in manifest["files"] if item["path"] != "app.py"]},
            {**manifest, "files": [{**item, "sha256": "0" * 64} for item in manifest["files"]]},
        ]
        for value in variants:
            value = {**value, "origin": {**value["origin"], "source_snapshot_sha256": hashlib.sha256(
                json.dumps(value["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()}}
            (self.root / "BUNDLE-MANIFEST.json").write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(ValueError):
                verifier.check_bundle(self.root)

    def test_manifest_linked_files_and_duplicate_json_keys_are_rejected(self):
        with patch.object(Path, "is_symlink", return_value=True), self.assertRaises(ValueError):
            verifier.safe_file(self.root, "app.py")
        path = self.root / "BUNDLE-MANIFEST.json"
        path.write_text('{"version":1,"version":2}', encoding="utf-8")
        with self.assertRaises(ValueError):
            verifier.check_bundle(self.root)
        path.write_text('{"version":NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            verifier.check_bundle(self.root)

    def test_claimed_source_fingerprint_and_windows_aliases_are_checked(self):
        manifest = self.bundle()
        manifest["origin"]["source_snapshot_sha256"] = "0" * 64
        (self.root / "BUNDLE-MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
            verifier.check_bundle(self.root)
        for path in ("app.py.", "app.py ", "NUL.txt", "folder/COM1.py", "a?b.py", "a\nb.py", "a\x00b.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                verifier.portable_name(path)
        self.assertEqual(str(verifier.portable_name("folder/normal.py")), "folder/normal.py")

    def test_fixture_and_nonpython_bundle_changes_during_checks_fail(self):
        original = b"id,value\n001,1\n"
        for number, mutate in enumerate((lambda: self.fixture.write_bytes(b"changed input"),
                                         lambda: (self.root / "README.md").write_text("changed evidence", encoding="utf-8"))):
            self.fixture.write_bytes(original)
            (self.root / "README.md").write_text("original evidence", encoding="utf-8")
            (self.root / "BUNDLE-MANIFEST.json").unlink(missing_ok=True)
            self.bundle()

            def check(root, output, name, command, *, timeout):
                value = self.successful_check(root, output, name, command, timeout=timeout)
                if name == "fulfillment":
                    mutate()
                return value

            with self.subTest(case=number), patch.object(verifier, "environment", return_value=self.environment()), \
                    patch.object(verifier, "run_check", side_effect=check):
                result, code = verifier.verify(self.root, self.output.with_name(f"nonpython-change-{number}"))
            self.assertEqual((result["status"], code), ("failed", 1))
            self.assertFalse(result["bundle_unchanged_during_checks"])
            self.assertEqual(result["source_changes_during_checks"], [])

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        (self.output / "results.json").write_text("retained evidence", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            verifier.verify(self.root, self.output)
        self.assertEqual((self.output / "results.json").read_text(), "retained evidence")


if __name__ == "__main__":
    unittest.main()
