"""Package completeness, deterministic content, and checked-extraction boundaries."""

from contextlib import contextmanager
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import build_verification_bundle as bundle


class VerificationBundleTests(unittest.TestCase):
    @contextmanager
    def tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in bundle.ROOT_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original synthetic package test\n", encoding="utf-8")
            for name in bundle.DIRECTORIES:
                (root / name).mkdir(parents=True, exist_ok=True)
            (root / "backend/engine.py").write_text("answer = 42\n", encoding="utf-8")
            (root / "tests/fixtures/original.csv").parent.mkdir(parents=True)
            (root / "tests/fixtures/original.csv").write_text("id,value\n001,2\n", encoding="utf-8")
            approved = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
            with patch.object(bundle, "origin_revision", return_value="synthetic-revision"), patch.object(bundle, "approved_inventory", return_value=approved):
                yield root

    def rewrite(self, data, *, change=None, add=None):
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(output, "w") as target:
            for info in source.infolist():
                value = source.read(info)
                target.writestr(info, change(info.filename, value) if change else value)
            if add:
                target.writestr(*add)
        return output.getvalue()

    def test_build_is_byte_identical_and_preserves_exact_source_bytes(self):
        with self.tree() as root:
            first, manifest = bundle.build_bundle(root)
            (root / "backend/engine.py").touch()
            second, _ = bundle.build_bundle(root)
            self.assertEqual(first, second)
            members, checked = bundle.checked_members(first)
            self.assertEqual(manifest, checked)
            self.assertEqual(members["backend/engine.py"], (root / "backend/engine.py").read_bytes())
            self.assertFalse(manifest["release_ready"])

    def test_allowlist_excludes_secrets_legacy_data_generated_runs_and_environment(self):
        with self.tree() as root:
            for name in (".streamlit/secrets.toml", ".env", "surat_uncleaned.csv", ".git/config",
                         ".venv/private.txt", "tests/fixtures/run.dataworks",
                         "docs/acceptance/evidence/phase-6/private.json", "backend/__pycache__/cached.py",
                         "backend/credentials.json", "environment/private.txt"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("must stay outside bundle", encoding="utf-8")
            data, _ = bundle.build_bundle(root)
            members, _ = bundle.checked_members(data)
            self.assertIn("tests/fixtures/original.csv", members)
            self.assertFalse(any("must stay outside bundle" in value.decode("utf-8", errors="ignore") for value in members.values()))

    def test_changed_source_bytes_are_rejected_before_extraction(self):
        with self.tree() as root:
            data, _ = bundle.build_bundle(root)
            corrupted = self.rewrite(data, change=lambda name, value: b"forged" if name == "backend/engine.py" else value)
            target = root / "new-extraction"
            with self.assertRaises(ValueError):
                bundle.extract_bundle(corrupted, target)
            self.assertFalse(target.exists())

    def test_added_and_duplicate_members_are_rejected(self):
        with self.tree() as root:
            data, _ = bundle.build_bundle(root)
            for addition in (("unlisted.txt", b"extra"), ("APP.py", b"case alias")):
                with self.subTest(addition=addition), self.assertRaises(ValueError):
                    bundle.checked_members(self.rewrite(data, add=addition))

    def test_traversal_absolute_and_windows_alias_paths_are_rejected(self):
        for name in ("../escape", "/absolute", "C:/drive", "a\\b", "a/./b", "a//b", "file.", "CON.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                bundle.safe_name(name)

    def test_archive_symlinks_are_rejected(self):
        with self.tree() as root:
            data, _ = bundle.build_bundle(root)
            link = zipfile.ZipInfo("backend/link.py")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with self.assertRaises(ValueError):
                bundle.checked_members(self.rewrite(data, add=(link, b"../../outside")))

    def test_checked_extraction_uses_new_directory_and_matches_all_entries(self):
        with self.tree() as root:
            data, expected = bundle.build_bundle(root)
            target = root / "separate-copy"
            self.assertEqual(bundle.extract_bundle(data, target), expected)
            for item in expected["files"]:
                self.assertEqual(bundle.sha256((target / item["path"]).read_bytes()), item["sha256"])
            with self.assertRaises(FileExistsError):
                bundle.extract_bundle(data, target)

    def test_false_release_claim_and_resource_overflow_are_rejected(self):
        with self.tree() as root:
            data, _ = bundle.build_bundle(root)
            def misleading(name, value):
                if name == bundle.MANIFEST:
                    manifest = json.loads(value)
                    manifest["release_ready"] = True
                    return json.dumps(manifest).encode()
                return value
            with self.assertRaises(ValueError):
                bundle.checked_members(self.rewrite(data, change=misleading))
            with patch.object(bundle, "MAX_FILES", 1), self.assertRaises(ValueError):
                bundle.checked_members(data)


if __name__ == "__main__":
    unittest.main()
