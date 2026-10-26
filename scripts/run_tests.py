"""Cross-platform unittest runner with an explicit, disposable AppTest temp root."""

from pathlib import Path
import os
import sys
import tempfile
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.chdir(root)
    scratch = root / ".test-tmp"
    scratch.mkdir(exist_ok=True)
    previous = tempfile.tempdir
    try:
        with tempfile.TemporaryDirectory(prefix="run-", dir=scratch) as directory:
            tempfile.tempdir = directory
            suite = unittest.defaultTestLoader.discover(str(root / "tests"))
            result = unittest.TextTestRunner(verbosity=1).run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        tempfile.tempdir = previous
        try:
            scratch.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
