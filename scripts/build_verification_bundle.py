"""Build/check a local, content-addressed verification bundle; never publish it."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "BUNDLE-MANIFEST.json"
ROOT_FILES = ("app.py", "README.md", "DEPLOYMENT.md", "REPRODUCING.md", "RELEASE_GATES.md",
              "requirements.txt", "requirements-verified-windows-py312.txt",
              ".streamlit/config.toml", ".github/workflows/ci.yml", ".gitignore")
DIRECTORIES = ("backend", "frontend", "scripts", "tests", "docs/acceptance", "environment")
EXTENSIONS = {".py", ".md", ".txt", ".json", ".png", ".toml", ".yml", ".yaml"}
MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES = 2000, 10 * 1024**2, 100 * 1024**2
ADDITIONS = {
    *ROOT_FILES, "scripts/build_verification_bundle.py", "scripts/verify_release.py",
    "tests/test_verification_bundle.py", "tests/test_release_verification.py",
    "tests/fixtures/ingestion_v1/README.md", "tests/fixtures/ingestion_v1/original-synthetic.csv",
    "docs/acceptance/phase-6-protocol.md", "environment/verified-windows-py312.json",
    "environment/declared-clean-windows-py312.json",
}


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate bundle manifest key.")
            result[key] = value
        return result

    def finite(value):
        raise ValueError("Nonfinite bundle manifest value.")

    return json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=finite)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name
            or name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/"))
            or PurePosixPath(name).is_absolute()):
        raise ValueError("Unsafe bundle member path.")
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(part.endswith((".", " ")) or part.split(".")[0].upper() in reserved
           or any(character in part for character in '<>"|?*') for part in name.split("/")):
        raise ValueError("Bundle member has an ambiguous or reserved Windows path.")
    return name


def approved_inventory(root):
    if (root / MANIFEST).is_file():
        manifest = strict_json((root / MANIFEST).read_bytes())
        return {safe_name(item["path"]) for item in manifest["files"]}
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    if Path(top).resolve() != root.resolve():
        raise ValueError("Build from the repository root or a checked extracted bundle.")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\x00")
    return {name for name in tracked if name} | ADDITIONS


def linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def collect_files(root):
    """Tracked allowlist plus named additions; never sweep arbitrary untracked data."""
    root = root.resolve()
    files = {}
    approved = approved_inventory(root)

    def include(path):
        name = safe_name(path.relative_to(root).as_posix())
        if linked(path) or not path.resolve().is_relative_to(root):
            raise ValueError("Bundle sources cannot be symlinks or leave the source root.")
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("A bundle member exceeds the size limit.")
        files[name] = data

    for name in ROOT_FILES:
        include(root / name)
    for directory in DIRECTORIES:
        folder = root / directory
        if linked(folder):
            raise ValueError("Bundle source directories cannot be symlinks.")
        if not folder.is_dir():
            raise ValueError("A required bundle source directory is missing.")
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(root)
            if ("__pycache__" in relative.parts or any(part.startswith(".") for part in relative.parts)
                    or relative.as_posix().startswith("docs/acceptance/evidence/phase-6/")
                    or relative.as_posix() == "docs/acceptance/phase-6-results.md"):
                continue
            if linked(path):
                raise ValueError("Bundle sources cannot contain symlinks.")
            if relative.as_posix() in approved and path.is_file() and (path.suffix in EXTENSIONS or
                                   path.suffix == ".csv" and relative.as_posix().startswith("tests/fixtures/")):
                include(path)
    if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_TOTAL_BYTES:
        raise ValueError("Bundle sources exceed the file count or total size limit.")
    return dict(sorted(files.items()))


def origin_revision(root):
    # Do not accidentally read the enclosing original checkout when rebuilding
    # from a separately extracted tree inside its evidence directory.
    existing = root / MANIFEST
    if existing.is_file():
        return strict_json(existing.read_bytes())["origin"]["git_head"]
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        if Path(top).resolve() == root.resolve():
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


def build_bundle(root=ROOT):
    root = Path(root).resolve()
    files = collect_files(root)
    entries = [{"path": name, "bytes": len(data), "sha256": sha256(data)} for name, data in files.items()]
    snapshot = sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    manifest = {"format": "dataworks-verification-bundle", "version": 1,
                "origin": {"git_head": origin_revision(root), "source_snapshot_sha256": snapshot,
                           "state": "exact content snapshot; may include uncommitted changes"},
                "release_ready": False,
                "scope": "local verification only; installation, licensing, operator and target-environment gates require separately attributed evidence",
                "exclusions": ["secrets", "Git metadata", "Python environments", "generated project/CSV runs", "surat_uncleaned.csv", "phase-6 result artifacts"],
                "files": entries}
    files[MANIFEST] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue(), manifest


def checked_members(data):
    if len(data) > 50 * 1024**2:
        raise ValueError("Bundle archive exceeds the compressed size limit.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES + 1 or sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
            raise ValueError("Bundle archive exceeds its resource limits.")
        names, members = set(), {}
        for info in infos:
            name = safe_name(info.filename)
            if (name.casefold() in names or info.file_size > MAX_FILE_BYTES or info.is_dir()
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError("Duplicate, unsupported, or oversized bundle member.")
            names.add(name.casefold())
            members[name] = archive.read(info)
    manifest = strict_json(members.pop(MANIFEST))
    if (not isinstance(manifest, dict) or manifest.get("format") != "dataworks-verification-bundle" or type(manifest.get("version")) is not int
            or manifest["version"] != 1 or manifest.get("release_ready") is not False):
        raise ValueError("Unsupported bundle manifest.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != len(members):
        raise ValueError("Bundle manifest member inventory differs.")
    expected = set()
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("Bundle manifest entry has unsupported fields.")
        name = safe_name(item["path"])
        if name in expected or name not in members or type(item["bytes"]) is not int:
            raise ValueError("Bundle manifest contains duplicate or missing members.")
        expected.add(name)
        if len(members[name]) != item["bytes"] or sha256(members[name]) != item["sha256"]:
            raise ValueError("Bundle member content does not match its manifest.")
    if expected != set(members):
        raise ValueError("Bundle contains an unlisted member.")
    snapshot = sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    origin = manifest.get("origin")
    if not isinstance(origin, dict) or snapshot != origin.get("source_snapshot_sha256"):
        raise ValueError("Bundle source snapshot fingerprint differs.")
    members[MANIFEST] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return members, manifest


def extract_bundle(data, destination):
    members, manifest = checked_members(data)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name, contents in sorted(members.items()):
        target = destination / name
        if not target.resolve().is_relative_to(destination):
            raise ValueError("Bundle extraction left its intended destination.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(contents)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New local .zip path")
    parser.add_argument("--extract-to", type=Path, help="Optional new directory for checked extraction")
    args = parser.parse_args()
    output = args.output.resolve()
    data, manifest = build_bundle()
    checked_members(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or checksum.exists():
        parser.error("Output/checksum already exists; choose a new path.")
    with output.open("xb") as stream:
        stream.write(data)
    checksum.write_text(f"{sha256(data)}  {output.name}\n", encoding="ascii")
    if args.extract_to:
        extract_bundle(data, args.extract_to)
    print(json.dumps({"archive": str(output), "sha256": sha256(data), "bytes": len(data),
                      "files": len(manifest["files"]), "source_snapshot_sha256": manifest["origin"]["source_snapshot_sha256"],
                      "release_ready": False}, indent=2))


if __name__ == "__main__":
    main()
