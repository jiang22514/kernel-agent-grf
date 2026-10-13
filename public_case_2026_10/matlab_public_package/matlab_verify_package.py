"""Verify public package bytes with Python stdlib; no MATLAB or computation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "matlab_manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != "delft_matlab_package_v1" or manifest["status"] != "COMPLETE":
        raise ValueError("Package manifest is not COMPLETE")
    count = 0
    for record in manifest["files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid package path: {record['path']}")
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Package file changed: {record['path']}")
        count += 1
    return {"status": "VERIFIED", "files_checked": count,
            "scope": "File bytes only; this does not claim MATLAB numerical validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
