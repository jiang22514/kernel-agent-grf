"""Verify and replay the current paper's archived evidence without network calls."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_files(root, records):
    for name, record in records.items():
        path = (root / name).resolve()
        path.relative_to(root.resolve())
        expected = record if isinstance(record, str) else record["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("File hash mismatch: " + name)
        if isinstance(record, dict) and "bytes" in record:
            if path.stat().st_size != record["bytes"]:
                raise ValueError("File size mismatch: " + name)
    return len(records)


def main():
    output = ROOT / "recomputed/current_reproduction"
    output.mkdir(parents=True, exist_ok=True)
    public = ROOT / "public_case_2026_10"
    counts = {}
    counts["public_case"] = check_files(public, read(public / "PUBLICATION_MANIFEST.json")["files"])
    data = public / "data_public_staging"
    counts["source_data_package"] = check_files(data, read(data / "data_package_manifest.json")["files"])
    matlab = public / "matlab_public_package"
    counts["matlab_package"] = check_files(matlab, {
        r["path"]: r for r in read(matlab / "matlab_manifest.json")["files"]})

    # Historical verifiers write reports beside their input files. Run copies
    # in a temporary directory so archived records and hashes remain unchanged.
    tasks = [
        ("public_search", public / "public_search", "verify_public_search.py"),
        ("synthetic_release", ROOT / "research_release_2026_10", "tools/verify_release.py"),
        ("feedback_ablation", ROOT / "feedback_followup_2026_10_10/feedback_ablation/records",
         "verify_public_ablation.py"),
        ("random_continuation", ROOT / "feedback_followup_2026_10_10/continuation_public",
         "reproduce_continuations.py"),
    ]
    results = {}
    with tempfile.TemporaryDirectory(prefix="kernel-agent-replay-") as temporary:
        for name, source, script in tasks:
            staged = Path(temporary) / name
            shutil.copytree(source, staged, ignore=shutil.ignore_patterns("__pycache__", "recomputed"))
            completed = subprocess.run([sys.executable, "-B", str(staged / script)],
                                       cwd=staged, text=True, encoding="utf-8", capture_output=True)
            if completed.returncode:
                raise RuntimeError(name + " failed:\n" + completed.stdout + completed.stderr)
            result = json.loads(completed.stdout)
            if result.get("status", "PASS") != "PASS":
                raise ValueError(name + " returned a non-PASS status")
            results[name] = result
            (output / (name + ".json")).write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(name + ": PASS")

    summary = {"status": "PASS", "hash_checks": counts,
               "offline_checks": list(results), "network_calls": 0, "gp_fits": 0,
               "matlab_runs": 0, "field_simulations": 0}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
