"""Verify rebuilt public-case arrays against the frozen public reference NPZs.

No model is fitted and no network is accessed. MAT headers may contain build
timestamps, so reproducibility is assessed on every named numerical array.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import scipy
from scipy.io import loadmat

CASES = ("main_12", "density_06", "density_18", "fine_12")


def verify(data_dir, reference_dir, log_atol=0.0):
    cases = {}
    all_ok = True
    all_exact = True
    for case in CASES:
        reference_path = reference_dir / f"{case}.npz"
        reference = np.load(reference_path, allow_pickle=False)
        rebuilt = np.load(data_dir / f"{case}.npz", allow_pickle=False)
        mat = {key: value for key, value in loadmat(data_dir / f"{case}.mat").items() if not key.startswith("__")}
        if set(reference.files) != set(rebuilt.files) or set(reference.files) != set(mat):
            raise AssertionError(f"Array names differ for {case}")
        checks = {}
        for key in reference.files:
            expected = reference[key]
            for format_name, actual in (("npz", rebuilt[key]), ("mat", mat[key])):
                exact = expected.shape == actual.shape and np.array_equal(expected, actual, equal_nan=True)
                tolerance = log_atol if key in ("y_log", "y_test_log") else 0.0
                passed = exact or (expected.shape == actual.shape and np.allclose(expected, actual, rtol=0.0, atol=tolerance, equal_nan=True))
                max_error = float(np.nanmax(np.abs(expected.astype(float) - actual.astype(float)))) if expected.shape == actual.shape else None
                checks[f"{format_name}:{key}"] = {"shape": list(actual.shape), "exact": bool(exact), "passed": bool(passed), "max_abs_error": max_error, "allowed_abs_error": tolerance}
                all_ok = all_ok and bool(passed)
                all_exact = all_exact and bool(exact)
        cases[case] = {"reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(), "n_train": int(reference["y"].size), "n_test": int(reference["y_test"].size), "arrays": checks}
        reference.close()
        rebuilt.close()
    return {"passed": all_ok, "all_arrays_exact": all_exact, "log_absolute_tolerance": log_atol, "response": "ln(qc / 1 MPa)", "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}, "cases": cases, "model_fitting_performed": False, "network_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).parent / "reference")
    parser.add_argument("--report", type=Path, help="Defaults to DATA/rebuild_verification.json")
    parser.add_argument("--log-atol", type=float, default=0.0, help="Optional cross-platform log rounding tolerance; all coordinates, raw qc and source indices remain exact. Default 0.")
    args = parser.parse_args()
    if not np.isfinite(args.log_atol) or not 0 <= args.log_atol <= 1e-12:
        raise ValueError("--log-atol must lie between 0 and 1e-12")
    result = verify(args.data.resolve(), args.reference.resolve(), args.log_atol)
    report = args.report or args.data / "rebuild_verification.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "all_arrays_exact", "log_absolute_tolerance", "versions")}, indent=2))
    if not result["passed"]:
        raise SystemExit("Rebuilt arrays differ; inspect the verification report")


if __name__ == "__main__":
    main()
