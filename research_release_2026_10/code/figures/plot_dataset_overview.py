"""Plot all observations from six frozen synthetic test datasets; no score inputs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {row["case"]: row["data_sha256"] for row in manifest["cases"]}
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.2047244094, 4.9212598425))  # 183 x 125 mm
    fig.subplots_adjust(left=0.068, right=0.915, bottom=0.127, top=0.83,
                        hspace=0.65, wspace=0.73)
    records = []
    for index, ax in enumerate(axes.flat, start=1):
        case = f"H{index:02d}"
        source = HERE / f"{case}.mat"
        source_hash = sha256(source)
        if source_hash != expected_hashes[case]:
            raise ValueError(f"Frozen data hash mismatch: {case}")
        data = loadmat(source, variable_names=["x_raw", "y"])
        coordinates = np.asarray(data["x_raw"], dtype=float)
        response = np.asarray(data["y"], dtype=float).reshape(-1)
        if coordinates.shape != (96, 2) or response.shape != (96,):
            raise ValueError(f"Unexpected data dimensions: {case}")
        if not (np.isfinite(coordinates).all() and np.isfinite(response).all()):
            raise ValueError(f"Nonfinite data: {case}")
        horizontal_count = np.unique(coordinates[:, 0]).size
        boreholes = horizontal_count == 6
        if boreholes:
            counts = np.unique(coordinates[:, 0], return_counts=True)[1]
            assert np.all(counts == 16)
        expected_boreholes = index % 2 == 0
        assert boreholes == expected_boreholes
        layout = "6 boreholes x 16 points" if boreholes else "96 scattered points"
        collection = ax.scatter(
            coordinates[:, 0], coordinates[:, 1], c=response,
            cmap="cividis", norm=Normalize(response.min(), response.max()),
            s=12, edgecolors="#ffffff", linewidths=0.20, zorder=3,
        )
        origin = 500 if index == 4 else 0
        ax.set_xlim(origin - 4, origin + 184)
        ax.set_xticks(origin + np.array([0, 60, 120, 180]))
        ax.set_ylim(18.4, -0.4)
        ax.set_yticks([0, 6, 12, 18])
        ax.set_xlabel("Horizontal position (m)", labelpad=3)
        ax.set_ylabel("Depth (m)", labelpad=3)
        ax.set_title(case, loc="left", fontweight="bold", pad=16)
        ax.text(0, 1.035, layout, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.5, color="#454545")
        ax.tick_params(length=2.5, pad=2.0)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.45, zorder=0)
        colorbar = fig.colorbar(collection, ax=ax, fraction=0.051, pad=0.045)
        colorbar.ax.tick_params(labelsize=6, length=2, pad=2)
        colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        colorbar.set_label("Response (a.u.)", fontsize=6.5, labelpad=3)
        colorbar.outline.set_linewidth(0.5)
        records.append({
            "case": case,
            "input": source.name,
            "sha256": source_hash,
            "observations_before": int(response.size),
            "observations_plotted": int(response.size),
            "observations_excluded": 0,
            "layout": "six_boreholes" if boreholes else "scattered",
            "horizontal_range_m": [float(coordinates[:, 0].min()), float(coordinates[:, 0].max())],
            "depth_range_m": [float(coordinates[:, 1].min()), float(coordinates[:, 1].max())],
            "response_range": [float(response.min()), float(response.max())],
            "independent_linear_color_scale": True,
            "depth_increases_downward": bool(ax.yaxis_inverted()),
        })
    fig.suptitle("Six independent synthetic test datasets", x=0.068, y=0.979,
                 ha="left", fontsize=10, fontweight="bold")
    fig.text(0.068, 0.935, "Each point is one observation; colours show the measured synthetic response.",
             ha="left", fontsize=7, color="#454545")
    fig.text(0.068, 0.036,
             "Separate colour scale in each panel. Depth increases downward. Axes are not to equal physical scale.",
             ha="left", fontsize=6.5, color="#454545")
    basename = OUT / "synthetic_test_data_overview"
    fig.savefig(basename.with_suffix(".png"), dpi=600)
    fig.savefig(basename.with_suffix(".pdf"), metadata={"Creator": "matplotlib"})
    fig.savefig(basename.with_suffix(".svg"))
    exports = []
    for extension in ("png", "pdf", "svg"):
        destination = basename.with_suffix(f".{extension}")
        exports.append({"file": destination.name, "sha256": sha256(destination), "bytes": destination.stat().st_size})
    plt.close(fig)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RENDERED_PENDING_VISUAL_QA",
        "script_sha256": sha256(Path(__file__)),
        "backend": "Python/matplotlib",
        "size_mm": [183, 125],
        "png_dpi": 600,
        "source_inputs": "Six frozen MAT files, only x_raw and y; manifest used for input hashes",
        "candidate_scores_read": False,
        "observations_total": sum(row["observations_plotted"] for row in records),
        "response_transform": "none",
        "interpolation_or_smoothing": "none",
        "cases": records,
        "exports": exports,
    }
    (OUT / "dataset_overview_render_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "observations": receipt["observations_total"], "exports": exports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
