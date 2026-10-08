"""Adapted display helpers: remove obsolete interpolated-grid comparator only."""

import numpy as np

from common import need as require

CASES=('EX1','EX2','EX3','EX4')

def display_matrix(values: np.ndarray, index: np.ndarray, ng: int) -> np.ndarray:
    """MATLAB meshgrid(:) maps back with column-major order, not NumPy default."""
    return np.asarray(values)[index].reshape((ng, ng), order="F")

def color_limits(data: dict) -> tuple[float, float, float]:
    idx = data["index"]
    values = np.concatenate([data[k][idx] for k in ("truth", "exact", "bcs")])
    errors = np.concatenate([(data[k] - data["truth"])[idx] for k in ("exact", "bcs")])
    return float(values.min()), float(values.max()), max(float(np.abs(errors).max()), 1e-12)

def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.linewidth": 0.5, "legend.frameon": False, "text.usetex": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    return plt

def make_figure(cases: list[dict], plt):
    nrows = len(cases)
    height_mm = 190 if nrows == 4 else 63
    fig = plt.figure(figsize=(7.204724409448819, height_mm / 25.4))  # 183 mm wide
    # Fixed millimetre positions leave explicit room for scale ticks and row notes.
    map_x_mm = [13, 42, 71, 112, 141]
    map_width_mm = 24
    titles = ["Known truth", "Exact GP", "BCS", "Exact GP\n− truth", "BCS\n− truth"]
    axes_all = []
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator
    for row, d in enumerate(cases):
        top_mm = 174 - 42 * row if nrows == 4 else 45
        bottom_mm = top_mm - map_width_mm
        vmin, vmax, emax = color_limits(d)
        value_norm, error_norm = Normalize(vmin=vmin, vmax=vmax), TwoSlopeNorm(vmin=-emax, vcenter=0, vmax=emax)
        arrays = [d["truth"], d["exact"], d["bcs"], d["exact"] - d["truth"], d["bcs"] - d["truth"]]
        axes, images = [], []
        for col, (x_mm, values) in enumerate(zip(map_x_mm, arrays)):
            ax = fig.add_axes([x_mm / 183, bottom_mm / height_mm,
                               map_width_mm / 183, map_width_mm / height_mm])
            image = ax.imshow(display_matrix(values, d["index"], d["ng"]), origin="lower", extent=(0, 1, 0, 1),
                              interpolation="nearest", aspect="equal", cmap="viridis" if col < 3 else "RdBu_r",
                              norm=value_norm if col < 3 else error_norm, rasterized=True)
            ax.set(xlabel=r"$x_1$", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], xlim=(0, 1), ylim=(0, 1))
            ax.tick_params(length=2, pad=1.5)
            if col == 0:
                ax.set_ylabel(r"$x_2$", labelpad=2)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                fig.text((x_mm + map_width_mm / 2) / 183, (183 if nrows == 4 else 50) / height_mm,
                         titles[col], ha="center", va="bottom", fontsize=7, linespacing=1.2)
            axes.append(ax)
            images.append(image)
        for x_mm, image, label in ((104, images[0], "Value"), (174, images[3], "Error")):
            cax = fig.add_axes([x_mm / 183, bottom_mm / height_mm, 1.3 / 183, map_width_mm / height_mm])
            cb = fig.colorbar(image, cax=cax)
            lo, hi = image.norm.vmin, image.norm.vmax
            ticks = MaxNLocator(nbins=3).tick_values(lo, hi)
            cb.set_ticks(ticks[(ticks >= lo) & (ticks <= hi)])
            cb.ax.yaxis.set_ticks_position("left")
            cb.ax.tick_params(labelsize=6, length=2, pad=1.5)
            cb.ax.set_title(label, fontsize=6, pad=3)
            cb.outline.set_linewidth(0.4)
        mean = {1: "zero", 2: "constant", 3: "linear", 4: "quadratic"}[int(d["meta"]["mean_id"])]
        expr = str(d["meta"]["expr"]).replace("*", "×")
        tag = f"{chr(97 + (CASES.index(d['case']) if nrows == 1 else row))}  {d['case']}  ·  {expr} + {mean} mean"
        if d["dim"] == 3:
            tag += r"  ·  $x_3=9/19$ (0.4737)"
        fig.text(13 / 183, (top_mm + 3 if nrows == 4 else 59) / height_mm,
                 tag, ha="left", va="bottom", fontsize=7, weight="bold")
        m = d["metrics"]
        stats = f"Full-grid RMSE: Exact GP {m['rmse_exact']:.4f}   ·   BCS {m['rmse_bcs']:.4f}"
        if m["rmse_bcs"] < m["rmse_exact"]:
            stats += "   ·   BCS is lower"
        elif m["rmse_exact"] < m["rmse_bcs"]:
            stats += "   ·   Exact GP is lower"
        fig.text(13 / 183, (bottom_mm - 8) / height_mm, stats, ha="left", va="top", fontsize=6.3)
        axes_all.extend(axes)
    note = ("Coordinates, responses and errors are dimensionless; negative/positive error = under-/over-prediction.\n"
            "Each case: 100 observations. RMSE uses all 60² points (EX1–3) or 20³ points (EX4); EX4 maps show one slice.")
    fig.text(13 / 183, 3 / height_mm, note, ha="left", va="bottom", fontsize=6.3, linespacing=1.5)
    fig.canvas.draw()
    from matplotlib.text import Text
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    for label in fig.findobj(Text):
        if label.get_visible() and label.get_text().strip():
            box = label.get_window_extent(renderer)
            require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width + 1 and box.y1 <= height + 1,
                    f"Figure text outside canvas: {label.get_text()}")
    return fig, axes_all
