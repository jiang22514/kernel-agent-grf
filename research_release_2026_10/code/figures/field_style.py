"""Frozen plotting/HDF helpers only; no experiment loaders or numerical GP calls."""

import numpy as np

import h5py

from common import need as require

def vector(value):
    return np.asarray(value, dtype=float).ravel(order="F")

class Mat73:
    """Selective v7.3 reader; HDF dimensions reverse MATLAB dimensions."""
    def __init__(self, path):
        self.file = h5py.File(path, "r")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def resolve(self, node):
        while isinstance(node, h5py.Dataset) and h5py.check_dtype(ref=node.dtype) is not None and node.size == 1:
            ref = node[()].ravel()[0]
            require(bool(ref), "Null MATLAB reference")
            node = self.file[ref]
        return node

    def read(self, node):
        if isinstance(node, str):
            node = self.file[node]
        if isinstance(node, h5py.Group):
            return {k: self.read(v) for k, v in node.items() if not k.startswith("#")}
        cls = node.attrs.get("MATLAB_class", b"")
        if isinstance(cls, bytes):
            cls = cls.decode("ascii")
        if int(np.asarray(node.attrs.get("MATLAB_empty", 0)).ravel()[0]):
            return {} if cls == "struct" else []
        a = node[()]
        if a.ndim > 1:
            a = a.transpose(tuple(reversed(range(a.ndim))))
        if h5py.check_dtype(ref=node.dtype) is not None:
            values = [self.read(self.file[r]) if r else None for r in a.ravel(order="F")]
            return values[0] if len(values) == 1 else values
        if cls == "char":
            return np.asarray(a, dtype="<u2").ravel(order="F").tobytes().decode("utf-16-le")
        return a.item() if a.size == 1 else a

    def field(self, group, key):
        return self.read(self.resolve(self.file[group])[key])

    def samples(self):
        ds = self.resolve(self.resolve(self.file["pack"])["F"])
        require(isinstance(ds, h5py.Dataset) and ds.ndim == 2 and ds.dtype.kind == "f",
                "F must be real floating-point samples-by-query HDF storage")
        return ds

def weights(g1, g2):
    def axis(g):
        g = vector(g)
        require(len(g) >= 2 and np.isfinite(g).all() and (np.diff(g) > 0).all(), "Invalid grid axis")
        require(np.allclose(np.diff(g), np.diff(g)[0], rtol=1e-10, atol=1e-10), "Nonuniform physical query grid")
        w = np.ones(len(g)); w[[0, -1]] = .5
        return w * (g[1] - g[0])
    return np.outer(axis(g2), axis(g1)).ravel(order="F")

def draw_figures(batch, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.right": False,
        "axes.spines.top": False, "axes.linewidth": .65, "pdf.fonttype": 42, "svg.fonttype": "none"})
    order = ["SE_main", "RQ_main", "champion_main"]
    labels = {"SE_main": "SE", "RQ_main": "RQ", "champion_main": "Selected kernel"}
    colors = {"SE_main": "#7D8E9F", "RQ_main": "#CB8E63", "champion_main": "#357F95"}
    scalar = {r["unit"]: r for r in batch["runs"]}
    wp = {u: np.array([r["weak_area_fraction"] for r in batch["perfield"] if r["unit"] == u]) for u in order}
    qa = {"font_body_pt": 7.5, "width_mm": 183, "raster_dpi": 600, "all_500_fields_retained": True,
          "new_hypothesis_tests": False, "maps_latent_moments_not_observation_intervals": True,
          "representative_field_rule": "First stored field, fixed before looking at outputs", "figures": []}
    def save(fig, name):
        fig.canvas.draw()
        fig.savefig(output / (name + ".pdf"), facecolor="white")
        fig.savefig(output / (name + ".svg"), facecolor="white")
        fig.savefig(output / (name + ".png"), dpi=600, facecolor="white")
        qa["figures"].append({"name": name, "width_mm": fig.get_figwidth() * 25.4,
                              "height_mm": fig.get_figheight() * 25.4})
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.20472440945, 3.07086614173), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=.085, right=.985, bottom=.23, top=.88, wspace=.38)
    for u in order:
        v = np.sort(wp[u]); axes[0].step(v * 100, np.arange(1, 501) / 500, where="post", label=labels[u], color=colors[u], lw=1.2)
    axes[0].axvline(10, ls="--", lw=.7, color=".45")
    axes[0].set(xlabel="Weak-area fraction (%)", ylabel="Empirical cumulative probability", ylim=(0, 1.02))
    axes[0].legend(loc="lower right", frameon=False)
    for j, u in enumerate(order):
        r = scalar[u]; p = r["event_probability"] * 100
        errors = np.array([p - 100 * r["wilson95_low"], 100 * r["wilson95_high"] - p])
        require(np.min(errors) >= -1e-12, "Wilson display interval excludes the point estimate")
        axes[1].errorbar(p, 2 - j, xerr=np.maximum(errors, 0).reshape(2, 1),
                         fmt="o", color=colors[u], ms=4, capsize=2, lw=1)
        axes[1].annotate(f"{r['event_count']}/500 [{100*r['wilson95_low']:.2f}, {100*r['wilson95_high']:.2f}]%",
                         (p, 2-j), xytext=(-3,-12), textcoords="offset points", ha="right", fontsize=7)
    axes[1].set(yticks=[2, 1, 0], yticklabels=[labels[u] for u in order], ylim=(-.5, 2.5), xlim=(0, 102),
                xlabel="P(weak-area fraction > 10%) (%)")
    for a, label in zip(axes, ("a", "b")):
        a.text(-.12, 1.07, label, transform=a.transAxes, weight="bold", fontsize=9)
    fig.text(.5, .035, "500 conditional fields per model; common 201 × 121 query grid. Whiskers: Wilson 95% Monte Carlo intervals.", ha="center", fontsize=7)
    save(fig, "main_model_weak_area")

    fs = batch["fields"]
    amplitude = np.concatenate([fs[u][k] for u in order for k in ("mu", "realization1")])
    vmin, vmax = float(amplitude.min()), float(amplitude.max())
    if vmin == vmax:
        vmin -= .5; vmax += .5
    sdmax = max(float(fs[u]["sd"].max()) for u in order)
    qa["map_color_limits_MPa"] = {"mean_and_first_field": [vmin, vmax], "latent_sd": [0, sdmax]}
    fig = plt.figure(figsize=(7.20472440945, 6.29921259843))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, .045], left=.105, right=.935, bottom=.09, top=.93, hspace=.32, wspace=.15)
    for row, (key, rowlabel) in enumerate((("mu", "Conditional mean"), ("sd", "Latent field SD"), ("realization1", "First saved realization"))):
        norm = Normalize(0, sdmax) if key == "sd" else Normalize(vmin, vmax)
        for col, u in enumerate(order):
            ax = fig.add_subplot(gs[row, col]); f = fs[u]
            z = f[key].reshape((len(f["g2"]), len(f["g1"])), order="F")
            im = ax.imshow(z, origin="lower", extent=[f["g1"][0], f["g1"][-1], f["g2"][0], f["g2"][-1]],
                           aspect="auto", interpolation="nearest", cmap="cividis" if key == "sd" else "viridis", norm=norm)
            ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(4))
            if row == 0: ax.set_title(labels[u])
            if row == 2: ax.set_xlabel("Horizontal coordinate (m)")
            else: ax.tick_params(labelbottom=False)
            if col == 0: ax.set_ylabel("Vertical coordinate (m)")
            else: ax.tick_params(labelleft=False)
            ax.text(.02, 1.035, chr(ord("a") + row * 3 + col), transform=ax.transAxes, weight="bold", fontsize=8)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[row, 3])); cb.set_label("MPa")
        fig.text(.105, [.975, .66, .365][row], rowlabel, fontsize=8)
    save(fig, "main_model_fields")

    pairs = [("champion_fine", "Selected kernel: fine − main"), ("SE_fine", "SE: fine − main"), ("RQ_main", "RQ: main − coarse")]
    ds = {u: np.array([r["same_field_finer_minus_coarser"] for r in batch["perfield"] if r["unit"] == u]) for u, _ in pairs}
    limit = max(float(np.max(np.abs(v))) * 100 for v in ds.values())
    limit = max(limit * 1.05, .001)
    fig, axes = plt.subplots(1, 3, figsize=(7.20472440945, 2.83464566929), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.26, top=.78, wspace=.20)
    for j, (ax, (u, title)) in enumerate(zip(axes, pairs)):
        v = np.sort(ds[u]) * 100
        ax.step(v, np.arange(1, 501) / 500, where="post", color=colors.get(u.replace("fine", "main"), "#357F95"), lw=1.2)
        ax.axvline(0, ls="--", color=".5", lw=.7)
        ax.set(title=title, xlim=(-limit, limit), ylim=(0, 1.02), xlabel="Same-field difference (percentage points)")
        ax.text(.02, .04, f"Median {100*scalar[u]['paired_delta_median']:.3g}\nRMS {100*scalar[u]['paired_delta_rms']:.3g}", transform=ax.transAxes, fontsize=7)
        ax.text(-.12, 1.20, chr(ord("a") + j), transform=ax.transAxes, weight="bold", fontsize=9)
    axes[0].set_ylabel("Empirical cumulative probability")
    fig.text(.5, .035, "Each curve uses all 500 differences from nested nodes of the same saved field; no independent-run pairing.", ha="center", fontsize=7)
    save(fig, "same_field_grid_sensitivity")
    return qa
