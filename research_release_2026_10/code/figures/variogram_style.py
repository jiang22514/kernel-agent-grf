"""Unchanged projected-observation diagnostic and renderer; v2 inputs only."""
import numpy as np

def residual_projection(x):
    """SVD handles the same column space as B @ pinv(B), including rank loss."""
    x = np.asarray(x, dtype=float)
    B = np.column_stack((np.ones(len(x)), x))
    u, s, _ = np.linalg.svd(B, full_matrices=False)
    rank = int(np.sum(s > max(B.shape) * np.finfo(float).eps * s[0]))
    q = u[:, :rank]
    return np.eye(len(x)) - q @ q.T, rank

def pair_expectation(R, mu, C, i, j):
    rm = R @ mu
    rc = R @ C @ R.T
    rc = (rc + rc.T) / 2
    g = 0.5 * ((rm[i] - rm[j]) ** 2 + rc[i, i] + rc[j, j] - 2 * rc[i, j])
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(C))), float(np.max(rm**2)))
    if np.any(g < -tolerance):
        raise ValueError("Negative expected semivariance beyond round-off tolerance")
    return g

def compute(x, y, mu, cov, edges, labels):
    x, y, mu, cov = map(lambda a: np.asarray(a, dtype=float), (x, y, mu, cov))
    n = len(y)
    assert x.shape == (n, 2) and mu.shape == (n, 3) and cov.shape == (n, n, 3)
    assert all(np.isfinite(a).all() for a in (x, y, mu, cov, edges))
    assert np.all(np.diff(edges) > 0)
    R, rank = residual_projection(x)
    residual = R @ y
    i, j = np.triu_indices(n, 1)
    sel = np.abs(x[i, 0] - x[j, 0]) < 2.0
    i, j = i[sel], j[sel]
    lag = np.abs(x[i, 1] - x[j, 1])
    empirical = 0.5 * (residual[i] - residual[j]) ** 2
    expected = np.column_stack([pair_expectation(R, mu[:, m], cov[:, :, m], i, j) for m in range(3)])
    rows = []
    binned_count = 0
    for b, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        keep = (lag >= left) & (lag < right)
        count = int(keep.sum())
        if not count:
            continue
        binned_count += count
        row = {"bin": b + 1, "left_m": float(left), "right_m": float(right),
               "lag_mean_m": float(lag[keep].mean()), "n_pairs": count,
               "empirical_MPa2": float(empirical[keep].mean())}
        row.update({f"model_{m+1}_MPa2": float(expected[keep, m].mean()) for m in range(3)})
        rows.append(row)
    assert rows and binned_count <= len(i)
    report = {
        "protocol": "projected_observation_semivariance_v1",
        "n_observations": n, "all_possible_pairs": n * (n - 1) // 2,
        "near_vertical_pairs": len(i), "pairs_in_bins": binned_count,
        "near_vertical_pairs_outside_range": len(i) - binned_count,
        "excluded_nonvertical_pairs": n * (n - 1) // 2 - len(i),
        "empty_bins": len(edges) - 1 - len(rows), "trend_rank": rank,
        "n_distinct_horizontal_locations": len(np.unique(x[:, 0])),
        "max_selected_horizontal_separation_m": float(np.max(np.abs(x[i, 0] - x[j, 0]))),
        "labels": list(labels),
        "interpretation": "In-sample fitted observation-model diagnostic at fixed parameters; not a conditional-field or out-of-sample validation.",
        "uncertainty": "No confidence interval: point pairs share observations. Parameter-estimation uncertainty is not included.",
        "noise": "Observation variance is included once in C, before residual projection. No numerical jitter is included.",
        "selection": "All 744 observations; unique i<j pairs with abs(dx1)<2m; 12 left-closed/right-open bins up to MATLAB 60th lag percentile.",
    }
    return rows, report

def render(rows, labels, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(183 / 25.4, 112 / 25.4), layout="constrained")
    h = [r["lag_mean_m"] for r in rows]
    for m, (color, style) in enumerate(zip(("#D55E00", "#009E73", "#0072B2"), ("--", "-.", "-"))):
        ax.plot(h, [r[f"model_{m+1}_MPa2"] for r in rows], color=color,
                linestyle=style, linewidth=1.6, label=labels[m])
    ax.scatter(h, [r["empirical_MPa2"] for r in rows], color="black", s=23,
               zorder=5, label="Observations (same detrending and pairs)")
    ax.set(xlabel="Mean vertical separation within bin (m)", ylabel="Semivariance (MPa²)", ylim=(0, None))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.savefig(outdir / 'borehole_variogram_corrected.svg')
    fig.savefig(outdir / 'borehole_variogram_corrected.pdf')
    fig.savefig(outdir / 'borehole_variogram_corrected.png', dpi=600)
    plt.close(fig)
