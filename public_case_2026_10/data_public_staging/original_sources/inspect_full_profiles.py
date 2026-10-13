"""Inspect the predeclared dyke-crest line, without fitting models or selecting a window.

Source GEF values are retained. Figures are diagnostic and do not interpolate
between CPTs. Every diagnostic nearest-point sample retains its source row.
"""
from pathlib import Path
from io import BytesIO
import csv
import hashlib
import json
import re
import zipfile

import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[2] / "output/public_geotechnical_data_search_2026-10-09/delft/data.zip"
CPT_IDS = list(range(34, 45)) + list(range(69, 87))


def read_profiles(archive_path=ARCHIVE):
    profiles = []
    with zipfile.ZipFile(archive_path) as archive:
        xl = loadmat(BytesIO(archive.read("XID_storage.mat")))["XID"].ravel()
        yl = loadmat(BytesIO(archive.read("YID_storage.mat")))["YID"].ravel()
        cells = loadmat(BytesIO(archive.read("CPTR_storage.mat")))["CPTR"].ravel()
        for cpt in CPT_IDS:
            member = f"Raw_data/608312_DKP{200 + cpt}.GEF"
            raw = archive.read(member)
            header, body = re.split(r"#EOH\s*=", raw.decode("latin1"), maxsplit=1)
            tags = {}
            for line in header.splitlines():
                m = re.match(r"#(\w+)\s*=\s*(.*)", line)
                if m:
                    tags.setdefault(m.group(1), []).append(m.group(2))
            a = np.loadtxt(BytesIO(body.encode("latin1")))
            surface = float(tags["ZID"][0].split(",")[1])
            rd = [float(v) for v in tags["XYID"][0].split(",")[1:3]]
            mat = cells[cpt - 1]
            if np.any(np.imag(mat[:, :9]) != 0):
                raise ValueError(f"Complex primary MAT value in CPT {cpt}")
            mat = np.real(mat)
            z = surface - a[:, 0]
            # Some MAT cells omit GEF tail rows when other channels are missing.
            # Align using elevation, never assume MAT row i is GEF row i + 1.
            mat_source_ix = np.argmin(np.abs(z[:,None] - mat[:,2]),axis=0)
            mat_error = float(np.max(np.abs(z[mat_source_ix] - mat[:, 2])))
            if mat_error > 1e-10:
                raise ValueError(f"Elevation mismatch in CPT {cpt}: {mat_error}")
            missing = (~np.isfinite(a[:, 1])) | (a[:, 1] == -9999)
            nonpositive = (~missing) & (a[:, 1] <= 0)
            # Measurement domain excludes only zero-penetration placeholder rows.
            measured = np.isfinite(a[:, 0]) & (a[:, 0] > 0) & (a[:, 0] != -9999)
            profiles.append(dict(cpt_id=cpt, x_local_m=float(xl[cpt-1]),
                                 y_local_m=float(yl[cpt-1]), x_rd_m=rd[0], y_rd_m=rd[1],
                                 surface_nap_m=surface, raw=a, z_nap_m=z,
                                 missing=missing, nonpositive=nonpositive, measured=measured,
                                 gef_member=member, gef_sha256=hashlib.sha256(raw).hexdigest(),
                                 start_date=tags["STARTDATE"][0],
                                 mat_z_max_abs_difference_m=mat_error,
                                 mat_rows=len(mat),
                                 mat_qc_max_abs_difference_mpa=float(np.max(np.abs(mat[:,3]-a[mat_source_ix,1])))))
    profiles.sort(key=lambda p: (p["x_local_m"], p["cpt_id"]))
    return profiles


def common_domain(profiles):
    return (max(float(p["z_nap_m"][p["measured"]].min()) for p in profiles),
            min(float(p["z_nap_m"][p["measured"]].max()) for p in profiles))


def nearest_samples(profiles, step_m, lower=None, upper=None):
    lo, hi = common_domain(profiles)
    lo = lo if lower is None else max(lo, lower)
    hi = hi if upper is None else min(hi, upper)
    # NAP-anchored grid. Rounding makes decimal-metre grid values reproducible.
    grid = np.round(np.arange(np.ceil((lo-1e-9)/step_m), np.floor((hi+1e-9)/step_m)+1)*step_m, 8)
    sampled = []
    for p in profiles:
        available = np.flatnonzero(p["measured"])
        # A tie selects the earlier GEF row, i.e. shallower penetration length.
        ix = available[np.argmin(np.abs(p["z_nap_m"][available,None] - grid), axis=0)]
        sampled.append((p, ix))
    return grid, sampled


def main():
    profiles = read_profiles()
    lo, hi = common_domain(profiles)
    qc_all = np.concatenate([p["raw"][p["measured"] & ~p["missing"],1] for p in profiles])
    summary = dict(n_cpts=len(profiles), ordered_cpt_ids=[p["cpt_id"] for p in profiles],
                   common_z_nap_m=[lo,hi], archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                   qc_mpa_quantile_probabilities=[0,.01,.05,.25,.5,.75,.95,.99,1],
                   qc_mpa_quantiles=np.quantile(qc_all,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),
                   profiles=[], nonpositive_points=[], nearest_grid_diagnostics={})
    bins = np.arange(-12.5,0.5001,.5)
    bin_rows = []
    for low, high in zip(bins[:-1],bins[1:]):
        q = np.concatenate([p["raw"][(p["z_nap_m"]>=low)&(p["z_nap_m"]<high)&p["measured"]&~p["missing"],1] for p in profiles])
        if len(q):
            quant = np.quantile(q,[.05,.25,.5,.75,.95])
            bin_rows.append(dict(z_bottom_nap_m=low,z_top_nap_m=high,n=len(q),qc_mean_mpa=float(q.mean()),
                                 qc_sd_mpa=float(q.std(ddof=1)),qc_p05_mpa=quant[0],qc_p25_mpa=quant[1],
                                 qc_p50_mpa=quant[2],qc_p75_mpa=quant[3],qc_p95_mpa=quant[4]))
    for rank,p in enumerate(profiles,1):
        within = p["measured"]&(p["z_nap_m"]>=lo-1e-9)&(p["z_nap_m"]<=hi+1e-9)
        summary["profiles"].append(dict(x_rank_1based=rank,**{k:p[k] for k in ["cpt_id","x_local_m","y_local_m","surface_nap_m","gef_member","gef_sha256","mat_z_max_abs_difference_m","mat_qc_max_abs_difference_mpa"]},
                                        raw_rows=len(p["raw"]), mat_rows=p["mat_rows"], measured_rows=int(p["measured"].sum()),
                                        common_domain_rows=int(within.sum()),missing_measured_qc=int((p["measured"]&p["missing"]).sum()),
                                        nonpositive_qc=int(p["nonpositive"].sum())))
        for i in np.flatnonzero(p["nonpositive"]):
            summary["nonpositive_points"].append(dict(cpt_id=p["cpt_id"],gef_row_1based=int(i+1),z_nap_m=float(p["z_nap_m"][i]),qc_mpa=float(p["raw"][i,1]),in_common_domain=bool(within[i])))
    for step in [.1,.2]:
        grid, sampled = nearest_samples(profiles,step)
        summary["nearest_grid_diagnostics"][str(step)] = dict(n_z=len(grid),n_points=len(grid)*len(profiles),
                    z_grid_min=float(grid.min()),z_grid_max=float(grid.max()),
                    missing_qc=sum(int(p["missing"][ix].sum()) for p,ix in sampled),
                    nonpositive_qc=sum(int(p["nonpositive"][ix].sum()) for p,ix in sampled),
                    max_abs_grid_offset_m=max(float(np.abs(p["z_nap_m"][ix]-grid).max()) for p,ix in sampled))
    (HERE/"full_profile_diagnostics.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    with (HERE/"qc_by_halfmetre_elevation.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(bin_rows[0]));writer.writeheader();writer.writerows(bin_rows)
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":9,"pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(12,6),gridspec_kw={"width_ratios":[1.6,1]},layout="constrained",sharey=True)
    positive = qc_all[qc_all>0]
    norm=LogNorm(vmin=float(positive.min()),vmax=float(positive.max()))
    for p in profiles:
        valid=p["measured"]&~p["missing"]&~p["nonpositive"]
        artist=axes[0].scatter(np.full(valid.sum(),p["x_local_m"]),p["z_nap_m"][valid],c=p["raw"][valid,1],s=2.5,marker="s",cmap="viridis",norm=norm,rasterized=True,linewidths=0)
        axes[0].scatter(np.full(p["nonpositive"].sum(),p["x_local_m"]),p["z_nap_m"][p["nonpositive"]],color="#C65044",marker="x",s=16)
        axes[1].plot(np.where(valid,p["raw"][:,1],np.nan),p["z_nap_m"],color="#527b91",alpha=.22,lw=.6)
    for ax in axes:
        ax.axhline(lo,color="#555555",ls="--",lw=.7)
        ax.axhline(hi,color="#555555",ls="--",lw=.7)
        ax.set_ylim(-12.5,0);ax.set_yticks(np.arange(-12,1,1));ax.grid(axis="y",alpha=.12)
    axes[0].set(xlabel="Distance along crest, local X (m)",ylabel="Elevation relative to NAP (m)",title="a  All 29 crest profiles; measured points only")
    axes[1].set(xlabel="Cone resistance, qc (MPa; log scale)",title="b  Raw profiles overlaid",xscale="log")
    cb=fig.colorbar(artist,ax=axes[0],pad=.025,fraction=.045);cb.set_label("qc (MPa; logarithmic colour scale)")
    fig.suptitle("Full-profile diagnostic: crest-line heterogeneity across the measured elevation range",fontsize=11)
    fig.savefig(HERE/"full_profile_diagnostic.png",dpi=300)
    fig.savefig(HERE/"full_profile_diagnostic.pdf")
    fig.savefig(HERE/"full_profile_diagnostic.svg")
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!="profiles"},ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
