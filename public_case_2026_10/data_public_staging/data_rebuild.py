"""Reproducible public CPT preparation: source-row traceability, no model fitting.

Portable entry point: --archive source/data.zip --output rebuilt.
Uses the frozen adjacent data_preparation_config.json unless --config is supplied.
The output directory must be new or empty; the source archive is never modified.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np
from scipy.io import savemat, loadmat
from data_profiles import read_profiles, common_domain, nearest_samples, ARCHIVE, HERE


DOI = "10.4121/uuid:0a9b77bc-3f6d-40e2-846c-ddb992dd78a6"
CHANNELS = ["qc_mpa", "fs_mpa", "u2_mpa", "inclination_deg", "friction_ratio_percent"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def source_row(p, index, rank, common):
    raw = p["raw"][index]
    is_missing = bool(p["missing"][index])
    nonpositive = bool(p["nonpositive"][index])
    row=dict(cpt_id=p["cpt_id"],x_rank_1based=rank,x_local_m=p["x_local_m"],y_local_m=p["y_local_m"],
             x_rd_m=p["x_rd_m"],y_rd_m=p["y_rd_m"],surface_nap_m=p["surface_nap_m"],
             penetration_m=float(raw[0]),z_nap_m=float(p["z_nap_m"][index]),
             qc_status="missing" if is_missing else ("nonpositive" if nonpositive else "positive"),
             is_measured_row=bool(p["measured"][index]),
             in_common_domain=bool(p["measured"][index] and common[0]-1e-9<=p["z_nap_m"][index]<=common[1]+1e-9),
             gef_member=p["gef_member"],source_row_1based=int(index+1),start_date=p["start_date"])
    for channel, value in zip(CHANNELS, raw[1:]):
        row[channel.replace("_", "_raw_",1)] = float(value)
        row[channel] = float(value) if np.isfinite(value) and value != -9999 else np.nan
    row["log_qc_over_1mpa"] = float(np.log(raw[1])) if not is_missing and raw[1]>0 else np.nan
    return row


def rows_to_arrays(rows):
    return {key:np.asarray([r[key] for r in rows]) for key in rows[0]}


def numeric_case(rows, prefix=""):
    return {f"x{prefix}":np.asarray([[r["x_local_m"],r["z_nap_m"]] for r in rows],dtype=float),
            f"y{prefix}":np.asarray([r["qc_mpa"] for r in rows],dtype=float).reshape(-1,1),
            f"sounding_id{prefix}":np.asarray([r["cpt_id"] for r in rows],dtype=np.int32).reshape(-1,1),
            f"source_row_1based{prefix}":np.asarray([r["source_row_1based"] for r in rows],dtype=np.int32).reshape(-1,1)}


def export_split(rows, split, grid_tag, manifest):
    train_ids=[int(v) for v in split["train_ids"]]
    test_ids=[int(v) for v in split["test_ids"]]
    if not train_ids or not test_ids or set(train_ids)&set(test_ids):
        raise ValueError("Train and test IDs must be nonempty and disjoint")
    if len(set(train_ids))!=len(train_ids) or len(set(test_ids))!=len(test_ids):
        raise ValueError("Duplicate sounding ID in split")
    available=set(r["cpt_id"] for r in rows)
    if (set(train_ids)|set(test_ids))-available:
        raise ValueError("Split includes unknown sounding IDs")
    train=[r for r in rows if r["cpt_id"] in train_ids]
    test=[r for r in rows if r["cpt_id"] in test_ids]
    # No silently excluded/filled values: invalid rows make an export fail.
    if any(not np.isfinite(r["qc_mpa"]) for r in train+test):
        raise ValueError("Missing qc in case export; choose an explicit handling policy")
    payload=numeric_case(train)
    tp=numeric_case(test,"_test")
    tp["test_sounding_id"]=tp.pop("sounding_id_test")
    payload.update(tp)
    payload["x_raw"]=payload["x"].copy()
    payload["train_id"]=payload["sounding_id"].copy()
    payload["test_id"]=payload["test_sounding_id"].copy()
    if np.any(payload["y"]<=0) or np.any(payload["y_test"]<=0):
        raise ValueError("Case log-response export requires strictly positive raw qc; no filter is applied")
    payload["y_log"]=np.log(payload["y"])
    payload["y_test_log"]=np.log(payload["y_test"])
    # x_query is geometry only and includes all 29 registered CPT positions.
    payload["x_query"]=np.asarray([[r["x_local_m"],r["z_nap_m"]] for r in rows],dtype=float)
    stem=split["name"]
    savemat(HERE/f"{stem}.mat",payload,do_compression=True)
    np.savez_compressed(HERE/f"{stem}.npz",**payload)
    meta=dict(case_name=stem,coordinate_names=["local_x_m","z_nap_m"],coordinate_units=["m","m relative to NAP"],
              response_name="qc",response_units="MPa",response_transform="none",train_ids=train_ids,test_ids=test_ids,
              additional_response="y_log and y_test_log contain natural log(qc / 1 MPa); primary response choice is external",
              n_train=len(train),n_test=len(test),split_description=split.get("description",""),
              source_doi=DOI,grid_tag=grid_tag,source_manifest="data_manifest.json",
              test_policy="Test qc must not be used in fitting, model ranking, LLM context, or candidate revision.",
              observation_order="Local X increasing, then NAP elevation increasing within each CPT.",
              x_query_policy="Geometry only; all registered positions. No query outcomes are attached.")
    write_json(HERE/f"{stem}.json",meta)
    manifest["cases"].append(meta)
    # Check numerical payload survives a MATLAB-compatible round trip.
    checked=loadmat(HERE/f"{stem}.mat")
    for key,val in payload.items():
        if not np.array_equal(checked[key],val,equal_nan=True):
            raise AssertionError(f"MAT round-trip mismatch: {stem}, {key}")


def main():
    global HERE, ARCHIVE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("data_preparation_config.json"))
    parser.add_argument("--archive",type=Path,default=Path(__file__).parent / "source" / "data.zip")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    ARCHIVE=args.archive.resolve()
    HERE=args.output.resolve()
    if HERE == Path(__file__).parent.resolve() or HERE == ARCHIVE.parent:
        raise ValueError("Output must be separate from code and source archive")
    if HERE.exists() and any(HERE.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {HERE}")
    config=json.loads(args.config.read_text(encoding="utf-8-sig"))
    if digest(ARCHIVE) != config["archive_sha256"]:
        raise ValueError("Official archive SHA-256 mismatch")
    HERE.mkdir(parents=True,exist_ok=True)
    profiles=read_profiles(ARCHIVE)
    common=common_domain(profiles)
    archive_sha=digest(ARCHIVE)
    if config.get("archive_sha256") and archive_sha!=config["archive_sha256"]:
        raise ValueError("Source archive SHA-256 mismatch")
    all_rows=[source_row(p,i,rank,common) for rank,p in enumerate(profiles,1) for i in range(len(p["raw"]))]
    clean_rows=[r for r in all_rows if r["in_common_domain"]]
    write_csv(HERE/"source_all_rows.csv",all_rows)
    write_csv(HERE/"clean_common_long.csv",clean_rows)
    flat=rows_to_arrays(clean_rows)
    np.savez_compressed(HERE/"clean_common_long.npz",**flat)
    savemat(HERE/"clean_common_long.mat",flat,do_compression=True,oned_as="column",long_field_names=True)
    coords=[dict(cpt_id=p["cpt_id"],x_rank_1based=rank,**{k:p[k] for k in ["x_local_m","y_local_m","x_rd_m","y_rd_m","surface_nap_m","gef_member","gef_sha256"]}) for rank,p in enumerate(profiles,1)]
    write_csv(HERE/"cpt_coordinates.csv",coords)
    manifest=dict(case_name="Leendert de Boerspolder crest, Zone 1, Line 7",source_doi=DOI,
                  license="CC0-1.0",archive_filename=ARCHIVE.name,
                  archive_sha256=archive_sha,ordered_cpt_ids=[p["cpt_id"] for p in profiles],
                  common_z_nap_m=list(common),qc_source="Original GEF column 2; not MAT processed qc",
                  z_definition="GEF NAP surface elevation minus original penetration length; no additional inclination correction",
                  original_rows=len(all_rows),measured_rows=sum(r["is_measured_row"] for r in all_rows),
                  common_domain_rows=len(clean_rows),common_missing_qc=sum(r["qc_status"]=="missing" for r in clean_rows),
                  common_nonpositive_qc=sum(r["qc_status"]=="nonpositive" for r in clean_rows),
                  processing_policy=config,coordinates=coords,grids=[],cases=[],
                  versions={"python":sys.version.split()[0],"numpy":np.__version__},
                  prep_script_sha256=digest(__file__),reader_script_sha256=digest(Path(__file__).with_name("data_profiles.py")))
    for grid_config in config["grids"]:
        step=float(grid_config["step_m"])
        lo=grid_config.get("z_min_m");hi=grid_config.get("z_max_m")
        grid,sampled=nearest_samples(profiles,step,lo,hi)
        tag=f"{step:.2f}".replace(".","p")+"m"
        rows=[]
        for rank,(p,indices) in enumerate(sampled,1):
            if len(np.unique(indices))!=len(indices):
                raise ValueError("Sampling grid reuses a raw point within one CPT")
            for target,index in zip(grid,indices):
                row=source_row(p,int(index),rank,common)
                row.update(z_target_nap_m=float(target),z_offset_m=float(p["z_nap_m"][index]-target),sampling_step_m=step)
                rows.append(row)
        write_csv(HERE/f"sampled_{tag}_long.csv",rows)
        matrix=dict(cpt_id=np.asarray([p["cpt_id"] for p in profiles],dtype=np.int32).reshape(1,-1),
                    x_local_m=np.asarray([p["x_local_m"] for p in profiles]).reshape(1,-1),
                    y_local_m=np.asarray([p["y_local_m"] for p in profiles]).reshape(1,-1),
                    z_target_nap_m=grid.reshape(-1,1),
                    z_actual_nap_m=np.column_stack([p["z_nap_m"][ix] for p,ix in sampled]),
                    qc_mpa=np.column_stack([np.where(p["missing"][ix],np.nan,p["raw"][ix,1]) for p,ix in sampled]),
                    qc_raw_mpa=np.column_stack([p["raw"][ix,1] for p,ix in sampled]),
                    qc_missing=np.column_stack([p["missing"][ix] for p,ix in sampled]),
                    qc_nonpositive=np.column_stack([p["nonpositive"][ix] for p,ix in sampled]),
                    source_row_1based_matrix=np.column_stack([ix+1 for p,ix in sampled]),
                    **numeric_case(rows))
        np.savez_compressed(HERE/f"grid_{tag}.npz",**matrix)
        savemat(HERE/f"grid_{tag}.mat",matrix,do_compression=True)
        info=dict(tag=tag,purpose=grid_config.get("purpose",""),step_m=step,z_min_m=float(grid.min()),
                  z_max_m=float(grid.max()),n_z=len(grid),n_cpts=len(profiles),n_points=len(rows),
                  max_abs_z_offset_m=max(abs(r["z_offset_m"]) for r in rows),
                  missing_qc=sum(r["qc_status"]=="missing" for r in rows),nonpositive_qc=sum(r["qc_status"]=="nonpositive" for r in rows),
                  sampling_method="Nearest original point to NAP-anchored grid; shallower original row breaks ties; no averaging or interpolation")
        manifest["grids"].append(info)
        for split in config.get("splits",[]):
            if tag in split.get("grids",[tag]):
                export_split(rows,split,tag,manifest)
    # Source-trace proof: common-domain raw values must be unaltered GEF observations.
    lookup={p["cpt_id"]:p for p in profiles}
    for row in clean_rows:
        raw=lookup[row["cpt_id"]]["raw"][row["source_row_1based"]-1]
        if row["qc_raw_mpa"]!=raw[1]:
            raise AssertionError("Source qc mismatch")
    manifest["checks"]={"qc_source_row_exact_match":True,"mat_primary_elevation_verified":True,
                        "mat_max_elevation_error_m":max(p["mat_z_max_abs_difference_m"] for p in profiles),
                        "nonpositive_qc_imputed":False,"missing_qc_imputed":False,"model_fitting_performed":False}
    write_json(HERE/"data_manifest.json",manifest)
    print(json.dumps({k:manifest[k] for k in ["common_z_nap_m","original_rows","common_domain_rows","common_missing_qc","common_nonpositive_qc","grids","checks"]},indent=2))


if __name__=="__main__":
    main()
