"""Freeze the root-approved geometry-only split and sampling specification.

No response is consulted in the ranking: only CPT IDs and local X coordinates.
"""
import json
from inspect_full_profiles import read_profiles, HERE


def main():
    profiles=read_profiles()
    test_ranks=[2,4,7,9,12,14,17,19,22,24,27]
    test_ids=[p["cpt_id"] for i,p in enumerate(profiles,1) if i in test_ranks]
    pool=[p for i,p in enumerate(profiles,1) if i not in test_ranks]
    chosen=[pool[0],pool[-1]]
    trace=[dict(selection_order=1,cpt_id=pool[0]["cpt_id"],x_local_m=pool[0]["x_local_m"],rule="leftmost training-pool CPT"),
           dict(selection_order=2,cpt_id=pool[-1]["cpt_id"],x_local_m=pool[-1]["x_local_m"],rule="rightmost training-pool CPT")]
    while len(chosen)<len(pool):
        remaining=[p for p in pool if p["cpt_id"] not in {s["cpt_id"] for s in chosen}]
        scored=[(min(abs(p["x_local_m"]-s["x_local_m"]) for s in chosen),p) for p in remaining]
        # Exact ties are broken by smaller local X, then CPT number.
        distance,p=min(scored,key=lambda dp:(-dp[0],dp[1]["x_local_m"],dp[1]["cpt_id"]))
        chosen.append(p)
        trace.append(dict(selection_order=len(chosen),cpt_id=p["cpt_id"],x_local_m=p["x_local_m"],
                          nearest_previously_selected_distance_m=distance,rule="maximize minimum local-X distance; tie uses smaller X"))
    order=[p["cpt_id"] for p in chosen]
    config=dict(archive_sha256="432d0b747d68c2141249d76895157ed0b61f4b1b0545770d9a209c1576216865",
                date="2026-10-11",physical_window_reason="Full dyke-crest profile in the shared original-GEF qc domain, rounded inward to a NAP-anchored 0.2 m grid; no model scores used",
                qc_policy="Original GEF point measurement; -9999 becomes NaN only in clean fields; retain negative values and explicit flags in source and common-domain exports; no smoothing, averaging, interpolation or imputation",
                response_policy="Export both raw qc (MPa) and natural log(qc / 1 MPa); no response-driven sample filter",
                sampling="Nearest original point on grid anchored at NAP 0 m; exact grid matches in this dataset; a tie would use the shallower raw row",
                grids=[dict(step_m=.2,z_min_m=-12.,z_max_m=-.6,purpose="main and training-density comparison"),
                       dict(step_m=.1,z_min_m=-12.,z_max_m=-.6,purpose="same-window vertical-resolution sensitivity")],
                test_x_ranks_1based=test_ranks,test_ids_in_x_order=test_ids,
                training_pool_ids_in_x_order=[p["cpt_id"] for p in pool],
                nested_training_selection_order=order,nested_training_selection_trace=trace,
                splits=[])
    for name,n,tag in [("main_12",12,"0p20m"),("density_06",6,"0p20m"),("density_18",18,"0p20m"),("fine_12",12,"0p10m")]:
        config["splits"].append(dict(name=name,train_ids=order[:n],test_ids=test_ids,grids=[tag],
                 description=f"{n} nested whole-CPT training profiles selected by coordinate-only farthest coverage; same 11 fixed whole-CPT test profiles"))
    (HERE/"preparation_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"test_ids":test_ids,"training_pool_x_order":config["training_pool_ids_in_x_order"],"nested_order":order},indent=2))


if __name__=="__main__":
    main()
