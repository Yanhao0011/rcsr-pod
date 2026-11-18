#!/usr/bin/env python3
"""Primary (coding-invariant) family of the v3 application: every covariate is one unit -- a continuous or binary
covariate as its single column, a categorical variable as the group of all its dummies (v3g_*.csv), the source
indicator separately as a frame term. Holm/BH over this family; delta sweep; the held-out specification sets.
Outputs: outputs/v3/v3_primary_{pathway}.csv, v3_primary_summary.json, heldout_specs.json"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task_a as TA
import run_v3_application as APP
import rcsr_v3 as V_
from run_v3_frame import SOURCE_TERM, frame_units, is_frame

OUT = APP.ROOT / "outputs" / "v3"; DELTA, GRID = APP.DELTA, APP.GRID


def main():
    summ, specs = {}, {}
    for pw in ["pavement", "infrastructure"]:
        allu = pd.read_csv(OUT / f"v3_{pw}.csv"); f = allu[allu.kind == "feature"].copy()
        g = allu[allu.kind == "group"].copy()            # groups include every non-source level (run_v3_application.units)
        feats = TA.PATHWAY_FEATURES[pw] + TA.CONTROL_FEATURES
        parent = {u: APP.parent_of(u, feats) for u in f.unit}
        grouped = {u.replace(" [group]", "") for u in g.unit}
        single = f[[(parent[u] not in grouped) or is_frame(u, pw) for u in f.unit]].copy()
        cols = ["unit", "kind", "prevalence", "Dmax", "Dmax_cut", "Dmax_nf", "cl_all_lcb", "cl_all_lcb_marg", "m_all_lcb", "cl_nf_lcb", "cl_all_c"] + [f"cl_all_p{d}" for d in GRID] + [c for c in ["cl_all_ncut", "rel_Dmax", "rel_lcb0.02", "rel_lcb0.05", "rel_lcb0.1", "j_all_lcb"] if c in allu.columns]
        single["n_cols"] = 1
        prim = pd.concat([single[cols + ["n_cols"]], g.assign(kind="group")[[c for c in cols + ["n_cols"]]]], ignore_index=True)
        prim["frame_term"] = prim.unit.isin(frame_units(pw))
        prim["label"] = [TA.LABELS.get(u, u.replace(" [group]", "").replace("mode_", "")) for u in prim.unit]
        for c in [f"cl_all_p{d}" for d in GRID]:
            prim[c] = prim[c].fillna(1.0)
        sub = prim[~prim.frame_term]
        for d in GRID:
            h, b = V_.multitest(sub[f"cl_all_p{d}"].values)
            prim.loc[sub.index, f"holm_{d}"] = h; prim.loc[sub.index, f"bh_{d}"] = b
        prim = prim.sort_values("Dmax", ascending=False); prim.to_csv(OUT / f"v3_primary_{pw}.csv", index=False)
        sub = prim[~prim.frame_term]
        cred = sub.loc[sub.cl_all_lcb > DELTA, "unit"].tolist()
        summ[pw] = dict(units=int(len(sub)), point_above_delta=sub.loc[sub.Dmax > DELTA, "unit"].tolist(), credible=cred,
                        marginal=sub.loc[sub.cl_all_lcb_marg > DELTA, "unit"].tolist(), model=sub.loc[sub.m_all_lcb > DELTA, "unit"].tolist(),
                        nonfatal=sub.loc[sub.cl_nf_lcb > DELTA, "unit"].tolist(), holm=sub.loc[sub[f"holm_{DELTA}"].astype(bool), "unit"].tolist(),
                        bh=sub.loc[sub[f"bh_{DELTA}"].astype(bool), "unit"].tolist(),
                        sweep_simultaneous={str(d): int((sub.cl_all_lcb > d).sum()) for d in GRID},
                        sweep_marginal={str(d): int((sub.cl_all_lcb_marg > d).sum()) for d in GRID},
                        sweep_holm={str(d): int(sub[f"holm_{d}"].astype(bool).sum()) for d in GRID},
                        relative={str(r): sub.loc[sub.get(f"rel_lcb{r}", pd.Series(-1.0, index=sub.index)) > r, "unit"].tolist() for r in [0.02, 0.05, 0.1]},
                        sets_by_delta={str(d): sub.loc[sub.cl_all_lcb > d, "unit"].tolist() for d in GRID},
                        source=prim.loc[prim.frame_term, ["Dmax", "cl_all_lcb"]].iloc[0].to_dict(),
                        near=sub.loc[(sub.cl_all_lcb <= DELTA) & (sub.Dmax > DELTA), ["unit", "Dmax", "cl_all_lcb"]].to_dict("records"))
        names = list(np.load(OUT / f"firth_fit_{pw}.npz", allow_pickle=True)["names"])
        free = []
        for u in cred:
            free += [n for n in names if APP.parent_of(n, feats) == u.replace(" [group]", "") and n != SOURCE_TERM[pw]] if u.endswith("[group]") else [u]
        js = json.loads((OUT / "v3_summary_pavement_infrastructure_B1000.json").read_text())
        dummy = [s for s in js if s["pathway"] == pw][0]["simultaneous"]
        specs[pw] = {"RCSR-POD": free + frame_units(pw), "RCSR-POD (dummy units)": list(dict.fromkeys(dummy + frame_units(pw)))}
        summ[pw]["n_threshold_specific_params"] = 3 * len(specs[pw]["RCSR-POD"]); summ[pw]["n_features"] = len(names)
        print(pw, json.dumps(summ[pw], indent=1, default=float))
    (OUT / "v3_primary_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    (OUT / "heldout_specs.json").write_text(json.dumps(specs, indent=1))


if __name__ == "__main__":
    main()
