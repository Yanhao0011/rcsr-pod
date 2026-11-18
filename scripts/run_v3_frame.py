#!/usr/bin/env python3
"""v3 analysis frame: de-duplicate the FARS-CRSS fusion and refit the full-frame Firth cut models.

The fusion (build_fars_crss_fusion.fuse_fars_and_crss) removes CRSS fatal PERSONS, but the non-fatal persons of a
CRSS-sampled fatal crash remained, although the same crash is in the FARS census with all its persons. Those CRSS
persons are dropped here (crash identified by YEAR and CASENUM in the raw CRSS person file), so every fatal crash
enters the frame once, through FARS. The Firth fits are then refitted on the de-duplicated frame, and a FLIC-type
intercept check is reported (mean fitted P(y > t) against the observed share; the ML intercept re-estimated with the
Firth slopes held as an offset).

Reads outputs/kabco_person_level_clean.csv (build_kabco_modeling_data.py) and data/CRSS{year}NationalCSV.zip.
Writes outputs/v3/kabco_person_level_v3.csv, outputs/v3/firth_fit_{pathway}.npz, outputs/v3/frame_v3_report.json
Paths are relative to RCPO_ROOT (default: the repository root, the parent of scripts/); RCPO_DATA overrides data/.
"""
from __future__ import annotations
import sys, json, time, zipfile
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_departure_diag as RD
import run_task_a as TA

ROOT = Path(os.environ.get("RCPO_ROOT", HERE.parent)); OUT = ROOT / "outputs" / "v3"; OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("RCPO_DATA", ROOT / "data"))      # raw NHTSA zips, e.g. data/CRSS2021NationalCSV.zip
# Encoded columns dropped from the v3 designs: the harm-event-12 dummy (nested in the manner-of-collision dummies) and
# two of the three infrastructure MISSING levels, which are identical to RUR_URB_MISSING (all three equal the CRSS
# indicator); the remaining one is the single source indicator of each pathway.
V3_DROP = {"infrastructure": ["HARM_EV_12.0", "FUNC_SYS_MISSING", "ROUTE_MISSING"], "pavement": []}
SOURCE_TERM = {"infrastructure": "RUR_URB_MISSING", "pavement": "mode_vpavetyp_MISSING"}
# selection-related controls: log persons per crash (the FARS census enters whole fatal crashes, so a
# person's chance of being the fatality falls with crash size) and person type; persons_in_crash is a frame term.
SELECTION_CONTROLS = ["persons_in_crash", "person_type"]
FRAME_EXTRA = ["persons_in_crash"]


def frame_units(pathway):
    return [SOURCE_TERM[pathway]] + FRAME_EXTRA


def is_frame(unit, pathway):
    return unit in frame_units(pathway)
# unknown-value indicators of the de-sentineled continuous fields enter the designs next to their field
for _c in ["AGE_UNK"]:
    if _c not in TA.CONTROL_FEATURES: TA.CONTROL_FEATURES.append(_c)
if "veh_vnum_lan_UNK" not in TA.PATHWAY_FEATURES["pavement"]: TA.PATHWAY_FEATURES["pavement"].append("veh_vnum_lan_UNK")
for _c in ["persons_in_crash", "person_type"]:
    if _c not in TA.CONTROL_FEATURES: TA.CONTROL_FEATURES.append(_c)


def lg(v): return 0.5 * (1.0 + np.tanh(0.5 * v))


# Fields that the earlier frame carried as numeric codes. Nominal codes become categories (sentinel codes for
# unknown / not reported become one 'UNK' level, levels with fewer than 100 records 'OTHER'); continuous fields
# lose their sentinels (median-imputed with a missing indicator).
NOMINAL = {"SEX": {8: "UNK", 9: "UNK"}, "DRINKING": {8: "UNK", 9: "UNK"}, "WRK_ZONE": {}, "LGT_COND": {7: "UNK", 8: "UNK", 9: "UNK"},
           "WEATHER": {98: "UNK", 99: "UNK"}, "mode_valign": {8: "UNK", 9: "UNK"}, "mode_vprofile": {8: "UNK", 9: "UNK"}}
SENTINEL = {"AGE": 998, "veh_vnum_lan": 8}            # values >= sentinel are unknown / not reported


def load_frame():
    """Read the v3 frame with the recoded nominal fields kept as strings (a field whose levels are all digits would
    otherwise come back numeric and be treated as continuous)."""
    return pd.read_csv(OUT / "kabco_person_level_v3.csv", low_memory=False, dtype={**{c: str for c in NOMINAL}, "person_type": str})


def recode(frame, rep):
    for c, unk in NOMINAL.items():
        v = pd.to_numeric(frame[c], errors="coerce").round()
        lev = v.map(lambda x: unk.get(int(x), str(int(x))) if pd.notna(x) else "UNK")
        cnt = lev.value_counts(); lev = lev.where(lev.isin(cnt[cnt >= 100].index), "OTHER")
        frame[c] = lev.astype(str); rep[f"levels_{c}"] = frame[c].value_counts().to_dict()
    cid = frame["source"].astype(str) + "_" + frame["YEAR"].astype(str) + "_" + frame["ST_CASE"].astype(str)
    frame["persons_in_crash"] = np.log(cid.map(cid.value_counts()).astype(float))
    pt = pd.to_numeric(frame["PER_TYP"], errors="coerce")
    frame["person_type"] = np.select([pt == 1, pt == 2, pt.isin([5, 6, 7, 8, 10])], ["driver", "passenger", "nonoccupant"], "other")
    rep["person_type"] = frame["person_type"].value_counts().to_dict()
    rep["persons_in_crash_quantiles"] = np.exp(frame["persons_in_crash"]).quantile([0.5, 0.9, 0.99]).tolist()
    for c, s in SENTINEL.items():
        v = pd.to_numeric(frame[c], errors="coerce"); miss = v.isna() | (v >= s)
        frame[c] = v.where(~miss, float(v[~miss].median())); frame[c + "_UNK"] = miss.astype(float)
        rep[f"unknown_{c}"] = int(miss.sum())
    return frame


def main(pathways=("pavement", "infrastructure"), rebuild=True):
    t0 = time.time()
    if not rebuild:
        frame = load_frame()
        rep = json.loads((OUT / "frame_v3_report.json").read_text()) if (OUT / "frame_v3_report.json").exists() else {}
        return refit(frame, rep, pathways, t0)
    frame = pd.read_csv(ROOT / "outputs" / "kabco_person_level_clean.csv", low_memory=False)
    rep = dict(n_before=int(len(frame)), by_origin_before=frame.record_origin.value_counts().to_dict())
    drop = np.zeros(len(frame), dtype=bool)
    for y in (2021, 2022, 2023):
        z = zipfile.ZipFile(DATA / f"CRSS{y}NationalCSV.zip")
        nm = [n for n in z.namelist() if n.lower().endswith("person.csv")][0]
        p = pd.read_csv(z.open(nm), usecols=["CASENUM", "INJ_SEV"], encoding="latin1", low_memory=False)
        fat = set(p.loc[p.INJ_SEV == 4, "CASENUM"].astype(int))
        m = (frame.record_origin == "crss").values & (frame.YEAR == y).values & frame.ST_CASE.isin(fat).values
        rep[f"crss_fatal_crashes_{y}"] = len(fat); rep[f"dropped_{y}"] = int(m.sum()); drop |= m
    frame = frame.loc[~drop].reset_index(drop=True)
    frame = recode(frame, rep)
    rep.update(n_after=int(len(frame)), dropped=int(drop.sum()), by_origin_after=frame.record_origin.value_counts().to_dict(),
               kabco_after=frame.kabco_ordinal.value_counts().sort_index().to_dict())
    print(json.dumps(rep, indent=1, default=int), flush=True)
    frame.to_csv(OUT / "kabco_person_level_v3.csv", index=False)
    return refit(frame, rep, pathways, t0)


def refit(frame, rep, pathways, t0):
    for pathway in pathways:
        dropc = V3_DROP[pathway]
        Xv, names, y = RD.build_design(frame, pathway, drop=dropc)
        print(f"{pathway}: n={len(y)} p={len(names)} [{time.time() - t0:.0f}s]", flush=True)
        thr, A, B, S, eta = RD.fit_all_cuts(Xv, y)
        np.savez_compressed(OUT / f"firth_fit_{pathway}.npz", thr=np.array(thr), alpha=A, beta=B, se=S, names=np.array(names))
        flic = []
        for i, t in enumerate(thr):
            obs = float(np.mean(y > t)); fit = float(np.mean(lg(eta[i])))
            off = eta[i] - A[i]
            a_ml = brentq(lambda a: float(np.sum((y > t) - lg(off + a))), A[i] - 5, A[i] + 5, xtol=1e-12)
            flic.append(dict(cut=int(t), observed=obs, mean_fitted_firth=fit, intercept_firth=float(A[i]),
                             intercept_flic=float(a_ml), intercept_shift=float(a_ml - A[i])))
        rep[f"flic_{pathway}"] = flic
        print(pd.DataFrame(flic).to_string(index=False), flush=True)
    (OUT / "frame_v3_report.json").write_text(json.dumps(rep, indent=1, default=int))
    print(f"DONE [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    pw = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("pavement", "infrastructure")
    main(pw, rebuild=not (len(sys.argv) > 2 and sys.argv[2] == "--refit-only"))
