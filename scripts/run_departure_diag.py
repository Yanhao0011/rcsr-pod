#!/usr/bin/env python3
"""RCSR-POD application diagnostic, design v2, DEPARTURE statistic, marginal bound.

Fits the four Firth threshold logits on the full frame (design from run_task_a with the
nested-redundant HARM_EV_12.0 removed) and computes for every feature:
  * the previous plain-PSC gap (for the cautionary comparison),
  * the departure profile D_k^(t), Dmax = max_t |D| with delta-method lower bound,
    the between-cut gap Dgap, the non-fatal-ladder version (cuts 0-2 only),
  * the per-unit (discrete-change) departure companion,
  * frame-term flags (MISSING levels that coincide with the CRSS share).
Writes departure_diag_{pathway}.csv, firth_fit_{pathway}.npz, and a summary JSON.

Usage: python run_departure_diag.py --pathway infrastructure [--frame ...] [--outdir ...]
"""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_task_a as TA
from po_diagnostic_core import firth_logit
from departure_core import departure_profile, discrete_change_profile, summarize, lcb_max

DELTA_GRID = [0.005, 0.01, 0.02, 0.03, 0.05]


def build_design(frame, pathway, drop=()):
    feats = TA.PATHWAY_FEATURES[pathway] + TA.CONTROL_FEATURES
    enc, names = TA.encode_features(frame, feats)
    keep = frame.dropna(subset=["kabco_ordinal"]).index
    y = pd.to_numeric(frame["kabco_ordinal"], errors="coerce").astype("Int64").dropna().astype(int).to_numpy()
    enc = enc.loc[enc.index.isin(keep)].reset_index(drop=True)
    for d in drop:
        if d in enc.columns:
            enc = enc.drop(columns=[d]); names = [c for c in names if c != d]
    Xv = TA.scale_no_mean(enc.values).astype(np.float64)
    return Xv, names, y


def fit_all_cuts(Xv, y, ridge=1e-6):
    n, p = Xv.shape
    Xm = np.column_stack([np.ones(n), Xv])
    thr = [t for t in sorted(np.unique(y))[:-1] if 0 < (y > t).sum() < n]
    B = np.zeros((len(thr), p)); S = np.zeros((len(thr), p)); A = np.zeros(len(thr)); eta = []
    for i, t in enumerate(thr):
        t0 = time.time()
        b, se, ok = firth_logit(Xm, (y > t).astype(int), ridge=ridge)
        B[i], S[i], A[i] = b[1:], se[1:], b[0]; eta.append(Xm @ b)
        print(f"    firth t>{t}: n1={int((y > t).sum())} converged={ok} [{time.time() - t0:.0f}s]", flush=True)
    return thr, A, B, S, eta


def feature_rows(eta, B, S, Xv, names, pathway, delta, alpha=0.05):
    T = B.shape[0]; nf = [i for i in range(T) if i <= 2]
    rows = []
    for j, f in enumerate(names):
        xj = Xv[:, j]; prev = float(np.mean(xj != 0))
        prof = departure_profile(eta, B[:, j], S[:, j], xj)
        s = summarize(prof, alpha)
        pnf = departure_profile(eta, B[:, j], S[:, j], xj, cuts=nf)
        Dnf, tnf, Dnf_lcb = lcb_max(pnf["D"], pnf["se_D"], alpha)
        dce = discrete_change_profile(eta, B[:, j], S[:, j], xj)
        DCE, tdce, DCE_lcb = lcb_max(dce["D"], dce["se_D"], alpha)
        rows.append(dict(feature=f, label=TA.LABELS.get(f, f), frame_term=TA.is_frame_term(f, prev),
                         kind=dce["kind"], prevalence=prev, bbar=s["bbar"],
                         coef_gap=float(B[:, j].max() - B[:, j].min()),
                         psc_gap=s["psc_gap"], psc_lcb=s["psc_lcb"], credible_psc=s["psc_lcb"] > delta,
                         **{f"D{t}": prof["D"][t] for t in range(T)}, **{f"seD{t}": prof["se_D"][t] for t in range(T)},
                         Dmax=s["Dmax"], Dmax_cut=s["Dmax_cut"], Dmax_lcb=s["Dmax_lcb"], credible_Dmax=s["Dmax_lcb"] > delta,
                         Dgap=s["Dgap"], Dgap_lcb=s["Dgap_lcb"], credible_Dgap=s["Dgap_lcb"] > delta,
                         Dmax_nf=Dnf, Dmax_nf_cut=nf[tnf], Dmax_nf_lcb=Dnf_lcb, credible_nf=Dnf_lcb > delta,
                         DCEmax=DCE, DCEmax_cut=tdce, DCEmax_lcb=DCE_lcb, credible_DCE=DCE_lcb > delta))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathway", default="infrastructure")
    ap.add_argument("--frame", default=str(HERE.parent / "outputs" / "kabco_person_level_clean.csv"))
    ap.add_argument("--outdir", default=str(HERE.parent / "outputs" / "departure"))
    ap.add_argument("--drop", default="", help="extra encoded dummies to drop (beyond run_task_a.EXCLUDE)")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--nrows", type=int, default=0)
    a = ap.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    drop = [d for d in a.drop.split(",") if d]
    t0 = time.time()
    frame = pd.read_csv(a.frame, low_memory=False)
    if a.nrows:
        frame = frame.sample(n=a.nrows, random_state=1).reset_index(drop=True)
    Xv, names, y = build_design(frame, a.pathway, drop)
    print(f"{a.pathway}: n={len(y)} p={len(names)} excluded={TA.EXCLUDE + drop} [{time.time() - t0:.0f}s]", flush=True)
    thr, A, B, S, eta = fit_all_cuts(Xv, y)
    np.savez_compressed(outdir / f"firth_fit_{a.pathway}.npz", thr=np.array(thr), alpha=A, beta=B, se=S, names=np.array(names))
    df = feature_rows(eta, B, S, Xv, names, a.pathway, a.delta).sort_values("Dmax", ascending=False)
    df.to_csv(outdir / f"departure_diag_{a.pathway}.csv", index=False)
    sub = df[~df.frame_term]
    summ = dict(pathway=a.pathway, n=int(len(y)), p=int(len(names)), excluded=TA.EXCLUDE + drop, delta=a.delta,
                frame_terms=df.loc[df.frame_term, "feature"].tolist(),
                credible_psc=int(df.credible_psc.sum()),
                credible_Dmax_all=int(df.credible_Dmax.sum()), credible_Dmax_substantive=int(sub.credible_Dmax.sum()),
                credible_Dgap_all=int(df.credible_Dgap.sum()),
                credible_nf_substantive=int(sub.credible_nf.sum()), credible_DCE_substantive=int(sub.credible_DCE.sum()),
                sweep_Dmax_substantive={str(d): int((sub.Dmax_lcb > d).sum()) for d in DELTA_GRID},
                sweep_Dmax_all={str(d): int((df.Dmax_lcb > d).sum()) for d in DELTA_GRID},
                sweep_psc={str(d): int((df.psc_lcb > d).sum()) for d in DELTA_GRID},
                set_Dmax_substantive=sub.loc[sub.credible_Dmax, "feature"].tolist(),
                set_Dmax_frame=df.loc[df.frame_term & df.credible_Dmax, "feature"].tolist(),
                set_DCE_substantive=sub.loc[sub.credible_DCE, "feature"].tolist(),
                set_nf_substantive=sub.loc[sub.credible_nf, "feature"].tolist(),
                set_psc=df.loc[df.credible_psc, "feature"].tolist(), seconds=time.time() - t0)
    (outdir / f"departure_diag_{a.pathway}_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    show = df[df.credible_Dmax | df.credible_DCE | df.credible_psc]
    cols = ["label", "frame_term", "prevalence", "coef_gap", "psc_gap", "Dmax", "Dmax_cut", "Dmax_lcb", "Dmax_nf_lcb", "DCEmax", "DCEmax_lcb", "credible_psc", "credible_Dmax", "credible_nf", "credible_DCE"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(show[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"DONE [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
