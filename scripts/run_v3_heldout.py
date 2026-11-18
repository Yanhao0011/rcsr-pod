#!/usr/bin/env python3
"""Held-out predictive payoff of parsimony (RCSR-POD v3).

Crash-grouped K-fold cross-validation (all persons of a crash in the same fold) of four cumulative-logit
specifications fitted with the penalized partial-PO likelihood of run_partial_po_fit.PartialPO:
    PO        every feature common across cuts
    RCSR-POD  threshold-specific for the v3 simultaneous credible set plus the frame terms
    Brant     threshold-specific for every feature with a marginal Brant/Wald p < 0.05 (full-frame Firth fits)
    GOL       every feature threshold-specific (generalized ordered logit)
Scores on the held-out fold: mean log loss, multi-category Brier score, ranked probability score (RPS),
share of records with a non-positive fitted category probability. The sets are selected once on the full
frame (disclosed; this favours the larger specifications if anything).

Usage: python run_v3_heldout.py --summary outputs/v3/heldout_specs.json --folds 5
"""
from __future__ import annotations
import argparse, sys, json, time
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_task_a as TA
import run_departure_diag as RD
from run_v3_frame import V3_DROP, SOURCE_TERM, load_frame, frame_units
from run_partial_po_fit import PartialPO, T

ROOT = Path(os.environ.get("RCPO_ROOT", HERE.parent))


def brant_set(pathway, names):
    z = np.load(ROOT / "outputs" / "v3" / f"firth_fit_{pathway}.npz", allow_pickle=True)
    B, S = z["beta"], z["se"]; out = []
    for j, f in enumerate(names):
        chi2 = float(np.sum((B[1:, j] - B[0, j]) ** 2 / (S[1:, j] ** 2 + S[0, j] ** 2)))
        if 1 - stats.chi2.cdf(chi2, B.shape[0] - 1) < 0.05: out.append(f)
    return out


def predict(model, th, X, idx1, idx2):
    m = PartialPO(X[:, idx1] if idx1 else None, X[:, idx2] if idx2 else None, np.zeros(len(X), dtype=int), model.lam)
    F = [np.clip(1 / (1 + np.exp(-e)), 0, 1) for e in m.etas(th)]
    return np.column_stack([1 - F[0]] + [F[t] - F[t + 1] for t in range(T - 1)] + [F[T - 1]])


def evaluate(P, y):
    n = len(y); Y = np.zeros_like(P); Y[np.arange(n), y] = 1
    py = P[np.arange(n), y]
    cP, cY = np.cumsum(P, 1)[:, :-1], np.cumsum(Y, 1)[:, :-1]
    Pt = np.clip(P, 1e-6, None); Pt = Pt / Pt.sum(1, keepdims=True)          # truncated and renormalized probabilities
    return dict(logloss=float(-np.mean(np.log(np.maximum(py, 1e-10)))), logloss_trunc=float(-np.mean(np.log(Pt[np.arange(n), y]))),
                brier=float(np.mean(np.sum((P - Y) ** 2, 1))),
                rps=float(np.mean(np.sum((cP - cY) ** 2, 1)) / (P.shape[1] - 1)), neg_share=float(np.mean((P <= 0).any(1))))


def fit_spec(X, y, idx2, lam, maxiter, th_po=None):
    p = X.shape[1]; idx1 = [j for j in range(p) if j not in idx2]
    alpha0 = np.array([np.log(np.mean(y > t) / (1 - np.mean(y > t))) for t in range(T)])
    m = PartialPO(X[:, idx1] if idx1 else None, X[:, idx2] if idx2 else None, y, lam)
    if th_po is None:
        th0 = np.concatenate([alpha0, np.zeros(len(idx1)), np.zeros(T * len(idx2))])
    else:
        a, b = th_po[:T], th_po[T:]
        th0 = np.concatenate([a, b[idx1], np.tile(b[idx2], T)])
    r = m.fit(th0, maxiter)
    return m, r, idx1


OUT_BIC = {}


def load_bic(pathway, names):
    import run_task_a as TA
    f = ROOT / "outputs" / "v3" / f"brant_v3_covariates_{pathway}.csv"
    if not f.exists():
        return
    cv = pd.read_csv(f); feats = TA.PATHWAY_FEATURES[pathway] + TA.CONTROL_FEATURES
    from run_v3_application import parent_of
    keep = []
    for u in cv.loc[cv.bic, "unit"]:
        par = u.replace(" [group]", "")
        keep += [j for j, nm in enumerate(names) if (nm == u) or (u.endswith("[group]") and parent_of(nm, feats) == par and nm != SOURCE_TERM[pathway])]
    OUT_BIC[pathway] = keep


def run_pathway(pathway, frame, sets, folds, lam, maxiter, seed, outdir):
    drop = V3_DROP[pathway]
    X, names, y = RD.build_design(frame, pathway, drop=drop)
    keep = frame.dropna(subset=["kabco_ordinal"]).index
    crash = (frame["source"].astype(str) + "_" + frame["YEAR"].astype(str) + "_" + frame["ST_CASE"].astype(str)).to_numpy()[frame.index.isin(keep)]
    load_bic(pathway, names)
    codes, uniq = pd.factorize(crash); fold_of = np.random.default_rng(seed).integers(0, folds, len(uniq))[codes]
    specs = {"PO": [], "PO + frame terms": [names.index(f) for f in frame_units(pathway)]}
    for sname, feats in sets[pathway].items():
        specs[sname] = [names.index(f) for f in feats if f in names]
    bic = OUT_BIC.get(pathway)
    if bic is not None:
        specs["BIC (covariate Wald)"] = sorted(set(bic) | {names.index(f) for f in frame_units(pathway)})
    specs.update({"Marginal Brant": [names.index(f) for f in brant_set(pathway, names)], "GOL": list(range(len(names)))})
    print(f"\n=== {pathway} n={len(y)} p={len(names)}; threshold-specific counts: " +
          ", ".join(f"{k}={len(v)}" for k, v in specs.items()), flush=True)
    rows = []
    for k in range(folds):
        tr, te = fold_of != k, fold_of == k; th_po = None
        for sname, idx2 in specs.items():
            t0 = time.time()
            m, r, idx1 = fit_spec(X[tr], y[tr], idx2, lam, maxiter, th_po)
            if sname == "PO": th_po = r["theta"]
            P = predict(m, r["theta"], X[te], idx1, idx2)
            ev = evaluate(P, y[te])
            rows.append(dict(pathway=pathway, fold=k, spec=sname, k_params=r["k"], n_test=int(te.sum()), **ev,
                             train_neg=r["n_neg"], status=r["status"], seconds=time.time() - t0))
            print(f"    fold {k} {sname:8s} k={r['k']:4d} logloss={ev['logloss']:.5f} brier={ev['brier']:.5f} rps={ev['rps']:.5f} "
                  f"neg={ev['neg_share']:.2e} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"heldout_v3_{pathway}.csv", index=False)
    return df, {k: [names[j] for j in v] for k, v in specs.items() if k not in ("PO", "GOL")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--pathways", default="pavement,infrastructure")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "v3"))
    ap.add_argument("--nrows", type=int, default=0, help="smoke test: random subsample")
    a = ap.parse_args()
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    sets = json.loads(Path(a.summary).read_text())     # {pathway: {spec name: [threshold-specific features]}}
    frame = load_frame()
    if a.nrows: frame = frame.sample(n=a.nrows, random_state=1).reset_index(drop=True)
    allres = []; used = {}
    for pw in a.pathways.split(","):
        df, u = run_pathway(pw.strip(), frame, sets, a.folds, a.lam, a.maxiter, a.seed, outdir); allres.append(df); used[pw.strip()] = u
    df = pd.concat(allres)
    summ = df.groupby(["pathway", "spec"]).agg(k_params=("k_params", "mean"), logloss=("logloss", "mean"), brier=("brier", "mean"),
                                               rps=("rps", "mean"), neg_share=("neg_share", "mean"), logloss_trunc=("logloss_trunc", "mean")).reset_index()
    base = summ[summ.spec == "PO"].set_index("pathway"); base_s = summ[summ.spec == "PO + frame terms"].set_index("pathway")
    br = summ[summ.spec == "Marginal Brant"].set_index("pathway")
    summ["share_of_brant_gain_vs_PO_frame"] = (summ.pathway.map(base_s.logloss) - summ.logloss) / (summ.pathway.map(base_s.logloss) - summ.pathway.map(br.logloss))
    for c in ["logloss", "brier", "rps"]:
        summ[f"d_{c}_vs_PO"] = summ[c] - summ.pathway.map(base[c])
    summ.to_csv(outdir / "heldout_v3_summary.csv", index=False)
    (outdir / "heldout_v3_sets.json").write_text(json.dumps(used, indent=1))
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
