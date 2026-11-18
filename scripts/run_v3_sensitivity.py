#!/usr/bin/env python3
"""RCSR-POD v3 sensitivity analyses on the de-duplicated frame (invariant departure throughout).

  strata    the diagnostic refitted within each source frame: FARS (four cuts) and CRSS (three non-fatal cuts);
            invariant departure and simultaneous bound from the joint crash-cluster sandwich.
  weighted  design-weighted version: threshold logits refitted by weighted Firth likelihood (CRSS design weights,
            FARS weight 1, normalised to mean 1); weighted departure (weighted mean rate preserved, weighted mean
            absolute change); simultaneous bound from the weighted joint crash-cluster sandwich.
  refit     refit counterpart of the departure: penalized PO and PPO(k) cumulative logits (exact five-category
            likelihood of run_partial_po_fit), R_k = max_t mean_i |P_PPO(k)(y_i > t) - P_PO(y_i > t)|, for the credible
            set, the source indicator and the three largest non-credible departures.

Usage: python run_v3_sensitivity.py --mode strata|weighted|refit --summary outputs/v3/v3_summary_..._B1000.json
"""
from __future__ import annotations
import argparse, sys, json, time
import os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_task_a as TA
import run_departure_diag as RD
import run_partial_po_fit as PP
import run_v3_application as APP
import rcsr_v3 as V_
from po_diagnostic_core import _xtwx, _leverage
from run_v3_frame import V3_DROP, SOURCE_TERM, load_frame, frame_units, is_frame
from run_simboot import chol_jitter
from scipy import sparse

ROOT = Path(os.environ.get("RCPO_ROOT", HERE.parent)); OUT = ROOT / "outputs" / "v3"
DELTA, GRID = APP.DELTA, APP.GRID


def lg(v): return 0.5 * (1.0 + np.tanh(0.5 * v))


def design(frame, pathway):
    Xv, names, y = RD.build_design(frame, pathway, drop=V3_DROP[pathway])
    keep = np.std(Xv, axis=0) > 0                       # columns constant within a stratum are dropped
    crash = (frame["source"].astype(str) + "_" + frame["YEAR"].astype(str) + "_" + frame["ST_CASE"].astype(str)).to_numpy()
    codes, uniq = pd.factorize(crash[frame.kabco_ordinal.notna().to_numpy()])
    return Xv[:, keep], [n for n, k in zip(names, keep) if k], y, codes, len(uniq)


def firth_w(Xm, by, w, ridge=1e-6, iters=100, tol=1e-6):
    """Weighted Firth logit: modified score U = X'[w(y - p) + h(1/2 - p)], h the weighted hat diagonal."""
    q = Xm.shape[1]; b = np.zeros(q); b[0] = np.log(np.average(by, weights=w) / (1 - np.average(by, weights=w)))
    for _ in range(iters):
        p = lg(Xm @ b); ww = w * p * (1 - p)
        cov = np.linalg.inv(_xtwx(Xm, ww) + ridge * np.eye(q))
        h = _leverage(Xm, ww, cov)
        step = cov @ (Xm.T @ (w * (by - p) + h * (0.5 - p)))
        m = np.abs(step).max()
        if m > 2.0: step *= 2.0 / m
        b = b + step
        if m < tol: break
    p = lg(Xm @ b); ww = w * p * (1 - p)
    cov = np.linalg.inv(_xtwx(Xm, ww) + ridge * np.eye(q))
    return b, cov, p


def joint_cluster(Xm, ys, ps, covs, codes, n_cl, w=None):
    q = Xm.shape[1]; T = len(ys)
    M = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(n_cl, len(codes)))
    G = [np.asarray(M @ (Xm * ((1.0 if w is None else w) * (ys[i] - ps[i]))[:, None])) for i in range(T)]
    V = np.zeros((T * q, T * q))
    for i in range(T):
        for j in range(i, T):
            blk = covs[i] @ (G[i].T @ G[j]) @ covs[j]
            V[i * q:(i + 1) * q, j * q:(j + 1) * q] = blk; V[j * q:(j + 1) * q, i * q:(i + 1) * q] = blk.T
    return 0.5 * (V + V.T)


def diagnose(tag, Xv, names, y, codes, n_cl, pathway, Bd, block, rng, w=None, fits=None):
    """Point profiles for every unit + simultaneous bound for candidates. Returns a DataFrame."""
    n, p = Xv.shape; Xm = np.column_stack([np.ones(n), Xv]); q = p + 1
    thr = [t for t in sorted(np.unique(y))[:-1] if 0 < (y > t).sum() < n]; T = len(thr)
    A = np.zeros(T); B = np.zeros((T, p)); S = np.zeros((T, p)); covs = []; ps = []; ys = []
    t0 = time.time()
    for i, t in enumerate(thr):
        by = (y > t).astype(float)
        if w is None:
            b, se, ok = RD.firth_logit(Xm, by.astype(int), ridge=1e-6)
            pr = lg(Xm @ b); cov = np.linalg.inv(_xtwx(Xm, pr * (1 - pr)) + 1e-6 * np.eye(q))
        else:
            b, cov, pr = firth_w(Xm, by, w)
        A[i], B[i], S[i] = b[0], b[1:], np.sqrt(np.diag(cov))[1:]; covs.append(cov); ps.append(pr); ys.append(by)
        print(f"    [{tag}] fit t>{t} ({time.time() - t0:.0f}s)", flush=True)
    V = joint_cluster(Xm, ys, ps, covs, codes, n_cl, w)
    eta = [Xm @ np.concatenate([[A[i]], B[i]]) for i in range(T)]
    feats = TA.PATHWAY_FEATURES[pathway] + TA.CONTROL_FEATURES
    U = APP.units(names, feats, [float(np.mean(Xv[:, j] != 0)) for j in range(p)])
    wt = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    Dh, W = {}, {}
    for u in U:
        cols = u["cols"]; c1 = np.array(cols) + 1
        Wk = V_.gls_weights(np.array([covs[i][np.ix_(c1, c1)] for i in range(T)])); bbar = V_.pooled(Wk, B[:, cols]); XG = V_.as_group(Xv[:, cols])
        Dh[u["unit"]] = np.array([(V_._dinv_core(eta[i], XG, np.ascontiguousarray(B[i, cols] - bbar)) if wt is None else
                                   V_._dinv_core_w(eta[i], XG, np.ascontiguousarray(B[i, cols] - bbar), wt)) for i in range(T)])
        W[u["unit"]] = Wk
    cand = {u["unit"]: u["cols"] for u in U if Dh[u["unit"]].max() > 0.5 * GRID[0] or is_frame(u["unit"], pathway)}
    L = chol_jitter(V); theta = np.concatenate([np.concatenate([[A[i]], B[i]]) for i in range(T)])
    Db = {k: np.empty((T, Bd)) for k in cand}; XG = {k: V_.as_group(Xv[:, c]) for k, c in cand.items()}
    done = 0
    while done < Bd:
        bs = min(block, Bd - done); dr = theta[:, None] + L @ rng.standard_normal((T * q, bs))
        draws = [dr[i * q:(i + 1) * q] for i in range(T)]; etaT = [np.ascontiguousarray((Xm @ d).T) for d in draws]
        for k, cols in cand.items():
            bj = np.stack([draws[i][np.array(cols) + 1] for i in range(T)]); bbar = V_.pooled(W[k], bj)
            for i in range(T):
                dB = np.ascontiguousarray((bj[i] - bbar).T)
                Db[k][i, done:done + bs] = V_.dinv_draws(etaT[i], XG[k], dB) if wt is None else V_.dinv_draws_w(etaT[i], XG[k], dB, wt)
        done += bs
    rows = []
    for u in U:
        k = u["unit"]; D = Dh[k]
        r = dict(analysis=tag, pathway=pathway, unit=k, kind=u["kind"], frame_term=is_frame(k, pathway), n=int(n), cuts=T,
                 Dmax=float(D.max()), Dmax_cut=thr[int(np.argmax(D))])
        if k in cand:
            m = V_.maxt_onesided(D, Db[k], DELTA, kappa=V_.kappa_n(n)); r.update(lcb=m["lcb"], lcb_marg=m["lcb_marg"], c=m["c"], p=m["p"])
        else:
            r.update(lcb=np.nan, lcb_marg=np.nan, c=np.nan, p=1.0)
        rows.append(r)
    df = pd.DataFrame(rows)
    for kind in ["feature", "group"]:
        msk = (df.kind == kind).values; h, b_ = V_.multitest(df.loc[msk, "p"].values); df.loc[msk, "holm"] = h; df.loc[msk, "bh"] = b_
    return df


def primary(df, pathway):
    """Coding-invariant family: single-column covariates plus categorical groups (source indicator separate)."""
    feats = TA.PATHWAY_FEATURES[pathway] + TA.CONTROL_FEATURES
    grouped = {u.replace(" [group]", "") for u in df.loc[df.kind == "group", "unit"]}
    keep = (df.kind == "group").values | np.array([(APP.parent_of(u, feats) not in grouped) for u in df.unit])
    out = df[keep & ~df.frame_term.values].copy()
    h, b = V_.multitest(out.p.values); out["holm"] = h; out["bh"] = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["strata", "weighted", "refit", "crashsize", "nmin"])
    ap.add_argument("--nmin", type=int, default=500)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--pathways", default="pavement,infrastructure")
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--block", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--maxiter", type=int, default=3000)
    a = ap.parse_args()
    frame = load_frame()
    js = json.loads(Path(a.summary).read_text())          # v3_primary_summary.json: {pathway: {credible: [...], ...}}
    rng = np.random.default_rng(a.seed); res = []; summ = {}
    for pw in [s.strip() for s in a.pathways.split(",")]:
        print(f"\n=== {a.mode} / {pw} ===", flush=True)
        main_csv = pd.read_csv(OUT / f"v3_primary_{pw}.csv")
        if a.mode == "strata":
            for src in ["fars", "crss"]:
                sub = frame[frame.record_origin == src].reset_index(drop=True)
                Xv, names, y, codes, n_cl = design(sub, pw)
                df = diagnose(f"strata_{src}", Xv, names, y, codes, n_cl, pw, a.B, a.block, rng); res.append(df)
                f = primary(df, pw)
                summ[f"{pw}_{src}"] = dict(n=int(len(y)), credible=f.loc[f.lcb > DELTA, "unit"].tolist(),
                                          main_set_credible={k: bool(k in set(f.loc[f.lcb > DELTA, "unit"])) for k in js[pw]["credible"]},
                                          main_set_Dmax={k: float(f.set_index("unit").Dmax.get(k, np.nan)) for k in js[pw]["credible"]})
                print(json.dumps(summ[f"{pw}_{src}"], indent=1), flush=True)
        elif a.mode == "weighted":
            Xv, names, y, codes, n_cl = design(frame, pw)
            w = pd.to_numeric(frame.loc[frame.kabco_ordinal.notna(), "WEIGHT"], errors="coerce").fillna(1.0).clip(lower=1e-6).to_numpy()
            w = w / w.mean()
            df = diagnose("weighted", Xv, names, y, codes, n_cl, pw, a.B, a.block, rng, w=w); res.append(df)
            f = primary(df, pw)
            summ[pw] = dict(credible=f.loc[f.lcb > DELTA, "unit"].tolist(), holm=f.loc[f.holm.astype(bool), "unit"].tolist(),
                            main_set_credible={k: bool(k in set(f.loc[f.lcb > DELTA, "unit"])) for k in js[pw]["credible"]},
                            sweep={str(d): int((f.lcb > d).sum()) for d in GRID})
            print(json.dumps(summ[pw], indent=1), flush=True)
        elif a.mode in ("crashsize", "nmin"):
            fr = frame.copy(); extra = []
            if a.mode == "crashsize":
                # persons per crash and person type (driver / passenger / non-occupant), the selection-related controls
                cid = fr["source"].astype(str) + "_" + fr["YEAR"].astype(str) + "_" + fr["ST_CASE"].astype(str)
                fr["persons_in_crash"] = np.log(cid.map(cid.value_counts()).astype(float))
                pt = pd.to_numeric(fr["PER_TYP"], errors="coerce")
                fr["person_type"] = np.select([pt == 1, pt == 2, pt.isin([5, 6, 7, 8, 10])], ["driver", "passenger", "nonoccupant"], "other")
                extra = ["persons_in_crash", "person_type"]
                for c in extra:
                    if c not in TA.CONTROL_FEATURES: TA.CONTROL_FEATURES.append(c)
            else:
                for c in TA.PATHWAY_FEATURES[pw] + TA.CONTROL_FEATURES:
                    if c in fr.columns and fr[c].dtype == object:
                        cnt = fr[c].value_counts(); rare = cnt[cnt < a.nmin].index
                        fr[c] = fr[c].where(~fr[c].isin(rare) | (fr[c] == "MISSING"), "OTHER")
            Xv, names, y, codes, n_cl = design(fr, pw)
            df = diagnose(f"{a.mode}", Xv, names, y, codes, n_cl, pw, a.B, a.block, rng); res.append(df)
            for c in extra:
                TA.CONTROL_FEATURES.remove(c)
            f = primary(df, pw)
            summ[pw] = dict(p=int(Xv.shape[1]), credible=f.loc[f.lcb > DELTA, "unit"].tolist(), holm=f.loc[f.holm.astype(bool), "unit"].tolist(),
                            main_set={k: dict(Dmax=float(f.set_index("unit").Dmax.get(k, np.nan)), lcb=float(f.set_index("unit").lcb.get(k, np.nan)))
                                      for k in js[pw]["credible"]},
                            top=f.sort_values("Dmax", ascending=False)[["unit", "Dmax", "lcb"]].head(6).to_dict("records"))
            print(json.dumps(summ[pw], indent=1, default=float), flush=True)
        else:
            Xv, names, y, codes, n_cl = design(frame, pw); n, p = Xv.shape; T = PP.T
            feat = main_csv.set_index("unit")
            nonc = feat[(feat.cl_all_lcb.fillna(-1) <= DELTA) & ~feat.frame_term]
            targets = list(dict.fromkeys(js[pw]["credible"] + [SOURCE_TERM[pw]] + nonc.sort_values("Dmax", ascending=False).index[:3].tolist()))
            alpha0 = np.array([np.log(np.mean(y > t) / (1 - np.mean(y > t))) for t in range(T)])
            po = PP.PartialPO(Xv, None, y, 1.0); r_po = po.fit(np.concatenate([alpha0, np.zeros(p)]), a.maxiter)
            a_po, b_po, _ = po.unpack(r_po["theta"]); _, F_po = po.probs(r_po["theta"])
            print(f"    PO nit={r_po['nit']} ({r_po['seconds']:.0f}s) {r_po['status'][:40]}", flush=True)
            # departure with the penalized PO-model coefficient as the reference instead of the GLS-pooled Firth slopes
            z = np.load(OUT / f"firth_fit_{pw}.npz", allow_pickle=True); Af, Bf = z["alpha"], z["beta"]
            Xm_ = np.column_stack([np.ones(n), Xv]); eta_f = [Xm_ @ np.concatenate([[Af[i]], Bf[i]]) for i in range(T)]
            fl = TA.PATHWAY_FEATURES[pw] + TA.CONTROL_FEATURES; poref = []
            for f in feat.index:
                ks = ([j for j, nm in enumerate(names) if APP.parent_of(nm, fl) == f.replace(" [group]", "") and nm != SOURCE_TERM[pw]]
                      if f.endswith("[group]") else ([names.index(f)] if f in names else []))
                if not ks: continue
                XG = V_.as_group(Xv[:, ks])
                Dp = [V_._dinv_core(eta_f[i], XG, np.ascontiguousarray(Bf[i, ks] - b_po[ks])) for i in range(T)]
                poref.append(dict(pathway=pw, unit=f, Dmax_gls=float(feat.loc[f, "Dmax"]), Dmax_poref=float(max(Dp)), cut_poref=int(np.argmax(Dp))))
            pd.DataFrame(poref).to_csv(OUT / f"sens_poref_{pw}.csv", index=False)
            pr_ = pd.DataFrame(poref); summ[pw + "_poref"] = dict(corr=float(np.corrcoef(pr_.Dmax_gls, pr_.Dmax_poref)[0, 1]),
                                                                 above_delta_gls=pr_.loc[pr_.Dmax_gls > DELTA, "unit"].tolist(),
                                                                 above_delta_poref=pr_.loc[pr_.Dmax_poref > DELTA, "unit"].tolist())
            print(json.dumps(summ[pw + "_poref"], indent=1), flush=True)
            rows = []
            for f in targets:
                fl = TA.PATHWAY_FEATURES[pw] + TA.CONTROL_FEATURES
                ks = ([j for j, nm in enumerate(names) if APP.parent_of(nm, fl) == f.replace(" [group]", "") and nm != SOURCE_TERM[pw]]
                      if f.endswith("[group]") else [names.index(f)])
                idx1 = [j for j in range(p) if j not in ks]
                m = PP.PartialPO(Xv[:, idx1], Xv[:, ks], y, 1.0)
                r = m.fit(np.concatenate([a_po, b_po[idx1], np.tile(b_po[ks], T)]), a.maxiter); _, F_k = m.probs(r["theta"])
                Rabs = [float(np.mean(np.abs(F_k[t] - F_po[t]))) for t in range(T)]; Rsgn = [float(np.mean(F_k[t] - F_po[t])) for t in range(T)]
                rows.append(dict(analysis="refit", pathway=pw, unit=f, credible=f in js[pw]["credible"], frame_term=is_frame(f, pw),
                                 R=max(Rabs), R_cut=int(np.argmax(Rabs)), Dmax=float(feat.loc[f, "Dmax"]), lcb=float(feat.loc[f, "cl_all_lcb"]),
                                 max_abs_signed=float(np.max(np.abs(Rsgn))), dLL=float(r["ll"] - r_po["ll"]), nit=int(r["nit"]),
                                 **{f"R{t}": Rabs[t] for t in range(T)}))
                print(f"    {f:26s} R={max(Rabs):.4f} (cut {int(np.argmax(Rabs))}) Dinv={feat.loc[f, 'Dmax']:.4f} signed max={np.max(np.abs(Rsgn)):.1e} nit={r['nit']} ({r['seconds']:.0f}s)", flush=True)
            df = pd.DataFrame(rows); res.append(df)
            summ[pw] = df[["unit", "credible", "R", "Dmax"]].to_dict("records")
    pd.concat(res).to_csv(OUT / f"sens_{a.mode}.csv", index=False)
    (OUT / f"sens_{a.mode}_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print("DONE", a.mode, flush=True)


if __name__ == "__main__":
    main()
